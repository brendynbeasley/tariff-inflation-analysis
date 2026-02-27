using Dates
using DataFrames
using CSV
using HTTP
using Statistics
using PyPlot
using PyCall

@pyimport datetime as pydt

const START_DATE        = today() - Month(27)
const END_DATE          = today()
const ACTIVE_ERA        = "second_term"
const INFLATION_PERIODS = 1
const SMOOTHING_WINDOW  = 1
const AS_PERCENTAGE     = false
const EXPORT_CSV        = true
const OUTPUT_DIR        = @__DIR__

const SERIES_DICT = Dict(
    "CPIAUCSL" => "All Items (Headline)",
    "CPILFESL" => "Core (ex. Food & Energy)",
    "CPIUFDSL" => "Food",
    "CPIENGSL" => "Energy",
)

struct TariffEvent
    date      :: Date
    label     :: String
    authority :: String
    scope_bn  :: Float64
    alpha     :: Float64
end

struct EraConfig
    title  :: String
    start  :: Date
    stop   :: Date
    events :: Vector{TariffEvent}
end

const TARIFF_DB = Dict(
    "first_term" => EraConfig(
        "Trump 1.0 — 2018–2019 Trade War",
        Date(2017, 9, 1),
        Date(2020, 9, 1),
        [
            TariffEvent(Date(2018,2,7),  "Washing Machines & Solar (Sec 201)",    "Section 201",   8.0, 0.60),
            TariffEvent(Date(2018,3,23), "Steel 25% / Aluminum 10% (Sec 232)",    "Section 232",  46.0, 0.80),
            TariffEvent(Date(2018,7,6),  "China Tranche 1 (\$34B, List 1)",       "Section 301",  34.0, 0.80),
            TariffEvent(Date(2018,9,24), "China Tranche 3 (\$200B, List 3)",      "Section 301", 200.0, 0.95),
        ]
    ),
    "second_term" => EraConfig(
        "Trump 2.0 — 2025–2026 Tariffs",
        START_DATE,
        END_DATE,
        [
            TariffEvent(Date(2025,2,4),  "China +10% (IEEPA) / Canada & Mexico 25%",      "IEEPA",        450.0, 0.85),
            TariffEvent(Date(2025,3,4),  "Canada/Mexico Fentanyl Tariffs",                 "IEEPA",        950.0, 0.90),
            TariffEvent(Date(2025,3,12), "Steel/Aluminum 25% Global (Sec 232)",            "Section 232",  110.0, 0.80),
            TariffEvent(Date(2025,3,27), "Venezuelan Oil Secondary Tariffs",               "IEEPA",         15.0, 0.55),
            TariffEvent(Date(2025,4,2),  "10% Global Baseline \"Liberation Day\" (IEEPA)", "IEEPA",       2400.0, 1.00),
        ]
    ),
)

const AUTHORITY_COLORS = Dict(
    "IEEPA"       => "#d62728",
    "Section 232" => "#ff7f0e",
    "Section 301" => "#9467bd",
    "Section 201" => "#17becf",
)

to_pydate(d::Date)       = pydt.date(Dates.year(d), Dates.month(d), Dates.day(d))
to_timedelta(days::Int)  = pydt.timedelta(days=days)
tariff_linewidth(s::Float64) = clamp(s / 80.0, 1.5, 8.0)

function fetch_fred_data(series::Dict{String,String}, start_date::Date, end_date::Date)::DataFrame
    println("[FRED] Fetching $(collect(keys(series))) from $start_date to $end_date ...")
    combined = nothing
    for (series_id, col_name) in series
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv" *
              "?id=$(series_id)&vintage_date=$(Dates.format(end_date, "yyyy-mm-dd"))"
        try
            resp = HTTP.get(url; connect_timeout=15, readtimeout=30)
            df   = CSV.read(IOBuffer(String(resp.body)), DataFrame;
                            header=["date", col_name],
                            skipto=2,
                            dateformat="yyyy-mm-dd",
                            types=Dict("date" => Date, col_name => Float64),
                            missingstring=".")
            dropmissing!(df)
            filter!(r -> start_date <= r.date <= end_date, df)
            combined = combined === nothing ? df : innerjoin(combined, df, on=:date)
        catch e
            throw(ErrorException("[ERROR] Could not fetch $series_id: $e"))
        end
    end
    sort!(combined, :date)
    println("[FRED] Retrieved $(nrow(combined)) monthly observations.")
    return combined
end

function validate_era_dates(config::EraConfig, df_dates::Vector{Date})
    s, e = minimum(df_dates), maximum(df_dates)
    for evt in config.events
        (evt.date < s || evt.date > e) && @warn "Tariff '$(evt.label)' on $(evt.date) is outside ($s – $e). Omitting."
    end
end

function calculate_inflation(df::DataFrame, periods::Int, as_percentage::Bool)::DataFrame
    series_cols = names(df, Not(:date))
    result = DataFrame(:date => df.date[periods+1:end])
    for col in series_cols
        vals = df[!, col]
        result[!, col] = as_percentage ?
            [(vals[i] / vals[i-periods] - 1.0) * 100.0 for i in (periods+1):length(vals)] :
            [vals[i] / vals[i-periods]                  for i in (periods+1):length(vals)]
    end
    return result
end

function apply_smoothing(df::DataFrame, window::Int)::DataFrame
    window == 1 && return df
    series_cols = names(df, Not(:date))
    half   = window ÷ 2
    n      = nrow(df)
    result = DataFrame(:date => df.date[half+1:n-half])
    for col in series_cols
        vals = df[!, col]
        result[!, col] = [mean(vals[i-half:i+half]) for i in (half+1):(n-half)]
    end
    return result
end

function plot_inflation(df::DataFrame, config::EraConfig, periods::Int, smoothing::Int, as_percentage::Bool)
    mpatches = pyimport("matplotlib.patches")

    series_colors = Dict(
        "All Items (Headline)"     => "#1f77b4",
        "Core (ex. Food & Energy)" => "#2ca02c",
        "Food"                     => "#ff7f0e",
        "Energy"                   => "#d62728",
    )

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(bottom=0.38)

    py_dates = to_pydate.(df.date)

    for (col, color) in series_colors
        col in names(df) || continue
        ax.plot(py_dates, df[!, col]; label=col, color=color, linewidth=2.2, zorder=3)
    end

    all_vals = vcat([df[!, c] for c in names(df, Not(:date))]...)
    y_min, y_max = minimum(all_vals), maximum(all_vals)
    pad = (y_max - y_min) * 0.12
    ax.set_ylim(y_min - pad, y_max + pad)

    era_start = minimum(df.date)
    era_end   = maximum(df.date)
    in_window = filter(e -> era_start <= e.date <= era_end, config.events)
    nudge_days = 6

    for (i, evt) in enumerate(in_window)
        c   = get(AUTHORITY_COLORS, evt.authority, "black")
        lw  = tariff_linewidth(evt.scope_bn)
        num = "[$i]"
        
        # Plot the vertical line at the exact date
        ax.axvline(x=to_pydate(evt.date); color=c, linewidth=lw, alpha=evt.alpha * 0.7, linestyle="--", zorder=2)
        
        # Calculate the nudge natively in Julia
        too_close = i > 1 && (evt.date - in_window[i-1].date).value < nudge_days * 4
        nudge_val = (i % 2 == 0 && too_close) ? nudge_days : 0
        
        # Add the days in Julia first, THEN convert to Python date
        d_nudged = to_pydate(evt.date + Day(nudge_val))
        
        ax.text(d_nudged, ax.get_ylim()[2] * 0.9995, num;
                color=c, fontsize=11, fontweight="bold", ha="center", va="top", zorder=5,
                bbox=Dict("facecolor"=>"white","alpha"=>0.85,"edgecolor"=>c,"linewidth"=>0.8,"boxstyle"=>"round,pad=0.2"))
    end

    if !as_percentage
        ax.axhline(y=1.0; color="gray", linewidth=1.0, linestyle=":", alpha=0.7, zorder=1)
        ax.text(py_dates[end], 1.0005, "No change (1.00)"; color="gray", fontsize=8, ha="right", va="bottom")
    end

    series_handles = [mpatches.Patch(color=c, label=l) for (l, c) in series_colors if l in names(df)]
    ax.legend(handles=series_handles; loc="upper left", frameon=true, facecolor="white",
              framealpha=0.9, title="CPI Series", fontsize=10)

    calc_label  = periods == 1 ? "MoM" : periods == 12 ? "YoY" : "$(periods)-Mo"
    smooth_text = smoothing > 1 ? "  |  $(smoothing)-Month Smoothed" : ""
    y_label     = as_percentage ? "Percentage Change (%)" : "Gross Rate  ( P\$_t\$ / P\$_{t-$periods}\$ )"

    ax.set_title("Sectoral Inflation Dynamics: $(config.title)\nCalculation: $calc_label Gross Rate$smooth_text";
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date"; fontsize=11, fontweight="bold")
    ax.set_ylabel(y_label; fontsize=11, fontweight="bold")

    table_data  = [["#", "Date", "Authority", "Description"]]
    for (i, evt) in enumerate(in_window)
        push!(table_data, ["[$i]", Dates.format(evt.date, "u d, yyyy"), evt.authority,
                           replace(evt.label, "\n" => " ")])
    end

    col_widths = [0.04, 0.10, 0.12, 0.52]
    cell_text  = permutedims(hcat(table_data[2:end]...))

    the_table = ax.table(;
        cellText=cell_text, colLabels=table_data[1],
        colWidths=col_widths, loc="bottom",
        bbox=[0.0, -0.52, 1.0, 0.44])
    the_table.auto_set_font_size(false)
    the_table.set_fontsize(8.5)

    # Shift indices up by 1 for PyCall compatibility
    for col in 1:length(col_widths)
        cell = the_table[1, col] # PyCall translates 1 -> 0 for Python
        cell.set_facecolor("#dde3ea")
        cell.set_text_props(fontweight="bold")
        col == 4 && cell.get_text().set_ha("left") # Column 4 is description
    end

    for (row_i, evt) in enumerate(in_window)
        py_row = row_i + 1 # Row 1 is the header, so data starts at Row 2
        c  = get(AUTHORITY_COLORS, evt.authority, "black")
        bg = row_i % 2 == 0 ? "#f9f9f9" : "white"
        
        for col in 1:length(col_widths)
            cell = the_table[py_row, col]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#dddddd")
            cell.get_text().set_ha(col == 4 ? "left" : "center")
        end
        the_table[py_row, 1].set_facecolor("$(c)22") # Tint the number cell
    end

    authority_handles = [
        mpatches.Patch(color=AUTHORITY_COLORS[a], label="$a  (line color)")
        for a in keys(AUTHORITY_COLORS) if any(e.authority == a for e in in_window)
    ]
    fig.legend(handles=authority_handles; loc="lower right", bbox_to_anchor=(0.98, 0.01),
               frameon=true, facecolor="white", framealpha=0.9,
               title="Tariff Authority  |  line weight ∝ import exposure (\$B)",
               fontsize=8.5, ncol=length(authority_handles))

    out_path = joinpath(OUTPUT_DIR, "inflation_plot_$(ACTIVE_ERA).png")
    plt.savefig(out_path; dpi=150, bbox_inches="tight")
    println("[PLOT] Figure saved to $out_path")
    plt.show()
end

function export_csv(df::DataFrame, suffix::String)
    path = joinpath(OUTPUT_DIR, "processed_inflation_$(suffix).csv")
    CSV.write(path, df)
    println("[CSV]  Data exported to $path")
end

function main()
    @assert ACTIVE_ERA in keys(TARIFF_DB) "ACTIVE_ERA='$ACTIVE_ERA' not defined. Choose from: $(collect(keys(TARIFF_DB)))"
    @assert INFLATION_PERIODS in (1, 3, 6, 12) "INFLATION_PERIODS must be 1, 3, 6, or 12."
    @assert 1 <= SMOOTHING_WINDOW <= 12 "SMOOTHING_WINDOW must be between 1 and 12."
    @assert isodd(SMOOTHING_WINDOW) "SMOOTHING_WINDOW must be odd (1, 3, 5…) for centered rolling average."

    config       = TARIFF_DB[ACTIVE_ERA]
    raw_df       = fetch_fred_data(SERIES_DICT, config.start, config.stop)
    validate_era_dates(config, raw_df.date)
    inflation_df = calculate_inflation(raw_df, INFLATION_PERIODS, AS_PERCENTAGE)
    filter!(r -> r.date >= START_DATE + Month(1), inflation_df)
    final_df     = apply_smoothing(inflation_df, SMOOTHING_WINDOW)
    EXPORT_CSV && export_csv(final_df, ACTIVE_ERA)
    plot_inflation(final_df, config, INFLATION_PERIODS, SMOOTHING_WINDOW, AS_PERCENTAGE)
    println("\n[DONE] Analysis complete.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end