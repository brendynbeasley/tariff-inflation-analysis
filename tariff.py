import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas_datareader.data as web


START_DATE = pd.Timestamp.today() - pd.DateOffset(months=27)
END_DATE   = pd.Timestamp.today()


ACTIVE_ERA = 'second_term'

INFLATION_PERIODS = 1

SMOOTHING_WINDOW = 1

AS_PERCENTAGE = False

EXPORT_CSV = True

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

SERIES_DICT = {
    'CPIAUCSL': 'All Items (Headline)',
    'CPILFESL': 'Core (ex. Food & Energy)',
    'CPIUFDSL': 'Food',
    'CPIENGSL': 'Energy',
}

TARIFF_DB = {
    'first_term': {
        'title': 'Trump 1.0 — 2018–2019 Trade War',
        'start': pd.Timestamp('2017-09-01'),
        'end'  : pd.Timestamp('2020-09-01'),
        'events': [
            {
                'date'     : pd.Timestamp('2018-02-07'),
                'label'    : 'Washing Machines\n& Solar (Sec 201)',
                'authority': 'Section 201',
                'scope_bn' : 8,
                'alpha'    : 0.6,
            },
            {
                'date'     : pd.Timestamp('2018-03-23'),
                'label'    : 'Steel 25%\nAluminum 10% (Sec 232)',
                'authority': 'Section 232',
                'scope_bn' : 46,
                'alpha'    : 0.8,
            },
            {
                'date'     : pd.Timestamp('2018-07-06'),
                'label'    : 'China Tranche 1\n($34B, List 1)',
                'authority': 'Section 301',
                'scope_bn' : 34,
                'alpha'    : 0.8,
            },
            {
                'date'     : pd.Timestamp('2018-09-24'),
                'label'    : 'China Tranche 3\n($200B, List 3)',
                'authority': 'Section 301',
                'scope_bn' : 200,
                'alpha'    : 0.95,
            },
        ]
    },
    'second_term': {
        'title': 'Trump 2.0 — 2025–2026 Tariffs',
        'start': START_DATE,
        'end'  : END_DATE,
        'events': [
            {
                'date'     : pd.Timestamp('2025-02-04'),
                'label'    : 'China +10% (IEEPA)\nCanada/Mexico 25%',
                'authority': 'IEEPA',
                'scope_bn' : 450,
                'alpha'    : 0.85,
            },
            {
                'date'     : pd.Timestamp('2025-03-04'),
                'label'    : 'Canada/Mexico\nFentanyl Tariffs',
                'authority': 'IEEPA',
                'scope_bn' : 950,
                'alpha'    : 0.9,
            },
            {
                'date'     : pd.Timestamp('2025-03-12'),
                'label'    : 'Steel/Aluminum\n25% Global (Sec 232)',
                'authority': 'Section 232',
                'scope_bn' : 110,
                'alpha'    : 0.8,
            },
            {
                'date'     : pd.Timestamp('2025-03-27'),
                'label'    : 'Venezuelan Oil\nSecondary Tariffs',
                'authority': 'IEEPA',
                'scope_bn' : 15,
                'alpha'    : 0.55,
            },
            {
                'date'     : pd.Timestamp('2025-04-02'),
                'label'    : '10% Global Baseline\n"Liberation Day" (IEEPA)',
                'authority': 'IEEPA',
                'scope_bn' : 2400,
                'alpha'    : 1.0,
            },
        ]
    }
}

AUTHORITY_COLORS = {
    'IEEPA'      : '#d62728',   # red  — emergency powers
    'Section 232': '#ff7f0e',   # orange — national security
    'Section 301': '#9467bd',   # purple — unfair trade practices
    'Section 201': '#17becf',   # teal  — safeguard / import relief
}

def _tariff_linewidth(scope_bn: float) -> float:
    return max(1.5, min(8.0, scope_bn / 80))

def fetch_fred_data(
    series: dict,
    start: pd.Timestamp,
    end: pd.Timestamp
) -> pd.DataFrame:
    print(f"[FRED] Fetching {list(series.keys())} from {start.date()} to {end.date()} ...")
    try:
        df = web.DataReader(list(series.keys()), 'fred', start, end)
        df.rename(columns=series, inplace=True)
        print(f"[FRED] Successfully retrieved {len(df)} monthly observations.")
        return df
    except Exception as exc:
        raise RuntimeError(
            "\n[ERROR] Could not fetch data from FRED.\n"
            "  • Check your internet connection.\n"
            "  • Verify that pandas-datareader is installed: pip install pandas-datareader\n"
            f"  • Technical detail: {exc}"
        ) from exc

def validate_era_dates(era_config: dict, df_index: pd.DatetimeIndex) -> None:
    """
    Warns if tariff event dates fall outside the fetched data window.
    This can happen if START_DATE is set too recently.
    """
    series_start = df_index.min()
    series_end   = df_index.max()
    for evt in era_config['events']:
        d = evt['date']
        if d < series_start or d > series_end:
            print(
                f"[WARN] Tariff '{evt['label'].strip()}' on {d.date()} is outside "
                f"the data window ({series_start.date()} – {series_end.date()}). "
                "It will be omitted from the plot."
            )

def calculate_inflation(
    df: pd.DataFrame,
    periods: int,
    as_percentage: bool
) -> pd.DataFrame:
    label = "MoM" if periods == 1 else ("YoY" if periods == 12 else f"{periods}-Month")
    if as_percentage:
        print(f"[CALC] {label} percentage change: (P_t/P_{{t-{periods}}} - 1) × 100")
        result = (df / df.shift(periods) - 1) * 100
    else:
        print(f"[CALC] {label} gross rate: P_t / P_{{t-{periods}}}")
        result = df / df.shift(periods)
    return result.dropna()

def apply_smoothing(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window > 1:
        print(f"[SMOOTH] Applying {window}-month rolling average.")
        return df.rolling(window=window, center=True).mean().dropna()
    return df

def plot_inflation(
    df: pd.DataFrame,
    era_config: dict,
    periods: int,
    smoothing: int,
    as_percentage: bool
) -> None:
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(bottom=0.38)

    series_colors = {
        'All Items (Headline)'    : '#1f77b4',
        'Core (ex. Food & Energy)': '#2ca02c',
        'Food'                    : '#ff7f0e',
        'Energy'                  : '#d62728',
    }
    for col, color in series_colors.items():
        if col in df.columns:
            ax.plot(df.index, df[col], label=col, color=color,
                    linewidth=2.2, zorder=3)

    # Y lim
    y_min, y_max = df.min().min(), df.max().max()
    pad = (y_max - y_min) * 0.12
    ax.set_ylim(y_min - pad, y_max + pad)

    era_start = df.index.min()
    era_end   = df.index.max()

    # Tariff vlines
    in_window = [e for e in era_config['events']
                 if era_start <= e['date'] <= era_end]

    CIRCLED = [f'[{i+1}]' for i in range(10)]
    NUDGE_DAYS = 6

    for i, evt in enumerate(in_window):
        color = AUTHORITY_COLORS.get(evt['authority'], 'black')
        lw    = _tariff_linewidth(evt['scope_bn'])
        num   = CIRCLED[i] if i < len(CIRCLED) else str(i + 1)

        ax.axvline(x=evt['date'], color=color, linewidth=lw,
                   alpha=evt['alpha'] * 0.7, linestyle='--', zorder=2)

        too_close = (i > 0 and
                     (evt['date'] - in_window[i-1]['date']).days < NUDGE_DAYS * 4)
        nudge = pd.Timedelta(days=NUDGE_DAYS if (i % 2 == 1 and too_close) else 0)

        ax.text(
            evt['date'] + nudge, ax.get_ylim()[1] * 0.9995,
            num,
            color=color, fontsize=11, fontweight='bold',
            ha='center', va='top', zorder=5,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor=color,
                      linewidth=0.8, boxstyle='round,pad=0.2')
        )

    if not as_percentage:
        ax.axhline(y=1.0, color='gray', linewidth=1.0,
                   linestyle=':', alpha=0.7, zorder=1)
        ax.text(df.index[-1], 1.0005, 'No change (1.00)',
                color='gray', fontsize=8, ha='right', va='bottom')

    # ── Legend ────────────────────────────────────────────────
    series_handles = [
        mpatches.Patch(color=c, label=l)
        for l, c in series_colors.items() if l in df.columns
    ]
    ax.legend(handles=series_handles, loc='upper left',
              frameon=True, facecolor='white', framealpha=0.9,
              title='CPI Series', fontsize=10)

    calc_label  = "MoM" if periods == 1 else ("YoY" if periods == 12 else f"{periods}-Mo")
    smooth_text = f"  |  {smoothing}-Month Smoothed" if smoothing > 1 else ""
    y_label     = (
        "Percentage Change (%)" if as_percentage
        else f"Gross Rate  ( P$_t$ / P$_{{t-{periods}}}$ )"
    )
    ax.set_title(
        f"Sectoral Inflation Dynamics: {era_config['title']}\n"
        f"Calculation: {calc_label} Gross Rate{smooth_text}",
        fontsize=13, fontweight='bold', pad=12
    )
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=11, fontweight='bold')

    table_data = [['#', 'Date', 'Authority', 'Description']]
    for i, evt in enumerate(in_window):
        num   = CIRCLED[i] if i < len(CIRCLED) else str(i + 1)
        label = evt['label'].replace('\n', ' ')
        table_data.append([
            num,
            evt['date'].strftime('%b %-d, %Y'),
            evt['authority'],
            label
        ])

    col_widths = [0.04, 0.10, 0.12, 0.52]

    the_table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        colWidths=col_widths,
        loc='bottom',
        bbox=[0.0, -0.52, 1.0, 0.44]
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8.5)

    for col in range(len(col_widths)):
        cell = the_table[0, col]
        cell.set_facecolor('#dde3ea')
        cell.set_text_props(fontweight='bold')
        if col == 3:
            cell.get_text().set_ha('left')

    for row_i, evt in enumerate(in_window, start=1):
        color = AUTHORITY_COLORS.get(evt['authority'], 'black')
        bg    = '#f9f9f9' if row_i % 2 == 0 else 'white'
        for col in range(len(col_widths)):
            cell = the_table[row_i, col]
            cell.set_facecolor(bg)
            cell.set_edgecolor('#dddddd')
            cell.get_text().set_ha('left' if col == 3 else 'center')
        the_table[row_i, 0].set_facecolor(color + '22')

    authority_handles = [
        mpatches.Patch(color=AUTHORITY_COLORS[a], label=f'{a}  (line color)')
        for a in AUTHORITY_COLORS
        if any(e['authority'] == a for e in in_window)
    ]
    fig.legend(
        handles=authority_handles,
        loc='lower right',
        bbox_to_anchor=(0.98, 0.01),
        frameon=True, facecolor='white', framealpha=0.9,
        title='Tariff Authority  |  line weight ∝ import exposure ($B)',
        fontsize=8.5, ncol=len(authority_handles)
    )

    out_path = os.path.join(OUTPUT_DIR, f"inflation_plot_{ACTIVE_ERA}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[PLOT] Figure saved to {out_path}")
    plt.show()

def export_csv(df: pd.DataFrame, suffix: str) -> None:
    """Saves the processed inflation DataFrame to a timestamped CSV."""
    filename = f"processed_inflation_{suffix}.csv"
    path     = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path)
    print(f"[CSV]  Data exported to {path}")

def main() -> None:
    assert ACTIVE_ERA in TARIFF_DB, (
        f"ACTIVE_ERA='{ACTIVE_ERA}' is not defined. "
        f"Choose from: {list(TARIFF_DB.keys())}"
    )
    assert INFLATION_PERIODS in (1, 3, 6, 12), (
        "INFLATION_PERIODS must be 1 (MoM), 3, 6, or 12 (YoY)."
    )
    assert 1 <= SMOOTHING_WINDOW <= 12, (
        "SMOOTHING_WINDOW must be between 1 (no smoothing) and 12."
    )

    config = TARIFF_DB[ACTIVE_ERA]

    raw_df      = fetch_fred_data(SERIES_DICT, config['start'], config['end'])

    validate_era_dates(config, raw_df.index)

    inflation_df = calculate_inflation(raw_df, INFLATION_PERIODS, AS_PERCENTAGE)

    inflation_df = inflation_df[inflation_df.index >= (START_DATE + pd.DateOffset(months=1))]

    final_df     = apply_smoothing(inflation_df, SMOOTHING_WINDOW)

    if EXPORT_CSV:
        export_csv(final_df, ACTIVE_ERA)

    plot_inflation(final_df, config, INFLATION_PERIODS, SMOOTHING_WINDOW, AS_PERCENTAGE)

    print("\n[DONE] Analysis complete.")

if __name__ == "__main__":
    main()