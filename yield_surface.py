"""
3-D U.S. Treasury Yield-Curve Surface
=====================================
A 3-D view of the U.S. Treasury yield curve and how its shape evolves over time,
built from live FRED data. Inspired by the New York Times' 2015 "A 3-D View of a
Chart That Predicts the Economic Future: The Yield Curve."

Axes (term structure of interest rates):
  * x-axis  : calendar time          (the curve evolving year by year)
  * y-axis  : maturity / tenor        (6-mo ... 10-yr  -> the "yield curve")
  * z-axis  : yield to maturity (%)   (the height of the surface)

Each front-to-back slice is one yield curve. Left-to-right shows how its shape
changes: upward-sloping (normal) -> flat/humped -> inverted (short > long, the
classic recession signal: 1990, 2000, 2006-07, 2019, and the deep 2022-24 one).

DATA:
  Real Treasury Constant-Maturity yields straight from FRED's CSV endpoint -
  no API key, no pandas-datareader, just an internet connection. By default
  `end` is today, so the surface always extends to the most recent data.
  Offline, it falls back to a Nelson-Siegel synthetic surface.

OUTPUT:
  yield_surface.png   - static matplotlib figure (also shown in Spyder's pane)
  yield_surface.html  - interactive, rotatable Plotly figure (open in a browser)

Usage:
  python yield_surface.py
"""

from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def _date_to_frac(d):
    """'YYYY-MM-DD' -> decimal year (e.g. 2026-06-xx -> 2026.42)."""
    y, m, _ = (int(p) for p in d.split("-"))
    return y + (m - 1) / 12.0


# ----------------------------------------------------------------------------
# 1. REAL DATA FROM FRED  (no API key, no pandas-datareader - just internet)
# ----------------------------------------------------------------------------
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# tenor (years) -> FRED series id. These seven have continuous data from 1990
# to today (hence 6-month as the short end; 3-month starts only in 2001).
FRED_SERIES = [
    (0.5, "DGS6MO"), (1, "DGS1"), (2, "DGS2"), (3, "DGS3"),
    (5, "DGS5"), (7, "DGS7"), (10, "DGS10"),
]


def _fred_series(series_id, start, end):
    """Download one FRED series as a date-indexed pandas Series."""
    df = pd.read_csv(FRED_CSV.format(sid=series_id), na_values=".")  # FRED uses '.'
    df.columns = ["date", series_id]            # positional rename (date col varies)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[series_id].loc[start:end]


def fred_surface(start="1990-01-01", end=None):
    """
    Build the surface from real Treasury yields on FRED.
    `end=None` means today. Returns (dates, maturities, Z) or None on failure.
    """
    if end is None:
        end = date.today().isoformat()
    try:
        cols = {sid: _fred_series(sid, start, end) for _, sid in FRED_SERIES}
    except Exception as exc:
        print(f"  FRED download failed ({exc}); falling back to synthetic.")
        return None

    frame = pd.DataFrame(cols)                    # columns = series ids (strings)
    frame = frame.resample("MS").mean().interpolate().dropna()  # daily -> monthly

    maturities = np.array([m for m, _ in FRED_SERIES], dtype=float)
    ordered_ids = [sid for _, sid in FRED_SERIES]
    Z = frame[ordered_ids].to_numpy()             # order columns by maturity
    dates = (frame.index.year + (frame.index.month - 1) / 12.0).to_numpy()
    return dates, maturities, Z


# ----------------------------------------------------------------------------
# 2. NELSON-SIEGEL MODEL  (offline fallback only, anchored through 2026)
# ----------------------------------------------------------------------------
def nelson_siegel(tau, beta0, beta1, beta2, lam):
    """beta0=LEVEL, beta1=SLOPE (neg=>upward), beta2=CURVATURE, lam=decay."""
    x = tau / lam
    loading_slope = (1.0 - np.exp(-x)) / x
    loading_curve = loading_slope - np.exp(-x)
    return beta0 + beta1 * loading_slope + beta2 * loading_curve


def synthetic_surface(start="1990-01-01", end=None):
    """Realistic-looking surface anchored to U.S. rate history, start -> end."""
    start_frac = _date_to_frac(start)
    end_frac = _date_to_frac(date.today().isoformat() if end is None else end)
    anchors = {                                   # date -> (level, slope, curvature)
        1990.0: (8.2, -0.5, -1.0), 1993.0: (6.0, -2.5, -1.0),
        1995.0: (6.5,  0.3,  0.5), 1998.0: (5.3, -0.6, -0.5),
        2000.5: (6.2,  0.9,  0.8),  # inversion -> 2001 recession
        2003.0: (4.3, -3.0, -1.5), 2006.5: (5.0,  0.4,  0.6),  # inversion -> GFC
        2009.0: (3.6, -3.6, -1.0),  # ZIRP begins, very steep
        2012.0: (2.3, -2.1, -0.8), 2015.0: (2.6, -2.4, -0.7),
        2016.5: (2.0, -1.6, -0.6),  # low, gradual hikes
        2018.5: (3.0, -0.6, -0.3),  # hiking cycle peak, curve flat
        2019.6: (1.9,  0.2,  0.0),  # mid-2019 mild inversion
        2020.4: (0.9, -0.8, -0.4),  # COVID crash, near-zero short rates
        2021.5: (1.5, -1.5, -0.5),  # reflation, steep curve
        2022.7: (3.4,  0.3, -0.2),  # rapid hikes flip curve toward inversion
        2023.5: (4.0,  1.5, -0.3),  # DEEP inversion (6-mo ~5.5%)
        2024.4: (4.2,  1.0, -0.3),  # still inverted, cuts begin
        2025.3: (4.3, -0.1, -0.1),  # un-inverting, roughly flat
        2026.4: (4.2, -0.5, -0.1),  # normalizing, mild upward slope
    }
    a_dates = np.array(sorted(anchors))
    a_b0 = np.array([anchors[d][0] for d in a_dates])
    a_b1 = np.array([anchors[d][1] for d in a_dates])
    a_b2 = np.array([anchors[d][2] for d in a_dates])

    dates = np.arange(start_frac, end_frac + 1e-9, 1 / 12)
    B0 = np.interp(dates, a_dates, a_b0)
    B1 = np.interp(dates, a_dates, a_b1)
    B2 = np.interp(dates, a_dates, a_b2)

    rng = np.random.default_rng(7)
    B0 = B0 + np.cumsum(rng.normal(0, 0.03, len(dates))) * 0.3
    B1 = B1 + rng.normal(0, 0.08, len(dates))

    maturities = np.array([0.5, 1, 2, 3, 5, 7, 10])
    Z = np.vstack([nelson_siegel(maturities, B0[i], B1[i], B2[i], 1.8)
                   for i in range(len(dates))])
    return dates, maturities, np.clip(Z, 0.02, None)


def load_data(use_real=True, start="1990-01-01", end=None):
    if use_real:
        out = fred_surface(start, end)
        if out is not None:
            print("  Loaded REAL Treasury yields from FRED.")
            return out
    print("  Using SYNTHETIC Nelson-Siegel surface.")
    return synthetic_surface(start, end)


# ----------------------------------------------------------------------------
# 3. STATIC NYT-STYLE RENDER  -> Spyder Plots pane + saved PNG
# ----------------------------------------------------------------------------
# Red (high yield) <-> blue (low yield), matching the Greeks plot.
# coolwarm's centre is light grey (not pure white), so the surface stays
# visible against a pale background. Swap to "RdBu_r" or "seismic" for punchier
# ends, or back to a single-hue "Blues" for the original NYT look.
CMAP = "coolwarm"


def plot_static(dates, maturities, Z, fname="yield_surface.png"):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    T, M = np.meshgrid(dates, maturities, indexing="ij")
    zmax = max(2.0, float(np.ceil(Z.max() / 2) * 2))
    norm = Normalize(vmin=float(Z.min()), vmax=float(Z.max()))

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    ax = fig.add_axes([0.0, 0.04, 0.80, 0.92], projection="3d", facecolor="white")

    # main surface
    surf = ax.plot_surface(T, M, Z, cmap=CMAP, norm=norm,
                           rcount=Z.shape[0], ccount=Z.shape[1],
                           linewidth=0, antialiased=True, shade=True)
    # black front edge along the shortest maturity, like the NYT chart
    ax.plot(dates, np.full_like(dates, maturities[0]), Z[:, 0],
            color="black", lw=0.8, zorder=10)

    # framing: zoom fills the frame and trims the empty "sky"
    ax.set_box_aspect((2.4, 1.0, 0.55), zoom=1.12)
    ax.view_init(elev=27, azim=-58)

    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(maturities.min(), maturities.max())
    ax.set_zlim(0, zmax)

    # adaptive year ticks (every 2 yrs for short spans, every 4 when long)
    yr0 = int(np.ceil(dates.min() / 2) * 2)
    yr1 = int(np.floor(dates.max()))
    step = 2 if (yr1 - yr0) <= 28 else 4
    xticks = list(range(yr0, yr1 + 1, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"'{str(y)[2:]}" for y in xticks], fontsize=12)

    lab = {0.5: "6-mo", 1: "1-yr", 2: "2-yr", 3: "3-yr",
           5: "5-yr", 7: "7-yr", 10: "10-yr"}
    show = [0.5, 2, 5, 10]
    ax.set_yticks(show)
    ax.set_yticklabels([lab[m] for m in show], fontsize=12)
    ax.set_zticks(range(0, int(zmax) + 1, 2))
    ax.tick_params(axis="z", labelsize=12)

    ax.set_xlabel("Year", fontsize=15, labelpad=18)
    ax.set_ylabel("Maturity", fontsize=15, labelpad=18)
    # z-axis kept for height reference; its label is dropped (the colorbar is
    # already labelled "Yield (%)", so a second one would just crowd the corner)

    ax.grid(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.set_title(f"U.S. Treasury Yield Curve ({int(dates.min())}\u2013{int(dates.max())})",
                 fontsize=18, fontweight="bold", y=0.99)

    # yield colorbar in its OWN axis on the far right (never overlaps the plot)
    sm = ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.88, 0.27, 0.018, 0.46])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Yield (%)", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.savefig(fname, dpi=130, facecolor="white")
    print(f"  Saved {fname}")
    plt.show()          # <-- renders the surface in Spyder's Plots pane
    return fig


# ----------------------------------------------------------------------------
# 4. INTERACTIVE RENDER  (Plotly -> opens in a browser, NOT the Plots pane)
# ----------------------------------------------------------------------------
# Red (high yield) <-> blue (low yield), coolwarm endpoints. The centre is a
# light GREY (#dddddd), deliberately darker than the pale background panel, so
# even mid-range yields stay visible from every camera angle.
RB_SCALE = [[0.00, "#3b4cc0"], [0.25, "#8db0fe"], [0.50, "#dddddd"],
            [0.75, "#f49a7b"], [1.00, "#b40426"]]


def plot_interactive(dates, maturities, Z, fname="yield_surface.html"):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not installed; skipping interactive version.")
        return

    pane = "rgb(238,243,250)"      # soft panel colour for contrast

    # Real datetime on the x-axis (rebuilt from the decimal years). Plotly then
    # formats the hover date natively via %{x|%b %Y} -> "May 2026", and the axis
    # shows real dates. Surface hover ignores customdata/text, so this is the
    # reliable way to get a readable date in the tooltip.
    yrs = np.floor(dates).astype(int)
    mos = np.clip(np.round((dates - yrs) * 12).astype(int) + 1, 1, 12)
    x_dates = pd.to_datetime(dict(year=yrs, month=mos,
                                  day=np.ones(len(dates), int))).to_numpy()

    fig = go.Figure(go.Surface(
        x=x_dates, y=maturities, z=Z.T,
        colorscale=RB_SCALE, cmin=float(Z.min()), cmax=float(Z.max()),
        # hover shows named axes (Date / Maturity / Yield) instead of x / y / z
        hovertemplate=("Date: %{x|%b %Y}<br>"
                       "Maturity: %{y} yr<br>"
                       "Yield: %{z:.2f}%<extra></extra>"),
        # shorter, slimmer colorbar centred vertically (was full height)
        colorbar=dict(title="Yield %", len=0.5, thickness=16,
                      y=0.5, yanchor="middle"),
        # NO contour lines drawn on the surface (clean faces)
        contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        # flat, matte lighting (high ambient, low specular) so NO camera angle
        # leaves the surface dark or washed out
        lighting=dict(ambient=0.85, diffuse=0.55, specular=0.08,
                      roughness=0.95, fresnel=0.05),
        lightposition=dict(x=0, y=0, z=10000),
    ))
    grid = "rgb(223,227,235)"      # soft grey walls grid (not stark white)
    fig.update_layout(
        title=dict(
            text=f"U.S. Treasury Yield Curve Surface ({int(dates.min())}-{int(dates.max())})",
            subtitle=dict(
                text="Post-2022: higher front-end yields changed the liquidity regime",
                font=dict(size=14, color="#5a6b7b")),
            font=dict(size=24, color="#1a2530"),
            x=0.5, xanchor="center", y=0.97, yanchor="top",
        ),
        paper_bgcolor="white",
        scene=dict(
            # scene fills the width; leave a sliver at the bottom for the source
            domain=dict(x=[0, 1], y=[0.05, 1]),
            xaxis=dict(title="Year", backgroundcolor=pane,
                       gridcolor=grid, showbackground=True, zeroline=False,
                       tickformat="%Y", dtick="M24"),
            yaxis=dict(title="Maturity (yrs)", backgroundcolor=pane,
                       gridcolor=grid, showbackground=True, zeroline=False),
            zaxis=dict(title="Yield %", backgroundcolor=pane,
                       gridcolor=grid, showbackground=True, zeroline=False),
            aspectratio=dict(x=1.8, y=1.0, z=0.6),
            # camera closer in so the surface fills the frame (was pulled back,
            # which created the empty space). aspect x=1.8 keeps clipping in check.
            camera=dict(eye=dict(x=-1.95, y=-1.45, z=0.85),
                        center=dict(x=0, y=0, z=-0.12)),
        ),
        annotations=[dict(
            text="Source: U.S. Treasury/FRED data, 2016\u20132026. "
                 "Visualization by author.",
            x=0.5, y=0.0, xref="paper", yref="paper", xanchor="center",
            yanchor="bottom", showarrow=False, font=dict(size=11, color="#7a8896")),
        ],
        margin=dict(l=0, r=0, t=64, b=12),
    )
    fig.write_html(fname, config={"displayModeBar": False})
    print(f"  Saved {fname}  (open in a browser to rotate/zoom)")


# ----------------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # 10-year window, 2016 -> today's most recent FRED data.
    # For the full history use start="1990-01-01"; pin the end with e.g.
    # end="2015-12-31" to reproduce the original NYT frame.
    dates, maturities, Z = load_data(use_real=True, start="2016-01-01", end=None)

    print(f"  Grid: {Z.shape[0]} dates x {Z.shape[1]} maturities, "
          f"{int(dates.min())}-{int(dates.max())}, "
          f"yields {Z.min():.2f}%-{Z.max():.2f}%")

    plot_static(dates, maturities, Z)
    plot_interactive(dates, maturities, Z)
    print("Done.")
