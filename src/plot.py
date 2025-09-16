# src/plot.py
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# ---------------------------
# Theme & helpers
# ---------------------------
def set_theme():
    """
    Set consistent modern theme for all plots.
    """
    plt.style.use("default")
    sns.set_theme(context="notebook", style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
    })


def _save_figure(save: Optional[Union[str, Path]] = None):
    if save:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to: {save_path}")


# ---------------------------
# 1) 折线图：年度产出趋势（按类型）
# ---------------------------
def line_count_by_year(df: pd.DataFrame, save: Optional[Union[str, Path]] = None):
    """
    Line plot: number of titles by release_year × type.
    """
    set_theme()

    if "release_year" not in df.columns or "type" not in df.columns:
        print("Columns 'release_year' and 'type' are required.")
        return

    data = df.copy()
    data = data[(data["release_year"] >= 1950) & (data["release_year"] <= 2024)]

    year_counts = data.groupby(["release_year", "type"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 8))
    for col in year_counts.columns:
        ax.plot(
            year_counts.index,
            year_counts[col],
            marker="o",
            linewidth=2.2,
            markersize=4,
            label=col,
            alpha=0.9,
        )

    ax.set_title("Netflix Content Production Over Time", pad=18, fontweight="bold")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Number of Titles")

    # Peak annotation
    totals = year_counts.sum(axis=1)
    if not totals.empty:
        peak_year = totals.idxmax()
        peak_count = int(totals.max())
        ax.annotate(
            f"Peak: {peak_year}\n({peak_count:,} titles)",
            xy=(peak_year, peak_count),
            xytext=(peak_year - 5, peak_count * 1.05),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
            fontsize=10,
            ha="center",
        )

    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure(save)
    plt.show()


# ---------------------------
# 2) Top 国家（水平条形）
# ---------------------------
def bar_top_countries(
    df_exploded: pd.DataFrame,
    top: int = 10,
    save: Optional[Union[str, Path]] = None,
):
    """
    Horizontal bar chart of top countries by count.
    NOTE: 为避免“国家×类型”联合 explode 的重复计数，
          先 unique 到 (show_id, country) 再计数。
    """
    set_theme()

    if "country" not in df_exploded.columns:
        print("Column 'country' is required in exploded dataframe.")
        return

    dat = df_exploded[["show_id", "country"]].dropna().copy()
    dat["country"] = dat["country"].astype(str).str.strip()
    dat = dat[(dat["country"] != "") & (dat["country"].str.lower() != "nan")]
    dat = dat.drop_duplicates(subset=["show_id", "country"])

    country_counts = (
        dat.groupby("country").size().sort_values(ascending=False).head(top)
    )

    if country_counts.empty:
        print("No country data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(country_counts.index, country_counts.values, alpha=0.9)

    ax.set_title(f"Top {top} Countries by Netflix Content Count", pad=18, fontweight="bold")
    ax.set_xlabel("Number of Titles")

    for bar, value in zip(bars, country_counts.values):
        ax.text(
            value + max(country_counts.values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}",
            ha="left",
            va="center",
            fontsize=11,
        )

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save_figure(save)
    plt.show()


# ---------------------------
# 3) 电影时长分布（直方图 + 统计线）
# ---------------------------
def hist_movie_duration(
    df: pd.DataFrame, bins: int = 30, save: Optional[Union[str, Path]] = None
):
    """
    Histogram of movie duration distribution (minutes).
    """
    set_theme()

    if "duration_minutes" not in df.columns or "type" not in df.columns:
        print("Columns 'duration_minutes' and 'type' are required.")
        return

    movie_durations = df[(df["type"] == "Movie") & (df["duration_minutes"].notna())][
        "duration_minutes"
    ]

    if movie_durations.empty:
        print("No movie duration data available.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    n, b, _ = ax.hist(movie_durations, bins=bins, alpha=0.8, edgecolor="black", linewidth=0.5)

    mean_val = movie_durations.mean()
    median_val = movie_durations.median()

    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, alpha=0.9, label=f"Mean: {mean_val:.0f} min")
    ax.axvline(median_val, color="orange", linestyle="--", linewidth=2, alpha=0.9, label=f"Median: {median_val:.0f} min")

    ax.set_title("Distribution of Netflix Movie Durations", pad=18, fontweight="bold")
    ax.set_xlabel("Duration (Minutes)")
    ax.set_ylabel("Number of Movies")

    stats_text = (
        f"Total Movies: {len(movie_durations):,}\n"
        f"Mean: {mean_val:.0f} min\n"
        f"Median: {median_val:.0f} min\n"
        f"Std: {movie_durations.std():.0f} min"
    )
    ax.text(
        0.75, 0.75, stats_text, transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_figure(save)
    plt.show()


# ---------------------------
# 4) 箱线图 + 小提琴图：时长对比
# ---------------------------
def box_duration_by_type(df: pd.DataFrame, save: Optional[Union[str, Path]] = None):
    """
    Box/Violin comparison:
      - Movies use duration_minutes
      - TV Shows use seasons (标注单位不同，放并列子图)
    """
    set_theme()

    need_cols = {"type", "duration_minutes", "seasons"}
    if not need_cols.issubset(df.columns):
        print(f"Columns {need_cols} are required.")
        return

    movies = df[(df["type"] == "Movie") & (df["duration_minutes"].notna())]["duration_minutes"]
    shows = df[(df["type"] == "TV Show") & (df["seasons"].notna())]["seasons"]

    if movies.empty and shows.empty:
        print("No duration data available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Movies: minutes
    if not movies.empty:
        sns.boxplot(y=movies, ax=axes[0])
        sns.violinplot(y=movies, ax=axes[0], inner=None, color="lightblue", alpha=0.3)
        axes[0].set_title("Movies (Minutes)", fontweight="bold")
        axes[0].set_ylabel("Duration (Minutes)")
        axes[0].set_xlabel("")
    else:
        axes[0].set_visible(False)

    # TV Shows: seasons
    if not shows.empty:
        sns.boxplot(y=shows, ax=axes[1])
        sns.violinplot(y=shows, ax=axes[1], inner=None, color="lightgreen", alpha=0.3)
        axes[1].set_title("TV Shows (Seasons)", fontweight="bold")
        axes[1].set_ylabel("Seasons")
        axes[1].set_xlabel("")
    else:
        axes[1].set_visible(False)

    plt.suptitle("Duration Distribution by Content Type", fontsize=16, fontweight="bold", y=0.95)
    for ax in axes:
        if ax.get_visible():
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_figure(save)
    plt.show()


# ---------------------------
# 5) 热力图：年份 × 类型
# ---------------------------
def heatmap_content_by_year_type(df: pd.DataFrame, save: Optional[Union[str, Path]] = None):
    """
    Heatmap of counts by release_year × type.
    """
    set_theme()

    need_cols = {"release_year", "type"}
    if not need_cols.issubset(df.columns):
        print(f"Columns {need_cols} are required.")
        return

    data = df[(df["release_year"] >= 2000) & (df["release_year"] <= 2024)]
    pivot = data.groupby(["release_year", "type"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        pivot.T, annot=True, fmt="d", cmap="YlOrRd",
        cbar_kws={"label": "Number of Titles"}, ax=ax
    )

    ax.set_title("Netflix Content Heatmap: Year vs Content Type", pad=18, fontweight="bold")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Content Type")
    plt.tight_layout()
    _save_figure(save)
    plt.show()


# ---------------------------
# 6) 散点：电影时长随年份趋势
# ---------------------------
def scatter_duration_vs_year(df: pd.DataFrame, save: Optional[Union[str, Path]] = None):
    """
    Scatter of movie duration vs release_year, with trend line.
    """
    set_theme()

    need_cols = {"type", "release_year", "duration_minutes"}
    if not need_cols.issubset(df.columns):
        print(f"Columns {need_cols} are required.")
        return

    movies = df[(df["type"] == "Movie") & df["duration_minutes"].notna()].copy()
    movies = movies[(movies["release_year"] >= 1950) & (movies["release_year"] <= 2024)]

    if movies.empty:
        print("No movie data available for scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    sc = ax.scatter(
        movies["release_year"], movies["duration_minutes"],
        alpha=0.6, s=28, c=movies["release_year"],
        cmap="viridis", edgecolors="black", linewidth=0.3
    )

    # trend
    z = np.polyfit(movies["release_year"], movies["duration_minutes"], 1)
    p = np.poly1d(z)
    ax.plot(movies["release_year"], p(movies["release_year"]), "r--", alpha=0.85, linewidth=2)

    ax.set_xlabel("Release Year")
    ax.set_ylabel("Duration (Minutes)")
    ax.set_title("Netflix Movie Duration Trends Over Time", pad=18, fontweight="bold")

    cbar = plt.colorbar(sc)
    cbar.set_label("Release Year", rotation=270, labelpad=20)

    corr = movies["release_year"].corr(movies["duration_minutes"])
    stats_text = (
        f"Correlation: {corr:.3f}\n"
        f"Movies: {len(movies):,}\n"
        f"Trend: {'Increasing' if z[0] > 0 else 'Decreasing'}"
    )
    ax.text(
        0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure(save)
    plt.show()


# ---------------------------
# 7) Top 类型（Genres）
# ---------------------------
def bar_top_genres(
    df_mlb: pd.DataFrame, top: int = 15, save: Optional[Union[str, Path]] = None
):
    """
    Horizontal bar chart of top genres (from multi-hot columns).
    Fallback: 若没有 one-hot 列，则从 exploded 的 'listed_in' 构造（要求调用方传对的 df）。
    """
    set_theme()

    # Prefer one-hot columns
    genre_cols = [c for c in df_mlb.columns if c.startswith("listed_in_")]

    if genre_cols:
        counts = df_mlb[genre_cols].sum().sort_values(ascending=False).head(top)
        index_labels = counts.index.str.replace("listed_in_", "", regex=False)
    else:
        # Fallback: try to use raw 'listed_in' column
        if "listed_in" not in df_mlb.columns:
            print("No genre columns found (expected 'listed_in_*' or 'listed_in').")
            return
        series = (
            df_mlb["listed_in"]
            .fillna("")
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
        )
        series = series[(series != "") & (series.str.lower() != "nan")]
        counts = series.value_counts().head(top)
        index_labels = counts.index

    if counts.empty:
        print("No genre data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(range(len(counts)), counts.values, color=sns.color_palette("plasma", len(counts)))

    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(index_labels)
    ax.set_xlabel("Number of Titles")
    ax.set_title(f"Top {top} Netflix Genres by Content Count", pad=18, fontweight="bold")

    total_titles = len(df_mlb)
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_width() + max(counts.values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}",
            ha="left",
            va="center",
            fontsize=11,
        )
        ax.text(
            bar.get_width() * 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{(value / total_titles) * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save_figure(save)
    plt.show()


# ---------------------------
# Dashboard
# ---------------------------
def create_dashboard(
    df: pd.DataFrame,
    df_exploded: pd.DataFrame,
    df_mlb: pd.DataFrame,
    save_dir: Optional[Union[str, Path]] = None,
):
    """
    Generate all figures in one go.
    """
    print("Creating Netflix Data Dashboard...")

    save_paths = {}
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_paths = {
            "line": save_dir / "content_by_year.png",
            "countries": save_dir / "top_countries.png",
            "duration_hist": save_dir / "movie_duration_distribution.png",
            "duration_box": save_dir / "duration_by_type.png",
            "genres": save_dir / "top_genres.png",
            "heatmap": save_dir / "content_heatmap.png",
            "scatter": save_dir / "duration_vs_year.png",
        }

    print("1) Content count by year...")
    line_count_by_year(df, save=save_paths.get("line"))

    print("2) Top countries...")
    bar_top_countries(df_exploded, save=save_paths.get("countries"))

    print("3) Movie duration distribution...")
    hist_movie_duration(df, save=save_paths.get("duration_hist"))

    print("4) Duration by content type...")
    box_duration_by_type(df, save=save_paths.get("duration_box"))

    print("5) Top genres...")
    bar_top_genres(df_mlb, save=save_paths.get("genres"))

    print("6) Year × type heatmap...")
    heatmap_content_by_year_type(df, save=save_paths.get("heatmap"))

    print("7) Duration vs year scatter...")
    scatter_duration_vs_year(df, save=save_paths.get("scatter"))

    print("✅ Dashboard completed!")
    if save_dir:
        print(f"All figures saved to: {save_dir}")


if __name__ == "__main__":
    print("Netflix Visualization Module ready. Use create_dashboard() to render all.")
