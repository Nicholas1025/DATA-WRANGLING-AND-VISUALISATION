# src/main.py
import argparse
from pathlib import Path
from src.clean import clean_pipeline
from src.plot import (create_dashboard, line_count_by_year, bar_top_countries,
                      hist_movie_duration, box_duration_by_type,
                      bar_top_genres, heatmap_content_by_year_type, scatter_duration_vs_year)
from src.eda import basic_overview, yearly_counts, top_countries, duration_stats_movies, top_genres
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=["clean","eda","viz","all"], default="all")
    p.add_argument("--raw", default="data/raw/netflix_titles.csv")
    p.add_argument("--out", default="data/processed/netflix_clean.csv")
    p.add_argument("--figdir", default="figures")
    args = p.parse_args()

    results = {}
    if args.step in ("clean","all"):
        results = clean_pipeline(Path(args.raw), Path(args.out))

    # read back if running only eda/viz
    if args.step in ("eda","viz") and not results:
        results = {
            "clean": pd.read_csv("data/processed/netflix_clean.csv"),
            "exploded": pd.read_csv("data/processed/netflix_exploded.csv"),
            "mlb": pd.read_csv("data/processed/netflix_mlb.csv"),
        }

    if args.step in ("eda","all"):
        df = results["clean"]
        df_expl = results["exploded"]
        df_mlb = results["mlb"]

        ov = basic_overview(df); ov.to_csv("data/processed/eda_overview.csv", index=False)
        yc = yearly_counts(df); yc.to_csv("data/processed/eda_yearly_counts.csv", index=False)
        tc = top_countries(df_expl, n=10); tc.to_csv("data/processed/eda_top_countries.csv", index=False)
        dm = duration_stats_movies(df); dm.to_csv("data/processed/eda_duration_stats_movies.csv")
        tg = top_genres(df_mlb, n=15); tg.to_csv("data/processed/eda_top_genres.csv", index=False)
        print("EDA tables saved under data/processed/")

    if args.step in ("viz","all"):
        create_dashboard(results["clean"], results["exploded"], results["mlb"], save_dir=args.figdir)

if __name__ == "__main__":
    main()
