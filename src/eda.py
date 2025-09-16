# src/eda.py
import pandas as pd

def basic_overview(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    out.append(["rows", len(df)])
    out.append(["columns", len(df.columns)])
    out.append(["duplicates_by_key", df.duplicated(subset=["title","type","release_year"]).sum()
                if set(["title","type","release_year"]).issubset(df.columns) else None])
    nulls = df.isnull().sum().sort_values(ascending=False)
    out.append(["missing_cells", int(nulls.sum())])
    return pd.DataFrame(out, columns=["metric","value"])

def yearly_counts(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[(df["release_year"]>=1950) & (df["release_year"]<=2024)]
    g = keep.groupby(["release_year","type"]).size().reset_index(name="count")
    return g.sort_values(["release_year","type"])

def top_countries(exploded_country: pd.DataFrame, n=10) -> pd.DataFrame:
    dat = exploded_country[["show_id","country"]].dropna().copy()
    dat["country"] = dat["country"].astype(str).str.strip()
    dat = dat[(dat["country"]!="") & (dat["country"].str.lower()!="nan")]
    dat = dat.drop_duplicates(subset=["show_id","country"])
    return dat.value_counts("country").head(n).rename_axis("country").reset_index(name="titles")

def duration_stats_movies(df: pd.DataFrame) -> pd.Series:
    x = df[(df["type"]=="Movie") & df["duration_minutes"].notna()]["duration_minutes"]
    return pd.Series({
        "movies": int(x.shape[0]),
        "mean_min": float(x.mean()),
        "median_min": float(x.median()),
        "std_min": float(x.std()),
        "min_min": float(x.min()) if not x.empty else None,
        "max_min": float(x.max()) if not x.empty else None
    })

def top_genres(df_mlb: pd.DataFrame, n=15) -> pd.DataFrame:
    gcols = [c for c in df_mlb.columns if c.startswith("listed_in_")]
    if not gcols:
        return pd.DataFrame(columns=["genre","count"])
    counts = df_mlb[gcols].sum().sort_values(ascending=False).head(n)
    return counts.rename_axis("genre").reset_index(name="count").assign(
        genre=lambda d: d["genre"].str.replace("listed_in_","",regex=False)
    )
