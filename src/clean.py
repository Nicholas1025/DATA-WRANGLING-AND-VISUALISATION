# src/clean.py
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import re
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

warnings.filterwarnings('ignore')

def load_raw(path: Path) -> pd.DataFrame:
    """
    Load raw Netflix data from CSV file
    
    Args:
        path: Path to the raw CSV file
        
    Returns:
        DataFrame with raw data
    """
    try:
        df = pd.read_csv(path)
        print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def normalize_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize country names: remove extra spaces, standardize common aliases
    
    Args:
        df: DataFrame with country column
        
    Returns:
        DataFrame with normalized country names
    """
    df = df.copy()
    
    if 'country' not in df.columns:
        return df
    
    # Country name standardization mapping
    country_mapping = {
        'United States': 'United States',
        'USA': 'United States',
        'US': 'United States',
        'United Kingdom': 'United Kingdom',
        'UK': 'United Kingdom',
        'Britain': 'United Kingdom',
        'South Korea': 'South Korea',
        'Korea': 'South Korea',
        'Republic of Korea': 'South Korea',
    }
    
    def clean_country_string(country_str):
        if pd.isna(country_str):
            return country_str
        
        # Split by comma and clean each country
        countries = [c.strip() for c in str(country_str).split(',')]
        normalized_countries = []
        
        for country in countries:
            # Apply mapping if exists, otherwise keep original
            normalized = country_mapping.get(country, country)
            normalized_countries.append(normalized)
        
        return ', '.join(normalized_countries)
    
    df['country'] = df['country'].apply(clean_country_string)
    print(f"Country names normalized")
    
    return df

def parse_date_added(df: pd.DataFrame, col: str = "date_added") -> pd.DataFrame:
    """
    Parse date_added column to standardized datetime format
    Handle different formats, missing values, and spaces
    
    Args:
        df: DataFrame containing date column
        col: Column name to parse
        
    Returns:
        DataFrame with parsed date column and extracted year/month
    """
    df = df.copy()
    
    if col not in df.columns:
        print(f"Column {col} not found")
        return df
    
    def parse_date_string(date_str):
        if pd.isna(date_str):
            return None
        
        # Clean the string
        date_str = str(date_str).strip()
        
        # Try different date formats
        formats = [
            '%B %d, %Y',  # January 1, 2020
            '%b %d, %Y',  # Jan 1, 2020
            '%Y-%m-%d',   # 2020-01-01
            '%m/%d/%Y',   # 01/01/2020
            '%d/%m/%Y',   # 01/01/2020 (European)
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except (ValueError, TypeError):
                continue
        
        # If no format works, try pandas' general parser
        try:
            return pd.to_datetime(date_str)
        except:
            return None
    
    # Parse dates
    df[f'{col}_parsed'] = df[col].apply(parse_date_string)
    
    # Extract year and month for analysis
    df[f'{col}_year'] = df[f'{col}_parsed'].dt.year
    df[f'{col}_month'] = df[f'{col}_parsed'].dt.month
    
    # Count successfully parsed dates
    parsed_count = df[f'{col}_parsed'].notna().sum()
    total_count = len(df)
    print(f"Date parsing: {parsed_count}/{total_count} ({parsed_count/total_count*100:.1f}%) successfully parsed")
    
    return df

def parse_duration(df: pd.DataFrame, col: str = "duration") -> pd.DataFrame:
    """
    Parse duration column into duration_minutes (for movies) and seasons (for TV shows)
    One column will be NaN for each row depending on content type
    
    Args:
        df: DataFrame containing duration column
        col: Column name to parse
        
    Returns:
        DataFrame with duration_minutes and seasons columns
    """
    df = df.copy()
    
    if col not in df.columns:
        print(f"Column {col} not found")
        return df
    
    def extract_duration_info(duration_str, content_type):
        if pd.isna(duration_str):
            return None, None
        
        duration_str = str(duration_str).strip().lower()
        
        # Extract numbers from the string
        numbers = re.findall(r'\d+', duration_str)
        
        if not numbers:
            return None, None
        
        number = int(numbers[0])
        
        # For movies: extract minutes
        if content_type == 'Movie':
            if 'min' in duration_str:
                return number, None
            else:
                return None, None
        
        # For TV Shows: extract seasons
        elif content_type == 'TV Show':
            if 'season' in duration_str:
                return None, number
            else:
                return None, None
        
        return None, None
    
    # Initialize columns
    df['duration_minutes'] = np.nan
    df['seasons'] = np.nan
    
    # Parse based on type
    for idx, row in df.iterrows():
        minutes, seasons = extract_duration_info(row.get(col), row.get('type'))
        df.at[idx, 'duration_minutes'] = minutes
        df.at[idx, 'seasons'] = seasons
    
    # Convert to appropriate types
    df['duration_minutes'] = df['duration_minutes'].astype('Int64')  # Nullable integer
    df['seasons'] = df['seasons'].astype('Int64')  # Nullable integer
    
    movie_duration_count = df['duration_minutes'].notna().sum()
    tv_seasons_count = df['seasons'].notna().sum()
    print(f"Duration parsing: {movie_duration_count} movies with minutes, {tv_seasons_count} TV shows with seasons")
    
    return df

def explode_multivalues(df: pd.DataFrame, 
                       cols=("country", "listed_in"), 
                       sep=",") -> dict:
    """
    Handle multi-value columns by creating both exploded and multi-hot encoded versions
    
    Args:
        df: DataFrame with multi-value columns
        cols: Column names to process
        sep: Separator for multi-values
        
    Returns:
        Dictionary with:
        - "exploded": DataFrame with multi-value columns exploded into separate rows
        - "mlb": DataFrame with multi-hot encoding for multi-value columns
    """
    df_exploded = df.copy()
    df_mlb = df.copy()
    
    # For exploded version: expand multi-value columns into separate rows
    for col in cols:
        if col in df.columns:
            # Split and explode
            df_exploded[col] = df_exploded[col].astype(str).str.split(sep)
            df_exploded = df_exploded.explode(col)
            # Clean up
            df_exploded[col] = df_exploded[col].str.strip()
            df_exploded = df_exploded[df_exploded[col] != 'nan']
            df_exploded = df_exploded[df_exploded[col] != '']
    
    # For multi-hot version: create binary columns for each unique value
    mlb_encodings = {}
    for col in cols:
        if col in df.columns:
            # Prepare data for MultiLabelBinarizer
            multi_values = df[col].fillna('').astype(str).apply(lambda x: [v.strip() for v in x.split(sep) if v.strip() and v.strip() != 'nan'])
            
            # Fit MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            encoded = mlb.fit_transform(multi_values)
            
            # Create DataFrame with encoded columns
            encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{label}" for label in mlb.classes_])
            
            # Add to main DataFrame
            df_mlb = pd.concat([df_mlb.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
            mlb_encodings[col] = mlb
    
    exploded_shape = df_exploded.shape
    mlb_shape = df_mlb.shape
    print(f"Multi-value processing:")
    print(f"  Exploded: {exploded_shape[0]} rows, {exploded_shape[1]} columns")
    print(f"  Multi-hot: {mlb_shape[0]} rows, {mlb_shape[1]} columns")
    
    return {
        "exploded": df_exploded,
        "mlb": df_mlb,
        "encoders": mlb_encodings
    }

def deduplicate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate titles based on ['title', 'type', 'release_year']
    Priority: keep records with director/cast information
    
    Args:
        df: DataFrame with potential duplicates
        
    Returns:
        DataFrame with duplicates removed
    """
    df = df.copy()
    original_count = len(df)
    
    # Create priority score based on available information
    df['_priority_score'] = 0
    
    # Higher score for having director
    if 'director' in df.columns:
        df['_priority_score'] += df['director'].fillna('').apply(lambda x: 1 if x.strip() else 0)
    
    # Higher score for having cast
    if 'cast' in df.columns:
        df['_priority_score'] += df['cast'].fillna('').apply(lambda x: 1 if x.strip() else 0)
    
    # Higher score for having description
    if 'description' in df.columns:
        df['_priority_score'] += df['description'].fillna('').apply(lambda x: 1 if len(x.strip()) > 50 else 0)
    
    # Sort by priority score (descending) and keep first of each group
    dedup_cols = ['title', 'type', 'release_year']
    df_dedup = (df.sort_values('_priority_score', ascending=False)
                .drop_duplicates(subset=dedup_cols, keep='first')
                .drop('_priority_score', axis=1)
                .reset_index(drop=True))
    
    final_count = len(df_dedup)
    removed_count = original_count - final_count
    print(f"Deduplication: {removed_count} duplicates removed ({removed_count/original_count*100:.1f}%)")
    
    return df_dedup

def quality_report(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """
    Generate data quality report comparing before and after cleaning
    
    Args:
        before: DataFrame before cleaning
        after: DataFrame after cleaning
        
    Returns:
        DataFrame with quality metrics comparison
    """
    def get_quality_metrics(df, label):
        metrics = {
            'dataset': label,
            'rows': len(df),
            'columns': len(df.columns),
            'missing_cells': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        }
        
        # Column-specific missing rates
        for col in ['title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description']:
            if col in df.columns:
                missing_rate = (df[col].isnull().sum() / len(df)) * 100
                metrics[f'{col}_missing_%'] = missing_rate
        
        return metrics
    
    before_metrics = get_quality_metrics(before, 'Before Cleaning')
    after_metrics = get_quality_metrics(after, 'After Cleaning')
    
    # Combine metrics
    report_df = pd.DataFrame([before_metrics, after_metrics])
    
    # Calculate improvements
    improvement_row = {
        'dataset': 'Improvement',
        'rows': after_metrics['rows'] - before_metrics['rows'],
        'columns': after_metrics['columns'] - before_metrics['columns'],
        'missing_cells': after_metrics['missing_cells'] - before_metrics['missing_cells'],
        'missing_percentage': after_metrics['missing_percentage'] - before_metrics['missing_percentage'],
    }
    
    # Column improvements
    for col in ['title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description']:
        before_key = f'{col}_missing_%'
        if before_key in before_metrics and before_key in after_metrics:
            improvement_row[before_key] = after_metrics[before_key] - before_metrics[before_key]
    
    report_df = pd.concat([report_df, pd.DataFrame([improvement_row])], ignore_index=True)
    
    print("Data Quality Report Generated")
    return report_df

def clean_pipeline(in_path: Path, out_path: Path) -> dict:
    """
    Complete data cleaning pipeline
    Read raw data -> clean -> save processed data
    
    Args:
        in_path: Path to raw data file
        out_path: Path to save cleaned data
        
    Returns:
        Dictionary with:
        - "clean": Main cleaned DataFrame
        - "exploded": DataFrame with multi-value columns exploded
        - "mlb": DataFrame with multi-hot encoding
        - "report": Data quality report
    """
    print("=== Netflix Data Cleaning Pipeline ===")
    print(f"Input: {in_path}")
    print(f"Output: {out_path}")
    
    # Load raw data
    print("\n1. Loading raw data...")
    df_raw = load_raw(in_path)
    
    # Create a copy for processing
    df = df_raw.copy()
    
    # Cleaning steps
    print("\n2. Normalizing country names...")
    df = normalize_country(df)
    
    print("\n3. Parsing date_added...")
    df = parse_date_added(df)
    
    print("\n4. Parsing duration...")
    df = parse_duration(df)
    
    print("\n5. Deduplicating titles...")
    df = deduplicate_titles(df)
    
    print("\n6. Processing multi-value columns...")
    multi_result = explode_multivalues(df, cols=["country", "listed_in"])
    df_exploded = multi_result["exploded"]
    df_mlb = multi_result["mlb"]
    
    print("\n7. Generating quality report...")
    quality_rep = quality_report(df_raw, df)
    
    # Save cleaned data
    print(f"\n8. Saving cleaned data to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    # Save additional versions
    out_dir = out_path.parent
    df_exploded.to_csv(out_dir / "netflix_exploded.csv", index=False)
    df_mlb.to_csv(out_dir / "netflix_mlb.csv", index=False)
    quality_rep.to_csv(out_dir / "quality_report.csv", index=False)
    
    print(f"\n✅ Pipeline completed!")
    print(f"Final data shape: {df.shape}")
    print(f"Files saved:")
    print(f"  - Main: {out_path}")
    print(f"  - Exploded: {out_dir / 'netflix_exploded.csv'}")
    print(f"  - Multi-hot: {out_dir / 'netflix_mlb.csv'}")
    print(f"  - Quality report: {out_dir / 'quality_report.csv'}")
    
    return {
        "clean": df,
        "exploded": df_exploded,
        "mlb": df_mlb,
        "report": quality_rep
    }

# Example usage
if __name__ == "__main__":
    # Example paths
    raw_path = Path("data/raw/netflix_titles.csv")
    processed_path = Path("data/processed/netflix_clean.csv")
    
    # Run pipeline
    results = clean_pipeline(raw_path, processed_path)
    
    # Display sample of results
    print("\nSample of cleaned data:")
    print(results["clean"].head())
    
    print("\nQuality report:")
    print(results["report"])