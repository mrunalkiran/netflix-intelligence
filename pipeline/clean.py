import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_data():
    path = os.path.join(DATA_DIR, 'netflix_titles.csv')
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates(subset='show_id')

    # Fill missing values
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['country'] = df['country'].fillna('Unknown')
    df['rating'] = df['rating'].fillna('Not Rated')
    df['duration'] = df['duration'].fillna('Unknown')

    # Clean date_added
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month

    # Clean listed_in into a list
    df['genres'] = df['listed_in'].apply(lambda x: [g.strip() for g in str(x).split(',')])

    # Separate movies and shows
    df['is_movie'] = df['type'] == 'Movie'

    return df

def save_clean(df):
    out = os.path.join(DATA_DIR, 'netflix_clean.csv')
    df.to_csv(out, index=False)
    print(f"Saved cleaned data to {out}")

if __name__ == '__main__':
    print("Loading data...")
    df = load_data()
    print(f"Raw rows: {len(df)}")

    print("Cleaning data...")
    df = clean_data(df)
    print(f"Clean rows: {len(df)}")

    print("Sample:")
    print(df[['title', 'type', 'country', 'year_added', 'rating']].head())

    save_clean(df)