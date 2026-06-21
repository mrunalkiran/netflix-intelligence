import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_clean():
    return pd.read_csv(os.path.join(DATA_DIR, 'netflix_clean.csv'))

# Content type split
def content_type_split(df):
    return df['type'].value_counts().reset_index()

# Top genres
def top_genres(df, top_n=10):
    genres = df['listed_in'].str.split(',').explode().str.strip()
    return genres.value_counts().head(top_n).reset_index()

# Content added per year
def content_per_year(df):
    return df.groupby('year_added')['show_id'].count().reset_index()\
             .rename(columns={'show_id': 'count'})\
             .dropna().sort_values('year_added')

# Top countries
def top_countries(df, top_n=10):
    countries = df['country'].str.split(',').explode().str.strip()
    countries = countries[countries != 'Unknown']
    return countries.value_counts().head(top_n).reset_index()

# Ratings distribution
def ratings_distribution(df):
    return df['rating'].value_counts().reset_index()

# Top directors
def top_directors(df, top_n=10):
    directors = df[df['director'] != 'Unknown']
    return directors['director'].value_counts().head(top_n).reset_index()

# Generate facts for RAG
def generate_facts(df):
    facts = []

    # Movies vs Shows
    counts = content_type_split(df)
    for _, row in counts.iterrows():
        facts.append(f"Netflix has {row['count']} {row['type']}s in its catalogue.")

    # Genres
    genres = top_genres(df)
    for _, row in genres.iterrows():
        facts.append(f"'{row['listed_in']}' is one of the most common genres on Netflix with {row['count']} titles.")

    # Countries
    countries = top_countries(df)
    for _, row in countries.iterrows():
        facts.append(f"{row['country']} has {row['count']} Netflix titles, making it one of the top content-producing countries.")

    # Per year
    per_year = content_per_year(df)
    for _, row in per_year.iterrows():
        facts.append(f"Netflix added {int(row['count'])} titles in {int(row['year_added'])}.")

    # Ratings
    ratings = ratings_distribution(df)
    for _, row in ratings.iterrows():
        facts.append(f"There are {row['count']} Netflix titles rated '{row['rating']}'.")

    return facts

if __name__ == '__main__':
    df = load_clean()

    print("=== Content Type ===")
    print(content_type_split(df))

    print("\n=== Top Genres ===")
    print(top_genres(df))

    print("\n=== Top Countries ===")
    print(top_countries(df))

    print("\n=== Content Per Year ===")
    print(content_per_year(df))

    print("\n=== Ratings ===")
    print(ratings_distribution(df))

    print(f"\nGenerated {len(generate_facts(df))} RAG facts")