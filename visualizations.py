import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast

# Set the style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load the cleaned data
print("Loading data...")
movies_df = pd.read_csv("./Datasets/clean_movies.csv")
ratings_df = pd.read_csv("./Datasets/clean_ratings.csv")

# Convert genres_list from string to actual list
movies_df['genres_list'] = movies_df['genres_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")

# Calculate average rating per movie for later use
avg_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
avg_ratings.columns = ['movieId', 'avg_rating', 'rating_count']

# Merge with movies dataframe
movies_with_ratings = movies_df.merge(avg_ratings, on='movieId', how='left')

# Fill NaN values (movies with no ratings)
movies_with_ratings['avg_rating'].fillna(0, inplace=True)
movies_with_ratings['rating_count'].fillna(0, inplace=True)

print("Data preparation complete. Creating visualizations...")

# 1. Histogram for rating distribution
plt.figure(figsize=(12, 8))
sns.histplot(ratings_df['rating'], bins=9, kde=True)
plt.title('Distribution of Movie Ratings', fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(np.arange(0.5, 5.5, 0.5))
plt.tight_layout()
plt.savefig(r'./charts/rating_distribution.png')
plt.show()

# 2. Boxplot of ratings by genre
# First, we need to explode the genres_list to have one row per movie-genre combination
exploded_genres = movies_with_ratings.explode('genres_list')
exploded_genres = exploded_genres.rename(columns={'genres_list': 'genre'})

# Get the top 10 most common genres for better visualization
genre_counts = Counter([genre for genres in movies_df['genres_list'] for genre in genres])
top_genres = [genre for genre, count in genre_counts.most_common(10)]

# Filter for top genres
genre_ratings = exploded_genres[exploded_genres['genre'].isin(top_genres)]

plt.figure(figsize=(14, 8))
sns.boxplot(x='genre', y='avg_rating', data=genre_ratings, palette='viridis')
plt.title('Distribution of Average Ratings by Genre (Top 10 Genres)', fontsize=16)
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Average Rating', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r'./charts/ratings_by_genre.png')
plt.show()

# 3. Countplot for most common genres
# Count genre occurrences
all_genres = [genre for genres in movies_df['genres_list'] for genre in genres]
genre_counts = Counter(all_genres)
top_genres_df = pd.DataFrame({
    'genre': [genre for genre, _ in genre_counts.most_common(15)],
    'count': [count for _, count in genre_counts.most_common(15)]
})

plt.figure(figsize=(14, 8))
sns.barplot(x='genre', y='count', data=top_genres_df, palette='viridis')
plt.title('Most Common Genres in Movies', fontsize=16)
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r'./charts/genre_counts.png')
plt.show()

# 4. Barplot for top 10 movies by average rating (with at least 100 ratings)
# Filter movies with at least 100 ratings for more meaningful results
popular_movies = movies_with_ratings[movies_with_ratings['rating_count'] >= 100]
top_rated = popular_movies.sort_values('avg_rating', ascending=False).head(10)

plt.figure(figsize=(14, 8))
sns.barplot(x='avg_rating', y='title', data=top_rated, palette='viridis')
plt.title('Top 10 Movies by Average Rating (Min. 100 ratings)', fontsize=16)
plt.xlabel('Average Rating', fontsize=14)
plt.ylabel('Movie Title', fontsize=14)
plt.xlim(4, 5)  # Most top movies will have ratings between 4 and 5
plt.tight_layout()
plt.savefig(r'./charts/top_rated_movies.png')
plt.show()

# 5. Scatter plot showing relationship between rating count and average rating
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='rating_count', 
    y='avg_rating', 
    data=movies_with_ratings[movies_with_ratings['rating_count'] > 0],
    alpha=0.5,
    hue='year',
    palette='viridis',
    size='rating_count',
    sizes=(20, 200),
    legend='brief'
)
plt.title('Relationship Between Number of Ratings and Average Rating', fontsize=16)
plt.xlabel('Number of Ratings', fontsize=14)
plt.ylabel('Average Rating', fontsize=14)
plt.xscale('log')  # Log scale for better visualization
plt.tight_layout()
plt.savefig(r'./charts/rating_count_vs_avg.png')
plt.show()

# 6. Heatmap for correlation between numerical columns
# Create a dataframe with numerical columns
numerical_df = pd.DataFrame({
    'avg_rating': movies_with_ratings['avg_rating'],
    'rating_count': movies_with_ratings['rating_count'],
    'year': movies_with_ratings['year']
})

plt.figure(figsize=(10, 8))
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Between Numerical Features', fontsize=16)
plt.tight_layout()
plt.savefig(r'./charts/correlation_heatmap.png')
plt.show()

# 7. Line plot showing number of ratings per year
ratings_per_year = ratings_df.groupby('year')['rating'].count().reset_index()
ratings_per_year.columns = ['year', 'rating_count']

plt.figure(figsize=(14, 8))
sns.lineplot(x='year', y='rating_count', data=ratings_per_year, marker='o', linewidth=2)
plt.title('Number of Ratings per Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Ratings', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(r'./charts/ratings_per_year.png')
plt.show()

print("All visualizations created successfully!")
