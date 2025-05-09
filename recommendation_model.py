import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

class MovieRecommendationModel:
    """
    A movie recommendation model that filters based on rating, year, genre, and movie names
    by performing intersections between different filtering criteria.
    """

    def __init__(self, movies_path="./Datasets/clean_movies.csv", ratings_path="./Datasets/clean_ratings.csv"):
        """
        Initialize the recommendation model with dataset paths

        Parameters:
        -----------
        movies_path : str
            Path to the cleaned movies dataset
        ratings_path : str
            Path to the cleaned ratings dataset
        """
        self.movies_path = movies_path
        self.ratings_path = ratings_path
        self.movies = None
        self.ratings = None
        self.user_movie_matrix = None
        self.similarity_matrix = None
        self.load_data()

    def load_data(self):
        """Load and preprocess the movie and rating data"""
        # Load datasets
        self.movies = pd.read_csv(self.movies_path)
        self.ratings = pd.read_csv(self.ratings_path)

        # Calculate average ratings for each movie
        average_ratings = self.ratings.groupby("movieId")["rating"].mean().reset_index()
        average_ratings.columns = ["movieId", "average_rating"]

        # Merge average ratings with movie data
        self.movies = self.movies.merge(average_ratings, on="movieId", how="left")
        self.movies["average_rating"] = self.movies["average_rating"].fillna(0)

        # Create user-movie rating matrix for collaborative filtering
        self._create_user_movie_matrix()

    def _create_user_movie_matrix(self):
        """Create a user-movie matrix for collaborative filtering"""
        # Instead of creating a full matrix, we'll use a more memory-efficient approach
        # We'll sample a subset of users and movies for the similarity matrix

        # Get unique users and movies
        unique_users = self.ratings['userId'].unique()

        # Sample users if there are too many (adjust sample size based on available memory)
        if len(unique_users) > 1000:
            np.random.seed(42)  # For reproducibility
            sampled_users = np.random.choice(unique_users, size=1000, replace=False)
            user_ratings = self.ratings[self.ratings['userId'].isin(sampled_users)]
        else:
            user_ratings = self.ratings

        # Create a smaller pivot table with the sampled users
        try:
            self.user_movie_matrix = user_ratings.pivot_table(
                index='userId',
                columns='movieId',
                values='rating'
            ).fillna(0)

            # Calculate similarity matrix between users
            self.similarity_matrix = cosine_similarity(self.user_movie_matrix)
        except Exception as e:
            print(f"Warning: Could not create full user-movie matrix due to memory constraints: {e}")
            # Create a dummy similarity matrix for fallback
            self.user_movie_matrix = pd.DataFrame()
            self.similarity_matrix = np.array([[1.0]])

    def filter_by_rating(self, min_rating=3.5):
        """
        Filter movies by minimum rating

        Parameters:
        -----------
        min_rating : float
            Minimum average rating threshold

        Returns:
        --------
        pd.DataFrame
            Filtered movies dataframe
        """
        return self.movies[self.movies["average_rating"] >= min_rating]

    def filter_by_year(self, start_year=None, end_year=None):
        """
        Filter movies by release year range

        Parameters:
        -----------
        start_year : int or None
            Earliest release year to include
        end_year : int or None
            Latest release year to include

        Returns:
        --------
        pd.DataFrame
            Filtered movies dataframe
        """
        filtered = self.movies.copy()

        if start_year is not None:
            filtered = filtered[filtered["year"] >= start_year]

        if end_year is not None:
            filtered = filtered[filtered["year"] <= end_year]

        return filtered

    def filter_by_genres(self, genres):
        """
        Filter movies by genres

        Parameters:
        -----------
        genres : list
            List of genres to filter by

        Returns:
        --------
        pd.DataFrame
            Filtered movies dataframe
        """
        if not genres:
            return self.movies.copy()

        # Convert genres to lowercase for case-insensitive comparison
        genres = [g.strip().lower() for g in genres]

        # Filter movies that contain all specified genres
        filtered = self.movies[
            self.movies["genres_list"].apply(
                lambda x: all(g in [genre.lower() for genre in eval(x)] for g in genres)
                if isinstance(x, str) else False
            )
        ]

        return filtered

    def find_movies_by_name(self, movie_names):
        """
        Find movies by partial name matches

        Parameters:
        -----------
        movie_names : list
            List of movie name strings to search for

        Returns:
        --------
        pd.DataFrame
            Dataframe of matching movies
        """
        if not movie_names:
            return self.movies.copy()

        # Convert names to lowercase for case-insensitive comparison
        movie_names = [name.strip().lower() for name in movie_names if name.strip()]

        # Find movies that match any of the provided names
        matches = pd.DataFrame()
        for name in movie_names:
            name_matches = self.movies[self.movies['title'].str.lower().str.contains(name, na=False)]
            matches = pd.concat([matches, name_matches])

        # Remove duplicates
        matches = matches.drop_duplicates(subset=['movieId'])

        return matches

    def extract_common_genres(self, movie_names):
        """
        Extract common genres from a list of movie names

        Parameters:
        -----------
        movie_names : list
            List of movie names to find common genres for

        Returns:
        --------
        list
            List of common genres
        """
        # Find movies matching the provided names
        found_movies = self.find_movies_by_name(movie_names)

        if found_movies.empty:
            return []

        # Extract all genres from found movies
        all_genres = []
        genre_counts = {}

        for genres_list in found_movies['genres_list']:
            if isinstance(genres_list, str):
                genres = eval(genres_list)
                # Convert to lowercase for consistency
                genres = [genre.lower() for genre in genres]
                all_genres.extend(genres)

                # Count genre occurrences
                for genre in genres:
                    if genre in genre_counts:
                        genre_counts[genre] += 1
                    else:
                        genre_counts[genre] = 1

        # Find genres that appear in at least half of the movies
        threshold = len(found_movies) / 2
        common_genres = [genre for genre, count in genre_counts.items()
                         if count >= threshold]

        # If no common genres meet the threshold, take the top 2 most frequent
        if not common_genres and genre_counts:
            sorted_genres = sorted(genre_counts.keys(),
                                  key=lambda g: genre_counts[g],
                                  reverse=True)
            common_genres = sorted_genres[:2] if len(sorted_genres) > 2 else sorted_genres

        return common_genres

    def recommend_movies(self, movie_names=None, genres=None, min_rating=None,
                        start_year=None, end_year=None, top_n=10):
        """
        Recommend movies based on intersection of multiple filtering criteria

        Parameters:
        -----------
        movie_names : list or None
            List of movie names to base recommendations on
        genres : list or None
            List of genres to filter by
        min_rating : float or None
            Minimum rating threshold
        start_year : int or None
            Earliest release year
        end_year : int or None
            Latest release year
        top_n : int
            Number of recommendations to return

        Returns:
        --------
        pd.DataFrame
            Dataframe of recommended movies
        """
        # Start with all movies
        recommendations = self.movies.copy()

        # Extract common genres from movie names if provided
        if movie_names and not genres:
            common_genres = self.extract_common_genres(movie_names)
            if common_genres:
                genres = common_genres

        # Apply filters based on provided criteria
        if genres:
            recommendations = self.filter_by_genres(genres)

        if min_rating is not None:
            rating_filtered = self.filter_by_rating(min_rating)
            # Intersection with previous filter
            recommendations = recommendations[recommendations['movieId'].isin(rating_filtered['movieId'])]

        if start_year is not None or end_year is not None:
            year_filtered = self.filter_by_year(start_year, end_year)
            # Intersection with previous filters
            recommendations = recommendations[recommendations['movieId'].isin(year_filtered['movieId'])]

        # Sort by rating and recency
        recommendations = recommendations.sort_values(
            by=['average_rating', 'year'],
            ascending=[False, False]
        )

        # Return top N recommendations
        return recommendations.head(top_n)

    def evaluate_recommendations(self, test_size=0.2, random_state=42):
        """
        Evaluate the recommendation model using precision, recall, and F1 score

        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility

        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        # Check if we have a valid user-movie matrix
        if self.user_movie_matrix.empty:
            print("Warning: Cannot evaluate model without a valid user-movie matrix")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }

        # Split ratings into train and test sets
        train_ratings, test_ratings = train_test_split(
            self.ratings, test_size=test_size, random_state=random_state
        )

        # Create a binary matrix of whether a user rated a movie above 3.5
        actual = test_ratings.copy()
        actual['liked'] = (actual['rating'] >= 3.5).astype(int)

        # Get unique users in test set that are also in our user_movie_matrix
        valid_users = set(self.user_movie_matrix.index)
        test_users = [user for user in actual['userId'].unique() if user in valid_users]

        # Prepare containers for metrics
        all_precision = []
        all_recall = []
        all_f1 = []

        # For each user in test set
        for user_id in test_users:
            try:
                # Get movies the user liked in test set
                user_liked = set(actual[(actual['userId'] == user_id) &
                                      (actual['liked'] == 1)]['movieId'])

                if not user_liked:
                    continue

                # Get user's ratings from training set
                user_train_ratings = train_ratings[train_ratings['userId'] == user_id]

                if user_train_ratings.empty:
                    continue

                # Find similar users based on training data
                user_idx = self.user_movie_matrix.index.get_loc(user_id)
                similar_users = np.argsort(self.similarity_matrix[user_idx])[::-1][1:11]  # Top 10 similar users

                # Get recommendations based on similar users' preferences
                recommended_movies = set()
                for sim_user_idx in similar_users:
                    sim_user_id = self.user_movie_matrix.index[sim_user_idx]
                    sim_user_ratings = train_ratings[train_ratings['userId'] == sim_user_id]
                    liked_movies = set(sim_user_ratings[sim_user_ratings['rating'] >= 3.5]['movieId'])
                    recommended_movies.update(liked_movies)

                # Remove movies the user has already rated in training set
                already_rated = set(user_train_ratings['movieId'])
                recommended_movies = recommended_movies - already_rated

                # Calculate metrics
                if recommended_movies and user_liked:
                    # True positives: recommended movies that user actually liked
                    true_positives = len(recommended_movies.intersection(user_liked))

                    # Precision: proportion of recommended items that are relevant
                    precision = true_positives / len(recommended_movies)

                    # Recall: proportion of relevant items that are recommended
                    recall = true_positives / len(user_liked)

                    # F1 score
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                    all_precision.append(precision)
                    all_recall.append(recall)
                    all_f1.append(f1)
            except Exception as e:
                print(f"Error evaluating user {user_id}: {e}")
                continue

        # Calculate average metrics
        avg_precision = np.mean(all_precision) if all_precision else 0
        avg_recall = np.mean(all_recall) if all_recall else 0
        avg_f1 = np.mean(all_f1) if all_f1 else 0

        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1
        }
