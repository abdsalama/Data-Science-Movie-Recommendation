import pandas as pd
from recommendation_model import MovieRecommendationModel

def main():
    # Initialize the model
    print("Initializing recommendation model...")
    model = MovieRecommendationModel()
    
    # Test 1: Recommend movies based on rating and year
    print("\nTest 1: Recommend movies with rating >= 4.0 from 2000-2010")
    recommendations = model.recommend_movies(
        min_rating=4.0,
        start_year=2000,
        end_year=2010,
        top_n=5
    )
    print_recommendations(recommendations)
    
    # Test 2: Recommend movies based on genres
    print("\nTest 2: Recommend Action and Adventure movies")
    recommendations = model.recommend_movies(
        genres=["Action", "Adventure"],
        top_n=5
    )
    print_recommendations(recommendations)
    
    # Test 3: Recommend movies based on movie names (extracting common genres)
    print("\nTest 3: Recommend movies similar to 'Star Wars' and 'Matrix'")
    recommendations = model.recommend_movies(
        movie_names=["Star Wars", "Matrix"],
        top_n=5
    )
    print_recommendations(recommendations)
    
    # Test 4: Intersection of multiple criteria
    print("\nTest 4: Recommend Sci-Fi movies with rating >= 4.0 from 1990-2000")
    recommendations = model.recommend_movies(
        genres=["Sci-Fi"],
        min_rating=4.0,
        start_year=1990,
        end_year=2000,
        top_n=5
    )
    print_recommendations(recommendations)
    
    # Test 5: Evaluate model performance
    print("\nTest 5: Evaluating model performance...")
    metrics = model.evaluate_recommendations()
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

def print_recommendations(recommendations):
    """Print formatted movie recommendations"""
    if recommendations.empty:
        print("No recommendations found matching the criteria.")
        return
        
    for _, movie in recommendations.iterrows():
        print(f"{movie['title']} ({movie['year']}) - Rating: {movie['average_rating']:.2f}")
        # Print genres
        genres = eval(movie['genres_list']) if isinstance(movie['genres_list'], str) else movie['genres_list']
        print(f"  Genres: {', '.join(genres)}")

if __name__ == "__main__":
    main()
