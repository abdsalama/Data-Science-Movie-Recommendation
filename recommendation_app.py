import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
from recommendation_model import MovieRecommendationModel

class MovieRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommendation System")
        self.root.geometry("800x600")
        
        # Initialize the model
        self.model = MovieRecommendationModel()
        
        # Create the main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create input frame
        self.create_input_frame()
        
        # Create results frame
        self.create_results_frame()
        
        # Global variables
        self.filtered_movies = pd.DataFrame()
        self.displayed_movies = pd.DataFrame()
        
    def create_input_frame(self):
        """Create the input frame with all input fields"""
        input_frame = ttk.LabelFrame(self.main_frame, text="Search Criteria", padding="10")
        input_frame.pack(fill=tk.X, pady=10)
        
        # Movie names input
        ttk.Label(input_frame, text="Movie Names (comma separated):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.movie_names_entry = ttk.Entry(input_frame, width=40)
        self.movie_names_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Genres input
        ttk.Label(input_frame, text="Genres (comma separated):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.genres_entry = ttk.Entry(input_frame, width=40)
        self.genres_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Rating input
        ttk.Label(input_frame, text="Minimum Rating (0.5-5.0):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.rating_entry = ttk.Entry(input_frame, width=40)
        self.rating_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Year range inputs
        ttk.Label(input_frame, text="Start Year:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.start_year_entry = ttk.Entry(input_frame, width=40)
        self.start_year_entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(input_frame, text="End Year:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.end_year_entry = ttk.Entry(input_frame, width=40)
        self.end_year_entry.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        # Search button
        self.search_button = ttk.Button(button_frame, text="Search", command=self.search_movies)
        self.search_button.grid(row=0, column=0, padx=5)
        
        # More button
        self.more_button = ttk.Button(button_frame, text="Show More", command=self.show_more_movies, state="disabled")
        self.more_button.grid(row=0, column=1, padx=5)
        
        # Evaluate button
        self.evaluate_button = ttk.Button(button_frame, text="Evaluate Model", command=self.evaluate_model)
        self.evaluate_button.grid(row=0, column=2, padx=5)
        
    def create_results_frame(self):
        """Create the results frame with text widget and scrollbar"""
        results_frame = ttk.LabelFrame(self.main_frame, text="Recommendations", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create text widget
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)
        
    def search_movies(self):
        """Search for movies based on input criteria"""
        # Clear previous results
        self.results_text.delete("1.0", tk.END)
        self.more_button["state"] = "disabled"
        
        # Get input values
        movie_names_input = self.movie_names_entry.get().strip()
        genres_input = self.genres_entry.get().strip()
        rating_input = self.rating_entry.get().strip()
        start_year_input = self.start_year_entry.get().strip()
        end_year_input = self.end_year_entry.get().strip()
        
        # Process movie names
        movie_names = None
        if movie_names_input:
            movie_names = [name.strip() for name in movie_names_input.split(",") if name.strip()]
            
        # Process genres
        genres = None
        if genres_input:
            genres = [genre.strip() for genre in genres_input.split(",") if genre.strip()]
            
        # Process rating
        min_rating = None
        if rating_input:
            try:
                min_rating = float(rating_input)
                if not (0.5 <= min_rating <= 5.0):
                    messagebox.showerror("Error", "Rating must be between 0.5 and 5.0")
                    return
            except ValueError:
                messagebox.showerror("Error", "Rating must be a number")
                return
                
        # Process years
        start_year = None
        if start_year_input:
            try:
                start_year = int(start_year_input)
            except ValueError:
                messagebox.showerror("Error", "Start year must be a number")
                return
                
        end_year = None
        if end_year_input:
            try:
                end_year = int(end_year_input)
            except ValueError:
                messagebox.showerror("Error", "End year must be a number")
                return
                
        # Validate year range
        if start_year and end_year and start_year > end_year:
            messagebox.showerror("Error", "Start year cannot be greater than end year")
            return
            
        # Check if at least one criterion is provided
        if not (movie_names or genres or min_rating is not None or start_year or end_year):
            messagebox.showerror("Error", "Please provide at least one search criterion")
            return
            
        # Get recommendations
        try:
            self.filtered_movies = self.model.recommend_movies(
                movie_names=movie_names,
                genres=genres,
                min_rating=min_rating,
                start_year=start_year,
                end_year=end_year,
                top_n=100  # Get more to allow for "Show More" functionality
            )
            
            if self.filtered_movies.empty:
                messagebox.showinfo("No Results", "No movies found matching your criteria")
                return
                
            # Display first batch of results
            self.displayed_movies = pd.DataFrame()
            self.show_more_movies()
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
    def show_more_movies(self, batch_size=5):
        """Show more movie recommendations"""
        # Get remaining movies
        remaining = (
            self.filtered_movies
            if self.displayed_movies.empty
            else self.filtered_movies[
                ~self.filtered_movies["movieId"].isin(self.displayed_movies["movieId"])
            ]
        )
        
        if remaining.empty:
            messagebox.showinfo("No More Results", "No more movies to recommend")
            self.more_button["state"] = "disabled"
            return
            
        # Get next batch
        next_batch = remaining.head(batch_size)
        
        # Display the batch
        for _, movie in next_batch.iterrows():
            self.results_text.insert(
                tk.END,
                f"{movie['title']} ({movie['year']}) - Rating: {movie['average_rating']:.2f}\n"
            )
            
            # Display genres
            genres = eval(movie['genres_list']) if isinstance(movie['genres_list'], str) else movie['genres_list']
            self.results_text.insert(tk.END, f"Genres: {', '.join(genres)}\n\n")
            
        # Update displayed movies
        self.displayed_movies = pd.concat([self.displayed_movies, next_batch])
        
        # Check if there are more movies to show
        next_remaining = self.filtered_movies[
            ~self.filtered_movies["movieId"].isin(self.displayed_movies["movieId"])
        ]
        
        if next_remaining.empty:
            self.more_button["state"] = "disabled"
        else:
            self.more_button["state"] = "normal"
            
    def evaluate_model(self):
        """Evaluate the recommendation model and display metrics"""
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "Evaluating model performance...\n\n")
        self.root.update()
        
        try:
            metrics = self.model.evaluate_recommendations()
            
            self.results_text.insert(tk.END, "Model Evaluation Results:\n")
            self.results_text.insert(tk.END, f"Precision: {metrics['precision']:.4f}\n")
            self.results_text.insert(tk.END, f"Recall: {metrics['recall']:.4f}\n")
            self.results_text.insert(tk.END, f"F1 Score: {metrics['f1_score']:.4f}\n\n")
            
            self.results_text.insert(tk.END, "Explanation of metrics:\n")
            self.results_text.insert(tk.END, "- Precision: Proportion of recommended items that are relevant\n")
            self.results_text.insert(tk.END, "- Recall: Proportion of relevant items that are recommended\n")
            self.results_text.insert(tk.END, "- F1 Score: Harmonic mean of precision and recall\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during evaluation: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommendationApp(root)
    root.mainloop()
