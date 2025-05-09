import pandas as pd
import tkinter as tk
from tkinter import messagebox

# تحميل البيانات
movies = pd.read_csv(r"./Datasets/clean_movies.csv")
ratings = pd.read_csv(r"./Datasets/clean_ratings.csv")

# حساب متوسط التقييم لكل فيلم
average_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
average_ratings.columns = ["movieId", "average_rating"]

# دمج متوسط التقييم مع بيانات الأفلام
movies = movies.merge(average_ratings, on="movieId", how="left")
movies["average_rating"] = movies["average_rating"].fillna(0)


# دالة لتصفية الأفلام
def filter_movies(genres=None, min_rating=None, start_year=None, end_year=None):
    filtered = movies.copy()
    if genres:
        # تحويل الأنواع المدخلة إلى أحرف صغيرة
        genres = [g.strip().lower() for g in genres]
        # مقارنة الأنواع بعد تحويل النص في genres_list إلى أحرف صغيرة
        filtered = filtered[
            filtered["genres_list"].apply(
                lambda x: all(g in str(x).lower() for g in genres)
            )
        ]
    if min_rating:
        filtered = filtered[filtered["average_rating"] >= min_rating]
    if start_year is not None:
        filtered = filtered[filtered["year"] >= start_year]
    if end_year is not None:
        filtered = filtered[filtered["year"] <= end_year]
    return filtered


# دالة لجلب أحدث n أفلام
def get_latest_movies(df, n=5):
    return df.sort_values(by="year", ascending=False).head(n)


# إنشاء النافذة الرئيسية
root = tk.Tk()
root.title("Movie Recommender")
root.geometry("600x400")  # تحديد حجم النافذة

# إطار للمدخلات
input_frame = tk.Frame(root, padx=10, pady=10)
input_frame.pack(fill="x")

# تسميات ومدخلات
genres_label = tk.Label(input_frame, text="الأنواع المفضلة (مفصولة بفواصل):")
genres_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
genres_entry = tk.Entry(input_frame, width=40)
genres_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

rating_label = tk.Label(input_frame, text="التقييم الأدنى (مثال: 4.0):")
rating_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
rating_entry = tk.Entry(input_frame, width=40)
rating_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

start_year_label = tk.Label(input_frame, text="سنة البداية (مثال: 2000):")
start_year_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
start_year_entry = tk.Entry(input_frame, width=40)
start_year_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

end_year_label = tk.Label(input_frame, text="سنة النهاية (مثال: 2020):")
end_year_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
end_year_entry = tk.Entry(input_frame, width=40)
end_year_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)

# إطار للأزرار
button_frame = tk.Frame(root, padx=10, pady=5)
button_frame.pack(fill="x")

# أزرار البحث والمزيد
search_button = tk.Button(button_frame, text="بحث", command=lambda: search_movies())
search_button.grid(row=0, column=0, padx=5, pady=5)
more_button = tk.Button(
    button_frame, text="المزيد", command=lambda: display_more_movies(), state="disabled"
)
more_button.grid(row=0, column=1, padx=5, pady=5)

# إطار للنتائج مع شريط تمرير
result_frame = tk.Frame(root, padx=10, pady=10)
result_frame.pack(fill="both", expand=True)

scrollbar = tk.Scrollbar(result_frame)
scrollbar.pack(side=tk.RIGHT, fill="y")
results_text = tk.Text(
    result_frame, height=10, width=60, yscrollcommand=scrollbar.set, wrap="word"
)
results_text.pack(fill="both", expand=True)
scrollbar.config(command=results_text.yview)

# متغيرات عالمية
filtered_movies_global = pd.DataFrame()
displayed_movies = pd.DataFrame()


# دالة البحث
def search_movies():
    results_text.delete("1.0", tk.END)
    more_button["state"] = "disabled"
    genres_input = genres_entry.get().strip()
    rating_input = rating_entry.get().strip()
    start_year_input = start_year_entry.get().strip()
    end_year_input = end_year_entry.get().strip()

    genres = None
    if genres_input:
        genres = [g.strip() for g in genres_input.split(",") if g.strip()]
        if not genres:
            genres = None

    min_rating = None
    if rating_input:
        try:
            min_rating = float(rating_input)
            if not (0.5 <= min_rating <= 5.0):
                messagebox.showerror("خطأ", "التقييم يجب أن يكون بين 0.5 و 5.0")
                return
        except ValueError:
            messagebox.showerror("خطأ", "التقييم يجب أن يكون رقمًا")
            return

    start_year = None
    if start_year_input:
        if start_year_input.isdigit():
            start_year = int(start_year_input)
        else:
            messagebox.showerror("خطأ", "سنة البداية يجب أن تكون عددًا صحيحًا")
            return

    end_year = None
    if end_year_input:
        if end_year_input.isdigit():
            end_year = int(end_year_input)
        else:
            messagebox.showerror("خطأ", "سنة النهاية يجب أن تكون عددًا صحيحًا")
            return

    if start_year and end_year and start_year > end_year:
        messagebox.showerror("خطأ", "سنة البداية لا يمكن أن تكون أكبر من سنة النهاية")
        return

    if not (genres or min_rating is not None or start_year or end_year):
        messagebox.showerror(
            "خطأ", "يجب تحديد معيار واحد على الأقل: الأنواع، التقييم، أو نطاق السنوات"
        )
        return

    filtered_movies = filter_movies(genres, min_rating, start_year, end_year)
    if filtered_movies.empty:
        messagebox.showinfo("لا توجد نتائج", "لم يتم العثور على أفلام مطابقة لمعاييرك")
        return

    global filtered_movies_global
    filtered_movies_global = filtered_movies
    global displayed_movies
    displayed_movies = pd.DataFrame()
    display_more_movies()


# دالة عرض المزيد
def display_more_movies():
    global displayed_movies
    global filtered_movies_global
    remaining = (
        filtered_movies_global
        if displayed_movies.empty
        else filtered_movies_global[
            ~filtered_movies_global["movieId"].isin(displayed_movies["movieId"])
        ]
    )
    if remaining.empty:
        messagebox.showinfo("لا يوجد المزيد", "لا توجد أفلام أخرى للتوصية")
        more_button["state"] = "disabled"
        return
    latest = get_latest_movies(remaining, n=5)
    for _, row in latest.iterrows():
        results_text.insert(
            tk.END,
            f"{row['title']} ({row['year']}) - التقييم: {row['average_rating']:.2f}\n",
        )
    displayed_movies = pd.concat([displayed_movies, latest], ignore_index=True)
    next_remaining = filtered_movies_global[
        ~filtered_movies_global["movieId"].isin(displayed_movies["movieId"])
    ]
    if not next_remaining.empty:
        more_button["state"] = "normal"
    else:
        more_button["state"] = "disabled"


# تشغيل النافذة
root.mainloop()
