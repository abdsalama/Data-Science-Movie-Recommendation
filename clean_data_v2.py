import pandas as pd
import re

# 1. تحميل البيانات
movies = pd.read_csv(r"./Datasets/movies.csv")
ratings = pd.read_csv(r"./Datasets/ratings.csv")


# 2. تنظيف أسماء الأعمدة والنصوص
for df in [movies, ratings]:
    # إزالة المسافات الزائدة من أسماء الأعمدة
    df.columns = [col.strip() for col in df.columns]
    # تنظيف نص كل عمود نصّي
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

# 3. إزالة القيم المفقودة والتكرارات
for df in [movies, ratings]:
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

# 4. تحويل الطابع الزمني وتفصيله
ratings['timestamp'] = pd.to_datetime(
    ratings['timestamp'], unit='s', errors='coerce'
)
ratings.dropna(subset=['timestamp'], inplace=True)

# استخراج التاريخ والوقت
ratings['year']  = ratings['timestamp'].dt.year
ratings['month'] = ratings['timestamp'].dt.month
ratings['day']   = ratings['timestamp'].dt.day
ratings['time']  = ratings['timestamp'].dt.strftime('%H:%M')

# حذف العمود الأصلي
ratings.drop(columns=['timestamp'], inplace=True)

# 5. تصفية التقييمات المنطقية
ratings = ratings[(ratings['rating'] >= 0.5) & (ratings['rating'] <= 5.0)]

# 6. إزالة الأفلام بدون نوع
movies = movies[movies['genres'] != "(no genres listed)"]

# 7. استخراج سنة الإصدار من العنوان ثم حذفها من العنوان
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype('Int64')
movies['title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

# 8. تحويل genres إلى قائمة ثم حذف العمود القديم
movies['genres_list'] = movies['genres'].str.split('|')
movies.drop(columns=['genres'], inplace=True)

# 9. حذف الأفلام قبل سنة 1990
print(f"عدد الأفلام قبل حذف الأفلام القديمة: {len(movies)}")
movies = movies.dropna(subset=['year'])  # حذف الأفلام بدون سنة
movies = movies[movies['year'] >= 1990]
print(f"عدد الأفلام بعد حذف الأفلام قبل 1990: {len(movies)}")

# 10. حذف الأفلام المكررة (نفس العنوان ونفس السنة)
print(f"عدد الأفلام قبل حذف المكررات: {len(movies)}")
movies = movies.drop_duplicates(subset=['title'], keep='first')
print(f"عدد الأفلام بعد حذف المكررات: {len(movies)}")

# 11. توافق البيانات بين الجداول بناءً على movieId
valid_movie_ids = set(movies['movieId'])
ratings = ratings[ratings['movieId'].isin(valid_movie_ids)]

# (genome_tags لا يحتوي على movieId)

# 12. حفظ الملفات النظيفة
movies.to_csv(r"./Datasets/clean_movies.csv", index=False)
ratings.to_csv(r"./Datasets/clean_ratings.csv", index=False)

print("✅ تم تنظيف البيانات وحفظها بنجاح.")
