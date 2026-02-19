import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


url = "https://gist.githubusercontent.com/cookiemonster/9aea43748c470d49434b61e48dee1f44/raw/89814e416bf79d2467f8910bda782dd0e46620a4/imdb_top_1000.csv"


df = pd.read_csv(url)


df = df[['Series_Title', 'Released_Year', 'Genre', 'Director', 'Star1', 'Star2', 'Star3']]
df['Released_Year'] = df['Released_Year'].astype(str)


df['combined_feature'] = (
    df['Series_Title'] + " " +
    df['Released_Year'] + " " +
    df['Genre'] + " " +
    df['Director'] + " " +
    df['Star1'] + " " +
    df['Star2'] + " " +
    df['Star3']
)


vectorizer = TfidfVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(df['combined_feature'])


similarity = cosine_similarity(matrix)



def recommend(movie_title, top_n=5):
    movie_title = movie_title.strip()

    if movie_title not in df['Series_Title'].values:
        return "Movie not found!"

    movie_index = df[df['Series_Title'] == movie_title].index[0]

    similarity_scores = list(enumerate(similarity[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:top_n + 1]

    recommended_movies = [df.iloc[i[0]]['Series_Title'] for i in similarity_scores]
    return recommended_movies




while True:
    movie_name = input("Enter a movie title (or type 'exit' to quit): ").strip()
    if movie_name.lower() == 'exit':
        print("Goodbye!")
        break

    result = recommend(movie_name)
    print(f"Movies similar to '{movie_name}': {result}\n")





