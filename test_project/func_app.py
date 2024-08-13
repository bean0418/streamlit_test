# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval
import pickle

# Load data
df1 = pd.read_csv("test_project/tmdb_5000_credits.csv")
df2 = pd.read_csv("test_project/tmdb_5000_movies.csv")

# Rename columns for merging
df1.columns = ['id', 'title', 'cast', 'crew']

# Merge dataframes on 'id'
df = df2.merge(df1[['id', 'cast', 'crew']], on='id')

# Calculate the mean vote average (C) and the 90th percentile of vote counts (m)
C = df["vote_average"].mean()
m = df["vote_count"].quantile(0.9)

# Filter movies with vote counts above the 90th percentile
q_movies = df2.loc[df["vote_count"] >= m].copy()

# Define the weighted rating function
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

# Apply the weighted rating function and sort by score
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values("score", ascending=False)

# Content-Based Filtering
df['overview'] = df['overview'].fillna("")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping of movie titles to indices
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Define a function to get recommendations based on cosine similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Credits, Genres, and Keywords Based Recommender
for feature in ['cast', 'crew', 'keywords', 'genres']:
    df[feature] = df[feature].apply(literal_eval)

df['director'] = df['crew'].apply(lambda x: next((i['name'] for i in x if i['job'] == 'Director'), np.nan))

for feature in ['cast', 'keywords', 'genres']:
    df[feature] = df[feature].apply(lambda x: [i['name'] for i in x][:3] if isinstance(x, list) else [])

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x if isinstance(i, str)]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    else:
        return ''

for feature in ['cast', 'keywords', 'director', 'genres']:
    df[feature] = df[feature].apply(clean_data)

df['soup'] = df.apply(lambda x: ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres']), axis=1)

# Create a CountVectorizer and compute the cosine similarity matrix
cnt = CountVectorizer(stop_words='english')
cnt_matrix = cnt.fit_transform(df['soup'])
cosine_sim2 = cosine_similarity(cnt_matrix, cnt_matrix)

# Update indices for the new cosine similarity matrix
df = df.reset_index()
indices = pd.Series(df.index, index=df['title'])

# Save the movies DataFrame and cosine similarity matrix for future use
movies = df[['id', 'title']].copy()
cosine_sim = cosine_sim2

# pickle.dump(movies, open('movies.pickle', 'wb'))
# pickle.dump(cosine_sim2, open("cosine_sim.pickle", 'wb'))

# Example usage
# print(get_recommendations("The Dark Knight Rises", cosine_sim2))
# print(get_recommendations("Interstellar", cosine_sim2))

import pickle
import streamlit as st
from tmdbv3api import Movie, TMDb

movie = Movie()
tmdb = TMDb()
tmdb.api_key = "9fe459b640ec1c5b5b1d867bb0a4e389"
tmdb.language = 'ko-KR'

def get_recommendations(title):
    # 영화제목을 통해서 전체 데이터 기준 영화의 idx 얻기
    idx = movies[movies['title'] == title].index[0]

    # 코사인 유사도 행렬에서 idx에 해당하는 데이터를 (idx, 유사도) 형태로 얻기
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 코사인 유사도 기준으로 내림차순 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 대각 행렬을 제외한 10개의 추천 영화 슬라이싱
    sim_scores = sim_scores[1:11]

    # 추천 영화 목록 10개의 인덱스 정보 추출
    movie_indices =  [i[0] for i in sim_scores]

    # 인덱스 정보를 통해 영화 제목을 추출
    images = []
    titles = []
    for i in movie_indices:
        id = movies['id'].iloc[i]
        details = movie.details(id)

        image_path = details['poster_path']
        if image_path:
            image_path = "https://image.tmdb.org/t/p/w500/" + details['poster_path']
        else:
            image_path = 'test_project/no_image.jpg'

        images.append(image_path)

        titles.append(details['title'])

    return images, titles



# movies = pickle.load(open("movies.pickle", "rb"))
# cosine_sim = pickle.load(open("cosine_sim.pickle", "rb"))

st.set_page_config(layout='wide') # 화면을 더 넓게 보기
st.header('Notflix')

movie_list = movies['title'].values
title = st.selectbox('Choose a movie you like', movie_list)

if st.button('Recommend'):
    with st.spinner("Please wait..."):
        images, titles = get_recommendations(title)

        idx = 0

        for i in range(0, 2):
            cols = st.columns(5)
            for col in cols:
                col.image(images[idx])
                col.write(titles[idx])
                idx += 1
