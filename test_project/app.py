import pickle
import streamlit as st
from tmdbv3api import Movie, TMDb

movie = Movie()
tmdb = TMDb()
tmdb.api_key = "9fe459b640ec1c5b5b1d867bb0a4e389"
tmdb.language = 'ko-KR'

import logging
from tmdbv3api.exceptions import TMDbException

# Configure logging
logging.basicConfig(filename='error.log', level=logging.ERROR)

def get_recommendations(title):
    try:
        # 영화제목을 통해서 전체 데이터 기준 영화의 idx 얻기
        idx = movies[movies['title'] == title].index[0]

        # 코사인 유사도 행렬에서 idx에 해당하는 데이터를 (idx, 유사도) 형태로 얻기
        sim_scores = list(enumerate(cosine_sim[idx]))

        # 코사인 유사도 기준으로 내림차순 정렬
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # 대각 행렬을 제외한 10개의 추천 영화 슬라이싱
        sim_scores = sim_scores[1:11]

        # 추천 영화 목록 10개의 인덱스 정보 추출
        movie_indices = [i[0] for i in sim_scores]

        # 인덱스 정보를 통해 영화 제목을 추출
        images = []
        titles = []
        for i in movie_indices:
            id = movies['id'].iloc[i]
            try:
                details = movie.details(id)
                image_path = details.get('poster_path')
                if image_path:
                    image_path = "https://image.tmdb.org/t/p/w500/" + image_path
                else:
                    image_path = 'no_image.jpg'

                images.append(image_path)
                titles.append(details['title'])
            except TMDbException as e:
                logging.error(f"Error fetching details for movie ID {id}: {e}")
                continue

        return images, titles

    except Exception as e:
        logging.error(f"An error occurred in get_recommendations: {e}")
        return [], []



movies = pickle.load(open("test_project/movies.pickle", "rb"))
cosine_sim = pickle.load(open("test_project/cosine_sim.pickle", "rb"))

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
