import streamlit as st
import pandas as pd
import pickle
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
# headers = {"Authorization": "Bearer " +  os.getenv('HUGGINGFACE_API') }
# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     sentiment= response.json()
#     print(sentiment)
#     first_value = sentiment[0][0]['label']
#     return first_value
# import nltk
# from textblob import TextBlob

# nltk.download('punkt')

# def classify_emotions(text):
#     blob = TextBlob(text)
#     emotions = {
#         'Joy': 0,
#         'Anger': 0,
#         'Fear': 0,
#         'Sadness': 0,
#         'Surprise': 0,
#         'Disgust': 0
#     }
#     for sentence in blob.sentences:
#         sentence_emotions = tuple(sentence.sentiment_assessments.assessments)
#         for e in sentence_emotions:
#             if e[0] in emotions:
#                 emotions[e[0]] += e[1]
#     return max(emotions, key=emotions.get)

from textblob import TextBlob

def classify_genre(text):
    blob = TextBlob(text)
    emotions = {
        'joy': ['comedy', 'romance', 'musical'],
        'anger': ['action', 'crime', 'war'],
        'fear': ['horror', 'thriller', 'mystery'],
        'sadness': ['drama', 'tragedy', 'melodrama'],
        'surprise': ['sci-fi', 'fantasy', 'adventure']
    }
    dominant_emotion = blob.sentiment.polarity
    if dominant_emotion > 0.2:
        return emotions['joy'][0]
    elif dominant_emotion < -0.2:
        return emotions['anger'][0]
    else:
        for sentence in blob.sentences:
            for e in emotions:
                if e in sentence.string.lower():
                    return emotions[e][0]
                for g in emotions[e]:
                    if g in sentence.string.lower():
                        return g
    return 'neutral'


# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
# model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# def classify_sentiment(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     outputs = model(**inputs)
#     logits = outputs.logits.detach().numpy()[0]
#     label_map = {0: "joy", 1: "fear", 2: "anger", 3: "sadness", 4: "neutral", 5: "surprise", 6: "disgust"}
#     label_index = int(torch.argmax(torch.softmax(torch.tensor(logits), dim=-1)))
#     return label_map[label_index]


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path
# def fetch_poster(movie_id):
#     response= requests.get('https://api.themoviedb.org/3/movie/{}?api_key=eff5d6bf98cc9dce7007c83f7e742a79&language=en-US'.format(movie_id))
#     data=response.json()
#     poster_path = data['poster_path']
#     print(data)
#     return "http://image.tmdb.org/t/p/w500/" + poster_path

movie_dict = pickle.load(open('pickle-files/movie_dict.pkl','rb'))
similarity = pickle.load(open('pickle-files/similarity.pkl','rb'))
movies= pd.DataFrame(movie_dict)
st.title('Movie Recommender system')
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)


def recommend(movie):
    movie_index=movies[movies['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list= sorted(list(enumerate(distances)),reverse = True, key = lambda x:x[1])[1:6]
    recommended_movie_posters=[]
    recommended_movie_names=[]
    recommended_movie_genres=[]
    for i in movies_list:
        movie_id = movies.iloc[i[0]].id
        #fetch poster api 
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
        recommended_movie_genres.append(classify_genre(movies.iloc[i[0]].tags))
        print(recommended_movie_genres) 
        
    return recommended_movie_names,recommended_movie_posters,recommended_movie_genres
   


if st.button('Recommend'):
    recommended_movie_names,recommended_movie_posters,recommended_movie_genres= recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
        st.text(recommended_movie_genres[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])
        st.text(recommended_movie_genres[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
        st.text(recommended_movie_genres[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
        st.text(recommended_movie_genres[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])
        st.text(recommended_movie_genres[4])
