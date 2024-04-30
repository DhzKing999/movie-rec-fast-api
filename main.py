import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)








movie_data = pd.read_csv("movies.csv")  
def predictMovies(movie:str):
    selected_features=['genres','keywords','tagline','cast','director']
    for feature in selected_features:
        movie_data[feature]=movie_data[feature].fillna('')
    combined_features = movie_data['genres']+''+movie_data['keywords']+''+movie_data['tagline']+''+movie_data['cast']+''+movie_data['director']
    vectorizer=TfidfVectorizer()
    feature_vectors=vectorizer.fit_transform(combined_features)
    similarity=cosine_similarity(feature_vectors)
    movie_name=movie
    list_of_all_titles=movie_data['title'].tolist()
    find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
    close_match=find_close_match[0]
    index_of_the_movie = movie_data[movie_data.title==close_match]['index'].values[0]
    similarity_score=list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)
    i=1
    data=[]
    for movie in sorted_similar_movies:
        index=movie[0]
        title_from_index=movie_data[movie_data.index==index]['title'].values[0]
        if(i<10):
            data.append(title_from_index)
            i+=1
    return data        



@app.get('/')
def home():
    return {"Working"}
    
@app.post('/predict')
def predict_movies( moviename : str):
    return { "Movies" : predictMovies(moviename)}

@app.get('/autocomplete/{name}')
def auto_suggestion(name:str):
    titles = movie_data["title"]
    titles = titles.str.lower()
    pattern = name.lower()
    matched_titles = [t for t in titles if re.search(pattern, t)] 
    return {"name":matched_titles[:10]}

@app.get('/{name}')
def get_movie(name: str):
    movie_data.set_index('title', inplace=True)
    result = movie_data.loc[name]
    result = result.drop("crew")
    data = result.to_dict()  # Convert Series object to dictionary
    print(data)
    movie_data.reset_index(inplace=True)
    return {"data":data}

