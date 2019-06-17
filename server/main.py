import json
import logging
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from server.model import generate_recommendations

app = FastAPI()
app.mount("/static", StaticFiles(directory="dist", html=True), name="static")
app.add_middleware(CORSMiddleware, allow_origins=['*'])

with open('model_resources/cosine_nary_pairwise_distances_top50.json', 'r') as fp:
    PAIRWISE_MAPPING = json.load(fp)

with open('model_resources/anime_info.json', 'r') as fp:
    ANIME_INFO = json.load(fp)

@app.get("/api/status")
def render_homepage():
    return {'msg': 'Ok'}

@app.get("/api/anime/titles")
def titles():
    return ANIME_INFO

@app.get("/api/anime/neighbors/{anime_id}")
def render_homepage(anime_id: int):
    return PAIRWISE_MAPPING[str(anime_id)]

@app.get("/api/anime/recommendations")
def recommendations(watch_history, topn=50):
    watch_history = json.loads(watch_history)
    print(watch_history)
    result = generate_recommendations(
        previous_watch_history=[int(_['id']) for _ in watch_history][:5],
        previous_watch_ratings=[int(_['rating']) for _ in watch_history][:5],
        topn=topn
    )
    print(result[['title', 'recommendation_rating']])
    return result.reset_index()[['title', 'recommendation_rating', 'target_anime_monotonic_id']].rename(
        columns={
            'recommendation_rating': 'rating',
            'target_anime_monotonic_id': 'id'
        }).to_dict(orient='record')
    # return [
    #     { 'title': 'a', 'rating': 2, 'id': 1 },
    #     { 'title': 'b', 'rating': 3, 'id': 2 },
    #     { 'title': 'c', 'rating': 4, 'id': 3 },
    #     { 'title': 'd', 'rating': 5, 'id': 4 },
    #     { 'title': 'e', 'rating': 6, 'id': 5 }
    # ]

logging.info('All application resources are loaded')
