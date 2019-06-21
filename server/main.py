import json
import logging
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from starlette.responses import RedirectResponse, FileResponse
from starlette.middleware.cors import CORSMiddleware
from server.model import generate_recommendations

app = FastAPI()
app.mount("/static/js", StaticFiles(directory="dist/js", html=True), name="static_js")
app.mount("/static/css", StaticFiles(directory="dist/css", html=True), name="static_css")
app.add_middleware(CORSMiddleware, allow_origins=['*'])

with open('model_resources/anime_with_genre_cosine_nary_pairwise_distances_top50.json', 'r') as fp:
    PAIRWISE_MAPPING = json.load(fp)

with open('model_resources/anime_info.json', 'r') as fp:
    ANIME_INFO = json.load(fp)

@app.exception_handler(404)
def validation_exception_handler(request, exc):
    logging.info('Redirecting')
    return FileResponse('dist/index.html')

@app.get("/static")
def render_homepage():
    return FileResponse('dist/index.html')

@app.get("/static/index.html")
def render_homepage_abs():
    return FileResponse('dist/index.html')

@app.get("/static/favicon.ico")
def render_ico():
    return FileResponse('dist/favicon.ico')

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
def recommendations(watch_history, specificity:int=50, genre_similarity:bool=False,
                    topn:int=50, inference:bool=True):
    watch_history = json.loads(watch_history)
    print(watch_history, specificity, topn)
    columns_interested = ['recommendation_rating', 'target_anime_monotonic_id']
    result = generate_recommendations(
        previous_watch_history=[int(_['id']) for _ in watch_history][:10],
        previous_watch_ratings=[int(_['rating']) for _ in watch_history][:10],
        topn=topn,
        topn_similar=specificity,
        inference=inference,
        genre_similarity=genre_similarity
    ).reset_index()
    if inference:
        columns_interested.extend(['inference_source'])
    print(result[columns_interested])
    return result.reset_index()[columns_interested].rename(
        columns={
            'recommendation_rating': 'rating',
            'target_anime_monotonic_id': 'id',
            'inference_source': 'inference_source_ids'
        }).to_dict(orient='record')

logging.info('All application resources are loaded')
