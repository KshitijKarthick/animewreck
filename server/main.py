import sys
import json
import logging
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from starlette.responses import RedirectResponse, FileResponse
from starlette.middleware.cors import CORSMiddleware
from server.recommender import generate_recommendations


DEBUG_MODE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler_fp = logging.FileHandler('logs/server.log')
handler_fp.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler_fp.setFormatter(formatter)
logger.addHandler(handler_fp)

app = FastAPI()
app.mount("/static/js", StaticFiles(directory="dist/js", html=True), name="static_js")
app.mount("/static/css", StaticFiles(directory="dist/css", html=True), name="static_css")

if DEBUG_MODE:
    app.add_middleware(CORSMiddleware, allow_origins=['*'])

with open('model_resources/anime_with_genre_cosine_nary_pairwise_distances_top50.json', 'r') as fp:
    logger.info('Loading graph similarity nodes.')
    PAIRWISE_MAPPING = json.load(fp)

with open('model_resources/anime_info.json', 'r') as fp:
    logger.info('Loading anime information.')
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
def render_status():
    return {'msg': 'Ok'}

@app.get("/api/anime/titles")
def titles():
    return ANIME_INFO

@app.get("/api/anime/neighbors/{anime_id}")
def render_neighbours(anime_id: int):
    return PAIRWISE_MAPPING[str(anime_id)]

@app.get("/api/anime/recommendations")
def recommendations(watch_history, specificity:int=50, genre_similarity:bool=False,
                    topn:int=50, inference:bool=True):
    watch_history = json.loads(watch_history)
    logger.info({
        'watch_history': watch_history,
        'specificity': specificity,
        'topn': topn
    })
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
    return result.reset_index()[columns_interested].rename(
        columns={
            'recommendation_rating': 'rating',
            'target_anime_monotonic_id': 'id',
            'inference_source': 'inference_source_ids'
        }).to_dict(orient='record')

logger.info('All application resources are loaded')
