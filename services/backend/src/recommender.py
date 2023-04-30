"""
TODO: This really needs a rewrite, this was copied of working jupyter notebooks to get a working implementation of the recommender
"""

import sqlite3
import pickle
import torch
import numpy as np
import pandas as pd

from functools import partial

from glob import glob
from fastai.collab import *
from fastai.tabular import *

from scipy.spatial.distance import cosine
from constants import max_slice, min_slice, random_seed, batch_size, device
from src.model import build_model, build_learner, load_pretrained_embeddings
from src.preprocessor import AnimeRatingsDataset, load_dataset, build_databunch

np.random.seed(random_seed)


anime_df = pd.read_feather("model_resources/animes.feather").set_index(
    "anime_monotonic_id"
)

train_df, test_df = load_dataset()
databunch = build_databunch(train_df=train_df, test_df=test_df)

all_embeddings = load_pretrained_embeddings()
model = build_model(
    anime_genre_embeddings=all_embeddings["anime_with_genre_embeddings"]
)
learn = build_learner(model=model, databunch=databunch)


def sort_by_distance(record, anime_monotonic_id_embeddings, reverse=True):
    target_embedding = anime_monotonic_id_embeddings[
        record["target_anime_monotonic_id"]
    ]
    sorted_mask = np.argsort(
        np.vectorize(
            lambda x: cosine(anime_monotonic_id_embeddings[x], target_embedding)
        )(record["input_anime_monotonic_id"]),
    )
    if reverse:
        sorted_mask = sorted_mask[::-1]
    record["input_anime_monotonic_id"] = record["input_anime_monotonic_id"][sorted_mask]
    record["input_rating"] = record["input_rating"][sorted_mask]
    record["input_status"] = record["input_status"][sorted_mask]
    return record


def get_similar_anime_from_watch_history(
    previous_watch_history, anime_embeddings, inference=False, topn=50
):
    neighbours_inference = None
    watched_animes = previous_watch_history
    not_watched_animes = list(set(anime_df.index).difference(set(watched_animes)))

    top_similar_not_watched_animes = [
        sorted(
            [
                (
                    not_watched,
                    cosine(anime_embeddings[not_watched], anime_embeddings[watched]),
                )
                for not_watched in not_watched_animes
            ],
            key=lambda x: x[1],
        )[:topn]
        for watched in watched_animes
    ]
    if inference:
        neighbours_inference = dict(
            zip(
                watched_animes,
                [[_[0] for _ in anime] for anime in top_similar_not_watched_animes],
            )
        )
    return list(itertools.chain(*top_similar_not_watched_animes)), neighbours_inference


def recommendation(
    previous_watch_history,
    previous_watch_ratings,
    anime_df,
    model,
    topn=50,
    only_similar=False,
    topn_similar=50,
    inference=False,
    genre_similarity=False,
):
    neighbours_inference = None
    watched_animes = previous_watch_history
    if only_similar:
        similarity_embeddings = all_embeddings["anime_embeddings"]
        if genre_similarity:
            similarity_embeddings = all_embeddings["anime_with_genre_embeddings"]
        (
            not_watched_similarity_scores,
            neighbours_inference,
        ) = get_similar_anime_from_watch_history(
            previous_watch_history,
            similarity_embeddings,
            topn=topn_similar,
            inference=inference,
        )
        (not_watched_animes, scores) = zip(*not_watched_similarity_scores)
        not_watched_animes = set(not_watched_animes)
    else:
        not_watched_animes = list(set(anime_df.index).difference(set(watched_animes)))

    ratings = []
    user_personalised_test_records = []

    for anime_id in not_watched_animes:
        slice_length = len(previous_watch_history) + 1
        needs_padding = max_slice - slice_length
        ordered = sort_by_distance(
            {
                "target_rating": 0,
                "target_anime_monotonic_id": anime_id,
                "target_status": None,
                "input_anime_monotonic_id": np.array(previous_watch_history),
                "input_rating": np.array(previous_watch_ratings),
                "input_status": np.array(
                    [0 for _ in range(len(previous_watch_ratings))]
                ),
            },
            all_embeddings["anime_with_genre_embeddings"],
        )
        ordered["input_anime_monotonic_id"] = np.concatenate(
            [
                np.zeros(needs_padding).astype(np.int32),
                ordered["input_anime_monotonic_id"],
            ]
        )
        ordered["input_rating"] = np.concatenate(
            [np.zeros(needs_padding).astype(np.int32), ordered["input_rating"]]
        )
        ordered["input_status"] = np.concatenate(
            [np.zeros(needs_padding).astype(np.int32), ordered["input_status"]]
        )
        user_personalised_test_records.append(ordered)

    user_personalised_predict_df = pd.DataFrame(user_personalised_test_records)

    dl = DataLoader(
        AnimeRatingsDataset(user_personalised_predict_df),
        batch_size=batch_size,
        shuffle=False,
    )

    prediction_model = model.eval()
    recommended_ratings = []

    with torch.no_grad():
        for X, y in dl:
            ratings = prediction_model(X.to(device))
            recommended_ratings.extend(ratings.tolist())
    recommended_ratings_df = pd.DataFrame(
        recommended_ratings, columns=["recommendation_rating"]
    )

    user_personalised_predict_df = (
        pd.concat(
            [user_personalised_predict_df.reset_index(), recommended_ratings_df], axis=1
        )
        .sort_values(by=["recommendation_rating"], ascending=False)
        .iloc[:topn]
    )

    result_df = pd.merge(
        user_personalised_predict_df,
        anime_df.reset_index()[["anime_monotonic_id", "title", "title_english", "anime_id"]].rename(
            columns={
                "anime_monotonic_id": "target_anime_monotonic_id",
                "anime_id": "mal_id"
            }
        ),
        on="target_anime_monotonic_id",
        how="inner",
    )

    if inference and only_similar:
        inference_df = (
            pd.merge(
                pd.DataFrame(
                    [
                        {
                            "input_anime_monotonic_id": watched,
                            "target_anime_monotonic_id": not_watched,
                        }
                        for watched, not_watched_list in neighbours_inference.items()
                        for not_watched in not_watched_list
                    ]
                ),
                anime_df.reset_index()[
                    ["anime_monotonic_id", "title", "title_english"]
                ].rename(columns={"anime_monotonic_id": "input_anime_monotonic_id"}),
                on="input_anime_monotonic_id",
                how="left",
            )
            .groupby("target_anime_monotonic_id")
            .agg(list)
            .reset_index()
        )
        result_df = pd.merge(
            result_df,
            inference_df.rename(
                columns={
                    "input_anime_monotonic_id": "inference_source",
                    "title": "inference_source_title",
                    "title_english": "inference_source_title_english",
                }
            ),
            on="target_anime_monotonic_id",
            how="inner",
        )

    return result_df.set_index("target_anime_monotonic_id")


generate_recommendations = partial(
    recommendation,
    anime_df=anime_df,
    model=learn.model,
    only_similar=True,
)
