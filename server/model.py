"""
TODO: This really needs a rewrite, this was copied of working jupyter notebooks to get a working implementation of the recommender
"""

import sqlite3
import pickle
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functools import partial

from glob import glob
from fastai.collab import *
from fastai.tabular import *
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from scipy.spatial.distance import cosine

np.random.seed(42)
batch_size = 512
num_users, num_anime = (108711, 6668)
past_anime_length = 5
encoded_values_for_rating = 11
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
anime_df = pd.read_feather('model_resources/animes.feather').set_index('anime_monotonic_id')
anime_embeddings = torch.from_numpy(torch.load('model_resources/anime_embeddings.torch'))


class AnimeRatingsDataset(Dataset):
    """Custom Dataset for loading entries from HDF5 databases"""

    def __init__(self, df, transform=None):
        self.df = df.copy()

        df = self.df
        target_rating = torch.tensor(df['target_rating'].values)
        target_anime_monotonic_id = torch.from_numpy(
            df['target_anime_monotonic_id'].astype(np.int).values.reshape(-1, 1))
        input_rating = torch.tensor(
            df['input_rating'].apply(AnimeRatingsDataset.one_hot_encode).values.tolist(),
            dtype=torch.int64
        ).view(-1, past_anime_length * encoded_values_for_rating)
        input_anime_monotonic_id = torch.tensor(
            df['input_anime_monotonic_id'].apply(lambda x: x.tolist()).values.tolist())
        self.x = torch.cat([
            target_anime_monotonic_id,
            input_anime_monotonic_id,
            input_rating,
        ], dim=1)
        self.y = target_rating.view(-1, 1).float()

    @staticmethod
    def one_hot_encode(ratings, value_range=encoded_values_for_rating):
        records = []
        if not (isinstance(ratings, list) or isinstance(ratings, np.ndarray)):
            ratings = [ratings]
        for rating in ratings:
            one_hot_encoded = np.zeros(value_range)
            one_hot_encoded[rating] = 1
            records.append(one_hot_encoded.tolist())
        return records

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = int(index)
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class Net(nn.Module):
    
    def __init__(self, anime_embedding_vocab, anime_embedding_dim,
                 lstm_hidden_dim, anime_embedding_weights=None,
                 num_past_animes=10, lstm_layers=1, bidirectional=False):
        super(Net, self).__init__()
        
        # Store all the constants.
        self.anime_embedding_weights = anime_embedding_weights
        self.anime_embedding_vocab = anime_embedding_vocab
        self.anime_embedding_dim = anime_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_past_animes = num_past_animes
        self.lstm_layers = lstm_layers
        self.y_range = [0, 10]
        self.bidirectional = bidirectional
        self.bidirectional_factor = 2 if bidirectional else 1

        if anime_embedding_weights is None:
            self.past_anime_embedding = nn.Embedding(anime_embedding_vocab, anime_embedding_dim)
            self.embedding_drop = nn.Dropout(0.2)
        else:
            self.past_anime_embedding = nn.Embedding.from_pretrained(anime_embedding_weights)
            self.embedding_drop = nn.Dropout(0)
        # LSTM is fed the concatenated output of past anime ratings with their respective embeddings.
        # It outputs the hidden state of size lstm_hidden_dim.
        # anime embedding_size + 1 would suffice as anime_embedding_size is already * number of past records
        self.lstm = nn.LSTM(anime_embedding_dim + 11, lstm_hidden_dim, lstm_layers, bidirectional=bidirectional)
        self.fc1 = nn.Linear(lstm_hidden_dim * 2 * lstm_layers * self.bidirectional_factor, self.lstm_hidden_dim)
        self.ln1 = nn.LayerNorm(self.lstm_hidden_dim)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(lstm_hidden_dim, self.anime_embedding_dim)
        self.ln2 = nn.LayerNorm(self.anime_embedding_dim)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(self.anime_embedding_dim + self.anime_embedding_dim, self.anime_embedding_dim)
        self.ln3 = nn.LayerNorm(self.anime_embedding_dim)
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(self.anime_embedding_dim, 1)

    def normalised_euclidean_dist(self, x, y, dim):
        # https://stackoverflow.com/questions/38161071/how-to-calculate-normalized-euclidean-distance-on-two-vectors
        distance = 0.5 * (torch.pow(torch.std(x - y, dim), 2)) / (
            torch.pow(torch.std(x, dim), 2) + torch.pow(torch.std(y, dim), 2)
        )
        return distance.to(device)

    def init_hidden(self, minibatch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden = (
            torch.zeros(self.lstm_layers * self.bidirectional_factor, minibatch_size, self.lstm_hidden_dim).to(device),
            torch.zeros(self.lstm_layers * self.bidirectional_factor, minibatch_size, self.lstm_hidden_dim).to(device)
        )

    def forward(self, X):

        self.init_hidden(minibatch_size=X.shape[0])

        target_anime_monotonic_id, past_anime_monotonic_ids, past_ratings, = X[:, 0:1], X[:, 1:past_anime_length + 1], X[:, past_anime_length + 1:]
        past_ratings = past_ratings.view(-1, past_anime_length, encoded_values_for_rating)
        target_anime_monotonic_id = target_anime_monotonic_id.view(-1)

        # Safety guard condition, for external manipulation
        if self.anime_embedding_weights is not None:
            self.past_anime_embedding.requires_grad = False
            self.past_anime_embedding.weight.requires_grad = False

        history_embeddings = self.embedding_drop(
            self.past_anime_embedding(past_anime_monotonic_ids)
        )

        future_embeddings = self.embedding_drop(
            self.past_anime_embedding(target_anime_monotonic_id)
        )

        lstm_input = torch.cat([
            past_ratings.float(),
            history_embeddings,
        ], 2).permute(1, 0, 2) # (seq_len, batch, input_size) for LSTM

        lstm_out, self.hidden = self.lstm(
            lstm_input,
            self.hidden
        )

        lstm_input = torch.cat([
            past_ratings.float(),
            history_embeddings,
        ], 2).permute(1, 0, 2)

        final_hidden_concat_state = torch.cat([
            self.hidden[0].permute(1, 0, 2),
            self.hidden[1].permute(1, 0, 2)
        ], 2).view(-1, self.lstm_hidden_dim * self.bidirectional_factor * self.lstm_layers * 2)

        fc1_out = self.drop1(F.relu(self.ln1(self.fc1(final_hidden_concat_state))))
        fc2_out = self.drop2(F.relu(self.ln2(self.fc2(fc1_out))))
        historical_state = fc2_out

        recommendation_input = torch.cat([
            future_embeddings,
            historical_state
        ], 1)

        fc3_out = self.drop3(F.relu(self.ln3(self.fc3(recommendation_input))))
        final_output = self.fc4(fc3_out)
        
        output_rating = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(final_output) + self.y_range[0]
        return output_rating


model = Net(
    anime_embedding_dim=50, anime_embedding_vocab=num_anime,
    lstm_hidden_dim=256, lstm_layers=1, anime_embedding_weights=anime_embeddings,
    bidirectional=True
)
model.to(device)

def sort_by_distance(record, anime_monotonic_id_embeddings, reverse=True):
    target_embedding = anime_monotonic_id_embeddings[record['target_anime_monotonic_id']]
    sorted_mask = np.argsort(
        np.vectorize(
            lambda x: cosine(anime_monotonic_id_embeddings[x], target_embedding)
        )(record['input_anime_monotonic_id']),
    )
    if reverse:
        sorted_mask = sorted_mask[::-1]
    record['input_anime_monotonic_id'] = record['input_anime_monotonic_id'][sorted_mask]
    record['input_rating'] = record['input_rating'][sorted_mask]
    record['input_status'] = record['input_status'][sorted_mask]
    return record

def get_similar_anime_from_watch_history(previous_watch_history, anime_embeddings, topn=50):
    watched_animes = previous_watch_history
    not_watched_animes = list(set(anime_df.index).difference(set(watched_animes)))
    return list(itertools.chain(*[
        sorted([
            (not_watched, cosine(anime_embeddings[not_watched], anime_embeddings[watched])) 
            for not_watched in not_watched_animes
        ], key=lambda x: x[1])[:topn]
        for watched in watched_animes
    ]))
        

def recommendation(
    previous_watch_history, previous_watch_ratings, anime_df, model,
    topn=50, only_similar=False, topn_similar=50
):
    watched_animes = previous_watch_history
    if only_similar:
        not_watched_animes, scores = zip(*get_similar_anime_from_watch_history(
            previous_watch_history, anime_embeddings, topn_similar
        ))
        not_watched_animes = set(not_watched_animes)
    else:
        not_watched_animes = list(set(anime_df.index).difference(set(watched_animes)))

    ratings = []
    user_personalised_test_records = []

    for anime_id in not_watched_animes:
        user_personalised_test_records.append(sort_by_distance({
            'target_rating': 0,
            'target_anime_monotonic_id': anime_id,
            'target_status': None,
            'input_anime_monotonic_id': np.array(previous_watch_history),
            'input_rating': np.array(previous_watch_ratings),
            'input_status': np.array([0 for _ in range(len(previous_watch_ratings))])
        }, anime_embeddings))

    user_personalised_predict_df = pd.DataFrame(user_personalised_test_records)

    dl = DataLoader(AnimeRatingsDataset(user_personalised_predict_df), batch_size=batch_size, shuffle=False)

    prediction_model = model.eval()
    recommended_ratings = []

    with torch.no_grad():
        for X, y in dl:
            ratings = prediction_model(X.to(device))
            recommended_ratings.extend(ratings.tolist())
    recommended_ratings_df = pd.DataFrame(recommended_ratings, columns=['recommendation_rating'])

    user_personalised_predict_df = pd.concat(
        [user_personalised_predict_df.reset_index(), recommended_ratings_df], axis=1
    ).sort_values(by=['recommendation_rating'], ascending=False).iloc[:topn]

    return pd.merge(
        user_personalised_predict_df,
        anime_df.reset_index()[['anime_monotonic_id', 'title', 'title_english']].rename(
            columns={'anime_monotonic_id': 'target_anime_monotonic_id'}
        ),
        on='target_anime_monotonic_id',
        how='inner'
    ).set_index('target_anime_monotonic_id')


# recommendation(
#     previous_watch_history=[3114, 5306, 2384, 5307, 4807],
#     previous_watch_ratings=[8, 8, 9, 7, 10],
#     anime_df=anime_df,
#     model=model
# )[['title', 'title_english', 'recommendation_rating']].head(10)

generate_recommendations = partial(
    recommendation,
    anime_df=anime_df,
    model=model,
    only_similar=True
)