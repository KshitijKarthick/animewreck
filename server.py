from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split


from flask import Flask, request, jsonify, send_from_directory, Response
app = Flask(__name__)


import json
mapping = json.load(open('datasets/processed_ratings/anime_user_rating_mapping.json'))
mapping.keys()

num_users, num_anime = (108709, 6668)
batch_size = 1024
device = torch.device('cpu')


class Net(nn.Module):

    def __init__(self, anime_embedding_vocab, anime_embedding_dim, lstm_hidden_dim,
                 num_past_animes=5, batch_size=batch_size):
        super(Net, self).__init__()
        
        # Store all the constants.
        self.anime_embedding_vocab = anime_embedding_vocab
        self.anime_embedding_dim = anime_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_past_animes = num_past_animes
        self.batch_size = batch_size

        self.past_anime_embedding = nn.Embedding(anime_embedding_vocab, anime_embedding_dim)
        self.embedding_drop = nn.Dropout(0.2)
        # LSTM is fed the concatenated output of past anime ratings with their respective embeddings.
        # It outputs the hidden state of size lstm_hidden_dim.
        # anime embedding_size + 1 would suffice as anime_embedding_size is already * number of past records
        self.lstm = nn.LSTM(anime_embedding_dim + 1, lstm_hidden_dim, bidirectional=True)

        # Take the LSTM hidden state for the past anime watched with the future anime embedding
        # as input to provide recommendation for the future anime.
        # Final Hidden cells state, hidden state hence * 2
        self.drop1 = nn.Dropout(0.2)
        # Bidirectional hence * 2
        self.ln1 = nn.LayerNorm((2 * lstm_hidden_dim * 2))
        # Bidirectional hence * 2
        self.fc1 = nn.Linear(lstm_hidden_dim * 2 * 2, self.anime_embedding_dim)
        self.ln2 = nn.LayerNorm((self.anime_embedding_dim))
        # Historical embeddings + lstm past state
        self.drop2 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(self.anime_embedding_dim + self.anime_embedding_dim, 1)
        self.init_hidden(batch_size)

    def init_hidden(self, minibatch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden = (
            torch.zeros(2, minibatch_size, self.lstm_hidden_dim).to(device),
            torch.zeros(2, minibatch_size, self.lstm_hidden_dim).to(device)
        )

    def forward(self, x):
        num_past_records = 5
        past_anime_historical_ids = x[:, 1: num_past_records + 1]
        past_anime_ratings = x[:, num_past_records + 1:-1]
        future_anime_id = x[:, -1:]

        history_embeddings = self.past_anime_embedding(past_anime_historical_ids)
        drop_history_embeddings = self.embedding_drop(history_embeddings)
        future_embeddings = self.past_anime_embedding(future_anime_id)

        lstm_input = torch.cat([
            past_anime_ratings.view(-1, num_past_records, 1).permute(2, 1, 0).float(),
            drop_history_embeddings.permute(2, 1, 0)
        ]).permute(1, 2, 0)

        lstm_out, self.hidden = self.lstm(
            lstm_input,
            self.hidden
        )

        final_hidden_concat_state = torch.cat([
            self.hidden[0].permute(2, 1, 0),
            self.hidden[1].permute(2, 1, 0)
        ]).permute(1, 2, 0).contiguous().view(-1, self.lstm_hidden_dim * 2 * 2) # bidirectional hence * 2

        dropout1 = self.drop1(final_hidden_concat_state)
        ln1 = self.ln1(dropout1)
        historical_state = F.relu(self.fc1(ln1))
        ln2_historical_state = self.ln2(historical_state)        
        recommendation_input = torch.cat([
            future_embeddings.view(-1, self.anime_embedding_dim).permute(1, 0),
            ln2_historical_state.permute(1, 0)
        ]).permute(1, 0)

        dropout2 = self.drop2(recommendation_input)
        return self.fc2(dropout2)


model = Net(anime_embedding_dim=50, anime_embedding_vocab=num_anime, lstm_hidden_dim=256)
model.load_state_dict(torch.load('augmented_files_10_epochs_10.383869513002022-10.021028261336069_state_dict.pt'))
prediction_model = model.eval()
for param in prediction_model.parameters():
    param.requires_grad = False


def build_record(history_anime_id, history_ratings, new_anime):
    return np.concatenate([
        [5],
        history_anime_id[:5],
        history_ratings[:5],
        [new_anime]
    ])

def anime_recommendations(history_anime_id, history_ratings, new_anime):
    X = torch.from_numpy(np.array(
        build_record(history_anime_id, history_ratings, new_anime)
    ).reshape(1, -1)).to(device)
    with torch.no_grad():
        current_batch_size = X.shape[0]
        prediction_model.zero_grad()
        prediction_model.init_hidden(current_batch_size)
        result = prediction_model(X)
    return result

def obtain_top_n(history_anime_id, history_ratings, topn=10):
    watched = set(history_anime_id)
    anime_ratings = []
    for new_anime_idx in tqdm(range(num_anime)):
        if new_anime_idx not in watched:
            anime_ratings.append((
                new_anime_idx,
                float(anime_recommendations(history_anime_id, history_ratings, new_anime_idx)[0][0])
            ))
    top_anime_ratings = [
        (anime_idx, mapping['anime_titles'][str(mapping['idx2anime'][str(anime_idx)])], rating)
        for anime_idx, rating in sorted(anime_ratings, key=lambda x: x[1], reverse=True)[:topn]
    ]
    return top_anime_ratings


@app.route('/recommendations', methods=['GET'])
def recommendations():
    params = json.loads(request.args.get('past_history'))
    anime_ids = [p['id'] for p in params]
    ratings = [p['rating'] for p in params]
    result = obtain_top_n(
        history_anime_id=anime_ids,
        history_ratings=ratings
    )
    result = [{'anime': r[1], 'rating': r[2]} for r in result]
    return Response(json.dumps(result), mimetype='application/json')


@app.route('/')
def index():
    return send_from_directory('static', 'recommendation.html')


@app.route('/anime_list.json')
def anime_list():
    return send_from_directory('static', 'anime_list.json')
