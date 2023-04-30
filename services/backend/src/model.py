import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from fastai.basic_train import Learner

from server.constants import (
    lstm_hidden_dim,
    lstm_layers,
    bidirectional,
    num_anime,
    device,
    pretrained_learner_fname,
    pretrained_anime_genre_embeddings_file,
    past_anime_length,
    encoded_values_for_rating,
)


class Net(nn.Module):
    def __init__(
        self,
        anime_embedding_vocab,
        anime_embedding_dim,
        lstm_hidden_dim,
        anime_embedding_weights=None,
        num_past_animes=10,
        lstm_layers=1,
        bidirectional=False,
    ):
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
            self.past_anime_embedding = nn.Embedding(
                anime_embedding_vocab, anime_embedding_dim
            )
            self.embedding_drop = nn.Dropout(0.2)
        else:
            self.past_anime_embedding = nn.Embedding.from_pretrained(
                anime_embedding_weights
            )
            self.embedding_drop = nn.Dropout(0)
        # LSTM is fed the concatenated output of past anime ratings with their respective embeddings.
        # It outputs the hidden state of size lstm_hidden_dim.
        # anime embedding_size + 1 would suffice as anime_embedding_size is already * number of past records
        self.lstm = nn.LSTM(
            anime_embedding_dim + 11,
            lstm_hidden_dim,
            lstm_layers,
            bidirectional=bidirectional,
        )
        self.fc1 = nn.Linear(
            lstm_hidden_dim * 2 * lstm_layers * self.bidirectional_factor,
            self.lstm_hidden_dim,
        )
        self.ln1 = nn.LayerNorm(self.lstm_hidden_dim)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(lstm_hidden_dim, self.anime_embedding_dim)
        self.ln2 = nn.LayerNorm(self.anime_embedding_dim)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(
            self.anime_embedding_dim + self.anime_embedding_dim,
            self.anime_embedding_dim,
        )
        self.ln3 = nn.LayerNorm(self.anime_embedding_dim)
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(self.anime_embedding_dim, 1)

    def normalised_euclidean_dist(self, x, y, dim):
        # https://stackoverflow.com/questions/38161071/how-to-calculate-normalized-euclidean-distance-on-two-vectors
        distance = (
            0.5
            * (torch.pow(torch.std(x - y, dim), 2))
            / (torch.pow(torch.std(x, dim), 2) + torch.pow(torch.std(y, dim), 2))
        )
        return distance.to(device)

    def init_hidden(self, minibatch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden = (
            torch.zeros(
                self.lstm_layers * self.bidirectional_factor,
                minibatch_size,
                self.lstm_hidden_dim,
            ).to(device),
            torch.zeros(
                self.lstm_layers * self.bidirectional_factor,
                minibatch_size,
                self.lstm_hidden_dim,
            ).to(device),
        )

    def forward(self, X):

        self.init_hidden(minibatch_size=X.shape[0])

        target_anime_monotonic_id, past_anime_monotonic_ids, past_ratings, = (
            X[:, 0:1],
            X[:, 1 : past_anime_length + 1],
            X[:, past_anime_length + 1 :],
        )
        past_ratings = past_ratings.view(
            -1, past_anime_length, encoded_values_for_rating
        )
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

        lstm_input = torch.cat([past_ratings.float(), history_embeddings,], 2).permute(
            1, 0, 2
        )  # (seq_len, batch, input_size) for LSTM

        lstm_out, self.hidden = self.lstm(lstm_input, self.hidden)

        lstm_input = torch.cat(
            [
                past_ratings.float(),
                history_embeddings,
            ],
            2,
        ).permute(1, 0, 2)

        final_hidden_concat_state = torch.cat(
            [self.hidden[0].permute(1, 0, 2), self.hidden[1].permute(1, 0, 2)], 2
        ).view(
            -1, self.lstm_hidden_dim * self.bidirectional_factor * self.lstm_layers * 2
        )

        fc1_out = self.drop1(F.relu(self.ln1(self.fc1(final_hidden_concat_state))))
        fc2_out = self.drop2(F.relu(self.ln2(self.fc2(fc1_out))))
        historical_state = fc2_out

        recommendation_input = torch.cat([future_embeddings, historical_state], 1)

        fc3_out = self.drop3(F.relu(self.ln3(self.fc3(recommendation_input))))
        final_output = self.fc4(fc3_out)

        output_rating = (self.y_range[1] - self.y_range[0]) * torch.sigmoid(
            final_output
        ) + self.y_range[0]
        return output_rating


def load_pretrained_embeddings(
    pretrained_anime_genre_embeddings_file=pretrained_anime_genre_embeddings_file,
):
    all_embeddings = torch.load(pretrained_anime_genre_embeddings_file)
    all_embeddings["anime_with_genre_embeddings"][0] = 0
    all_embeddings["anime_embeddings"][0] = 0
    return all_embeddings


def build_learner(
    databunch,
    model,
    loss_func=nn.MSELoss(),
    pretrained_learner_fname=pretrained_learner_fname,
):
    learn = Learner(data=databunch, model=model, loss_func=loss_func)
    if pretrained_learner_fname:
        learn = learn.load(pretrained_learner_fname)
    return learn


def build_model(
    anime_genre_embeddings,
    num_anime=num_anime,
    lstm_hidden_dim=lstm_hidden_dim,
    lstm_layers=lstm_layers,
    bidirectional=bidirectional,
    device=device,
):
    anime_genre_embeddings = torch.from_numpy(anime_genre_embeddings).float().to(device)
    model = Net(
        anime_embedding_dim=anime_genre_embeddings.shape[1],
        anime_embedding_vocab=num_anime,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layers=lstm_layers,
        anime_embedding_weights=anime_genre_embeddings,
        bidirectional=bidirectional,
    )

    model.to(device)
    return model
