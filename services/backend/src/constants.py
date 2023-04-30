import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # For pure recommendation
random_seed = 42

num_genres = 44
min_slice = 6
max_slice = 11
past_anime_length = 10
num_users, num_anime = (108711, 6668)
train_frac = 0.8
encoded_values_for_rating = 11
batch_size = 512
num_workers_dataloader = 0


genre_embedding_size = 20
lstm_hidden_dim = 256
bidirectional = True
lstm_layers = 1

pretrained_learner_fname = (
    "lstm_anime_genre_learner_len_min5_max10_6rep_8cyc_1.785-1.752"
)
pretrained_anime_genre_embeddings_file = (
    "model_resources/anime_with_genre_embeddings.wts"
)
sliced_user_grouped_rating_file = (
    "model_resources/ordered_sliced_user_grouped_rating_minlen5_maxlen10_rep6_small.pkl"
)
