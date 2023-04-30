from torch.utils.data import Dataset, random_split

from fastai.collab import *
from fastai.tabular import *
from src.constants import (
    train_frac,
    encoded_values_for_rating,
    past_anime_length,
    batch_size,
    num_workers_dataloader,
    device,
    sliced_user_grouped_rating_file,
)


class AnimeRatingsDataset(Dataset):
    """Custom Dataset for loading entries from HDF5 databases"""

    def __init__(self, df, transform=None):
        self.df = df.copy()

        df = self.df
        target_rating = torch.tensor(df["target_rating"].values)
        target_anime_monotonic_id = torch.from_numpy(
            df["target_anime_monotonic_id"].astype(np.int).values.reshape(-1, 1)
        )
        input_rating = torch.tensor(
            df["input_rating"]
            .apply(AnimeRatingsDataset.one_hot_encode)
            .values.tolist(),
            dtype=torch.int64,
        ).view(-1, past_anime_length * encoded_values_for_rating)
        input_anime_monotonic_id = torch.tensor(
            df["input_anime_monotonic_id"].apply(lambda x: x.tolist()).values.tolist()
        )
        self.x = torch.cat(
            [
                target_anime_monotonic_id,
                input_anime_monotonic_id,
                input_rating,
            ],
            dim=1,
        )
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


def load_dataset(
    sliced_user_grouped_rating_file=sliced_user_grouped_rating_file,
    train_frac=train_frac,
):
    grouped_rating_df = pd.read_pickle(sliced_user_grouped_rating_file)
    shuffled_ratings_genre_df = grouped_rating_df.sample(frac=1)
    msk = np.random.rand(len(shuffled_ratings_genre_df)) < train_frac
    train_df = shuffled_ratings_genre_df[msk]
    test_df = shuffled_ratings_genre_df[~msk]
    return train_df, test_df


def build_databunch(
    train_df,
    test_df,
    batch_size=batch_size,
    num_workers=num_workers_dataloader,
    device=device,
):
    return DataBunch.create(
        train_ds=AnimeRatingsDataset(train_df),
        valid_ds=AnimeRatingsDataset(test_df),
        device=device,
        bs=batch_size,
        num_workers=num_workers,
    )
