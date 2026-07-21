"""PCA dimensionality reduction for the legacy hand-crafted feature vectors."""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from . import tweet_features
def tweet_pca_reduce(tweets_train, tweets_test, output_dim: int):
    train_array = np.asarray([tweet_features.tweet_dict_to_nparr(features) for features, _ in tweets_train], dtype=float)
    test_array = np.asarray([tweet_features.tweet_dict_to_nparr(features) for features, _ in tweets_test], dtype=float)
    pca = PCA(n_components=output_dim); train_reduced, test_reduced = pca.fit_transform(train_array), pca.transform(test_array)
    return ([(tweet_features.tweet_nparr_to_dict(vector), label) for vector, (_, label) in zip(train_reduced, tweets_train)], [(tweet_features.tweet_nparr_to_dict(vector), label) for vector, (_, label) in zip(test_reduced, tweets_test)])
