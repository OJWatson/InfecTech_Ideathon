import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from jaxtyping import Array
from jax import numpy as jnp
from typing import Tuple

nlp = spacy.load("en_core_web_sm")

def _aggregate_by_date(embeddings, date, full_date):
    df = pd.merge(
        full_date,
        pd.DataFrame(embeddings),
        left_on='date',
        right_on=date,
        how='left'
    ).fillna(0)

    return df.groupby('date').agg('sum').values

def count_embeddings(tweets: pd.Series):
    with nlp.select_pipes(enable=['tagger', 'attribute_ruler', 'lemmatizer']):
        docs = list(nlp.pipe(tweets, n_process=-1))

    vectoriser = CountVectorizer()

    corpus = [
        ' '.join([
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha
        ])
        for doc in docs
    ]

    return vectoriser.fit_transform(corpus)

def model_topics(
    tweets: pd.DataFrame,
    count_embeddings: Array,
    train_dates: pd.Series,
    test_dates: pd.Series) -> Tuple[Array, Array]:

    topic_model = LatentDirichletAllocation()
    train = tweets.date.dt.date.isin(train_dates.dt.date)
    test = tweets.date.dt.date.isin(test_dates.dt.date)
    X_train = _aggregate_by_date(
        topic_model.fit_transform(count_embeddings[train]),
        tweets[train].date,
        train_dates
    )
    X_test = _aggregate_by_date(
        topic_model.transform(count_embeddings[test]),
        tweets[test].date,
        test_dates
    )
    return jnp.array(X_train), jnp.array(X_test)
