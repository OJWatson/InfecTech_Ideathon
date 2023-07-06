import argparse
import pandas as pd
import ast
from .embed import count_embeddings, model_topics
from .batch import windowing
from .train import train
from .forecasts import make_forecasts
from jax import random

parser = argparse.ArgumentParser(
    description='Train a forecaster for an epi indicator on twitter data'
)
parser.add_argument('indicator_path', nargs='+', help='a csv of indicator values') 
parser.add_argument('tweet_path', nargs='+', help='a csv of tweets')
parser.add_argument('-o', '--output', default='forecast.csv', help='output path')
parser.add_argument(
    '-m',
    '--mode',
    default='forecast',
    help='training mode, either forecasting or out of sample'
)
parser.add_argument(
    '-l',
    '--lag',
    type=int,
    default=7,
    help='number of days to forecast'
)

def train_model(X, y, lag):
    #TODO: cycle the key
    key = random.PRNGKey(0)
    X, y_input, y_output = windowing(X, y, lag)
    state = train(key, X, y_input, y_output, batch_size=-1)
    return state

def load_indicator(path):
    indicator = pd.read_csv(path)
    indicator.date = pd.to_datetime(indicator.date)
    return indicator.sort_values('date')

def load_tweets(path):
    dtype = {
        'user_location': 'str',
        'user_description': 'str',
        'user_followers': float,
        'user_friends': float,
        'user_favourites': float,
        'user_verified': 'str',
        'date': 'str',
        'text': 'str',
        'hashtags': 'object'
    }

    # 3 tweets removed because they had different columns
    tweets = pd.read_csv(
        path,
        usecols=range(1, 10),
        dtype=dtype
    )

    # fill in missing tweets
    tweets.text = tweets.text.fillna('')

    # set zeros
    tweets.user_followers = tweets.user_followers.fillna(0).astype(int)
    tweets.user_friends = tweets.user_friends.fillna(0).astype(int)
    tweets.user_favourites = tweets.user_favourites.fillna(0).astype(int)

    # parse dates
    tweets.date = pd.to_datetime(tweets.date, format="%d/%m/%Y %H:%M")
    # verified
    tweets.user_verified = tweets.user_verified.str.lower() == 'true'
    # hashtags
    tweets.hashtags = tweets.hashtags.fillna('[]').apply(ast.literal_eval)
    return tweets.sort_values('date')

def forecast_dataset(indicator, tweets, lag:int):
    embeddings = count_embeddings(tweets.text)
    start = lag * 4
    for i in range(start, len(indicator) - lag):
        y_train = indicator.loc[:start]
        y_test = indicator.loc[start:start+lag]
        X_train, X_test = model_topics(tweets, embeddings, y_train.date, y_test.date)
        yield X_train, y_train.n.values, X_test, y_test.n.values

#TODO: needs some work
def oss_dataset(indicators, tweets):
    raise Exception('not done yet')

if __name__ == '__main__':
    args = parser.parse_args()

    indicators = [load_indicator(path) for path in args.indicator_path]
    tweets = [load_tweets(path) for path in args.tweet_path]

    if args.mode == 'forecast':
        dataset = forecast_dataset(
            indicators[0],
            tweets[0].iloc[:10000], #TODO: remove after testing
            args.lag
        )
    else:
        dataset = oss_dataset(
            indicators,
            tweets
        )

    forecasts = list()
    for X_train, y_train, X_test, y_test in dataset:
        state = train_model(X_train, y_train, args.lag)
        forecasts.append(make_forecasts(state, X_test, y_test))

    pd.concat(forecasts).to_csv(args.output)
