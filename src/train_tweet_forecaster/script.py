import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    description='Train a forecaster for an epi indicator on twitter data'
)
parser.add_argument('indicator_path', nargs='+', help='a csv of indicator values') 
parser.add_argument('tweet_path', nargs='+', help='a csv of tweets')
parser.add_argument('-o', '--output', default='forecast.csv', help='output path')
parser.add_argument('-m', '--mode', default='forecast', help='training mode, either forecasting or out of sample')
parser.add_argument('--lag', type=int, default=7, help='number of days to forecast')

def train_model(indicator, tweets, lag: int):
    pass

def make_forecasts(model, x, y):
    pass

def load_indicator(path):
    pass

def load_tweets(path):
    pass

def forecast_dataset(indicator, tweets):
    pass

def oss_dataset(indicator, tweets):
    pass

if __name__ == '__main__':
    args = parser.parse_args()

    indicators = [load_indicator(path) for path in args.indicator_path]
    tweets = [load_twitter(path) for path in args.tweet_path]

    if args.mode = 'forecast':
        dataset = forecast_dataset(
            indicators[0],
            tweets[0],
            args.lag
        )
    else:
        dataset = oss_dataset(
            indicators,
            tweets,
            args.lag
        )

    forecasts = list()
    for X_train, y_train, X_test, y_test in dataset:
        model = train_model(X_train, y_train, args.lag)
        forecasts.append(make_forecasts(model, X_test, y_test))

    pd.concat(forecasts).to_csv(args.output)
