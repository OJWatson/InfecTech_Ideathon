# Tweet forecaster

We attempted to build a Recurrent Neural Network (RNN) to estimate target data
(traditional indicators cases/deaths/...) from a twitter
dataset. For endemic or late-stage disease transmission, social media data may
have unknown, complex, non-linear relationships with the target data. Recent
applications of deep learning to predict public health indicators have shown
that they can outperform traditional time-series prediction methods,
Matero, M. et al. (2023), Núñez, M. et al. (2023).

This package produces forecasts from each date, D, in the target data,
for a given time horizon, H, (default 7 days). For each forecast, tweets before
D are processed by removing stop words, lower casing, lemmatisation,
topic modelling into a 10 dimensional vector using Latent Dirichlet Allocation.
The tweets are aggregated by date to produce 10 topic weighted tweet count
timeseries. The RNN is then trained in a supervised fashion by windowing the
timeseries, i.e. the training set consists of all sequence of length H in the
target data before D as the output, and the topic timeseries along with the
target data in the preceding dates as input.

The RNN assumes a heteroskedastic gaussian distribution in the target data. It
outputs a mean and standard deviation for each output date and minimises the
negative log likelihood of its predictions during the training process.

The RNN is implemented in an encoder-decoder architecture with teacher
forcing. The encoder reads in the previous topic-weighted tweet counts and
target data to produce a vector, Z, in a latent space. Z is passed to the
decoder as the initial carry state to produce the forecast. During training, we
apply teacher forcing, i.e. the target data for the previous date is input to
the decoder. During prediction, the previous prediction is used as input.

## Future work

This model needs to be validated on an appropriate dataset for policy making.
The architecture does not require tweet aggregation by date and could be
extended to work with individual tweets and/or social network data if available
or useful.

## WIP

This package is incomplete. Here is a list of the short term features to
implement:

 - [x] Forecast dataset creation (windowing)
 - [x] Text cleaning
 - [x] LDA modelling
 - [x] RNN architecture
 - [x] RNN teacher forcing
 - [x] Training loop
 - [] RNN forecasting function
 - [] Forecast evaluation loop

## Installation

Install dependencies with:

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

Get script help with:

```
python -m package.script -h
```

Run forecasts with:

```
python -m package.script ../data_simulate_tweet_counts/outputs/data.csv [twitter data]
```

## References

Matero, M. et al. (2023) ‘Opioid death projections with AI-based forecasts using
social media language’, npj Digital Medicine, 6(1), pp. 1–11. doi:
10.1038/s41746-023-00776-0.

Núñez, M. et al. (2023) ‘Forecasting virus outbreaks with social media data via
neural ordinary differential equations’, Scientific Reports, 13(1), p. 10870.
doi: 10.1038/s41598-023-37118-9.
