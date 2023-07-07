library(fable)
library(tsibble)
library(feasts)
library(urca)
library(dplyr)
library(tidyverse)
library(cowplot)
library(distributional)
source("time_series_analysis_forecast/support.R")

## The output of this script is .csv files of forecast and
## graphs of all forecasts over time and the last set of forecast based on
## all data available

## Read GB Covid-19 time series of cases data from global.health
data <- read_csv("data_covid_cases/outputs/cases_GB.csv")

## Read processed tweets data from tweet_analysis folder
tweet_data <- read_csv("tweet_analysis/tweets_and_topics.csv")

## Summarise tweet data - daily
tweet_daily_data <- tweet_data %>% 
  mutate(date = as.Date(date)) %>% 
  group_by(date) %>% 
  summarise(topic_0 = sum(topic_0), topic_1 = sum(topic_1),
            topic_2 = sum(topic_2), topic_3 = sum(topic_3),
            topic_4 = sum(topic_4), topic_5 = sum(topic_5),
            topic_6 = sum(topic_6), topic_7 = sum(topic_7),
            topic_8 = sum(topic_8), topic_9 = sum(topic_9)) %>% 
  filter(date <= max(ymd(data$date)))

## Time series model: ETS and ARIMA
## Forecast starting in March 2020
## Produce forecasts for 14 days on the 1st and 15th dates of each month

## Create forecast date for every training data before forecast
start_date <- ymd(min(data$date))
end_date <- ymd(max(data$date))
forecast_date <- sort(c(seq(ymd(paste0(year(start_date),"-03-01")),ymd(max(data$date)),"1 month"),
                        seq(ymd(paste0(year(start_date),"-03-15")),ymd(max(data$date)),"1 month"),
                        end_date))

## Forecast using ETS
fc_ets_list <- list()
fit_ets_list <- list()
for (i in seq_along(forecast_date)){
  training_end_date <- forecast_date[i] - 1
  fc_ets_out <- time_series_forecast(data = data,
                                     method = 'ets',
                                     date_start = start_date,
                                     date_end = training_end_date,
                                     len = "14 days",ci=95)
  fc_ets_list[[i]] <- fc_ets_out$fc %>% mutate(forecast_date=forecast_date[i])
  fit_ets_list[[i]] <- fc_ets_out$fit
}
fc_ets <- as_tibble(bind_rows(fc_ets_list))

## Forecast using ARIMA
fc_arima_list <- list()
fit_arima_list <- list()
for (i in seq_along(forecast_date)){
  training_end_date <- forecast_date[i] - 1
  fc_arima_out <- time_series_forecast(data = data,
                                       method = 'arima',
                                       date_start = start_date,
                                       date_end = training_end_date,
                                       len = "14 days", ci = 95)
  fc_arima_list[[i]] <- fc_arima_out$fc %>% mutate(forecast_date=forecast_date[i])
  fit_arima_list[[i]] <- fc_arima_out$fit
}
fc_arima <- as_tibble(bind_rows(fc_arima_list))

## Forecast using ARIMA with tweets data
fc_arima_tweet_list <- list()
fit_arima_tweet_list <- list()
for (i in seq_along(forecast_date)){
  training_end_date <- forecast_date[i] - 1
  fc_arima_tweet_out <- time_series_tweets_forecast(data = data,
                                                          tweet_data = tweet_daily_data,
                                                          method = 'arima',
                                                          date_start = start_date,
                                                          date_end = training_end_date,
                                                          len = "14 days", ci = 95)
  fc_arima_tweet_list[[i]] <- fc_arima_tweet_out$fc %>% mutate(forecast_date=forecast_date[i])
  fit_arima_tweet_list[[i]] <- fc_arima_tweet_out$fit
}
fc_arima_tweet <- as_tibble(bind_rows(fc_arima_tweet_list))

## Forecast using ensemble
fc_ensemble_list <- list()
fit_ensemble_list <- list()
for (i in seq_along(forecast_date)){
  training_end_date <- forecast_date[i] - 1
  fc_ensemble_out <- time_series_ensemble_forecast(data = data,
                                                         tweet_data = tweet_daily_data,
                                                         date_start = start_date,
                                                         date_end = training_end_date,
                                                         len = "14 days", ci = 95)
  fc_ensemble_list[[i]] <- fc_ensemble_out$fc %>% mutate(forecast_date=forecast_date[i])
  fit_ensemble_list[[i]] <- fc_ensemble_out$fit
}
fc_ensemble <- as_tibble(bind_rows(fc_ensemble_list))

## Output
## .csv output
write_csv(fc_ets,"time_series_analysis_forecast/outputs/output_fc_ets.csv")
write_csv(fc_arima,"time_series_analysis_forecast/outputs/output_fc_arima.csv")
write_csv(fc_arima_tweet,"time_series_analysis_forecast/outputs/output_fc_arima_tweet.csv")
write_csv(fc_ensemble,"time_series_analysis_forecast/outputs/output_fc_ensemble.csv")

## Model fit output
saveRDS(fit_ets_list,"time_series_analysis_forecast/outputs/fit_ets_list.rds")
saveRDS(fit_arima_list,"time_series_analysis_forecast/outputs/fit_arima_list.rds")
saveRDS(fit_arima_tweet_list,"time_series_analysis_forecast/outputs/fit_arima_tweet_list.rds")
saveRDS(fit_ensemble_list,"time_series_analysis_forecast/outputs/fit_ensemble_list.rds")

## Combine data and all forecast
output <- data %>% mutate(model="data") %>% 
  rename(x = cases) %>% 
  bind_rows(fc_ets,fc_arima,fc_arima_tweet,fc_ensemble) %>% 
  mutate(model = factor(model, levels = c("arima","ets","arima_tweet","ensemble","data")))

fc_plot <- output %>% ggplot() +
  geom_ribbon(aes(x = date, ymin = lower, ymax = upper,
                  col = model, fill = model)) +
  geom_point(aes(x = date, y = x, col = model, fill = model)) +
  geom_line(aes(x = date, y = mean, 
                col = model, group = forecast_date)) +
  scale_y_log10() +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(x = "Date", y = "Cases", col = NULL, fill = NULL)

ggsave("time_series_analysis_forecast/outputs/fc_over_time.png")

## Last set of forecast only
output_last_fc <- output %>% 
  filter(model == "data" | forecast_date == forecast_date[length(forecast_date)])

last_fc_plot <- output_last_fc %>% tail(100) %>% 
  ggplot() +
  geom_ribbon(aes(x = date, ymin = lower, ymax = upper,
                  col = model, fill = model)) +
  geom_point(aes(x = date, y = x, col = model, fill = model)) +
  geom_line(aes(x = date, y = mean, 
                col = model, fill = model, group = forecast_date)) +
  # scale_y_log10() +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(x = "Date", y = "Cases", col = NULL, fill = NULL)

ggsave("time_series_analysis_forecast/outputs/fc_last.png")





