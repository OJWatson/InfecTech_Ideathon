library(fable)
library(tsibble)
library(feasts)
library(urca)
library(dplyr)
library(tidyverse)
library(cowplot)
library(distributional)
library(lubridate)
library(egg)
source("src/time_series_analysis_forecast/support.R")

## The output of this script is .csv files of forecast and
## graphs of all forecasts over time and the last set of forecast based on
## all data available

## Read GB Covid-19 time series of cases data from global.health
data <- read_csv("src/data_covid_cases/outputs/cases_GB.csv")

## Read processed tweets data from tweet_analysis folder
tweet_data <- read_csv("src/tweet_analysis/tweets_and_topics.csv")

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
  print(i)
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
write_csv(fc_ets,"src/time_series_analysis_forecast/outputs/output_fc_ets.csv")
write_csv(fc_arima,"src/time_series_analysis_forecast/outputs/output_fc_arima.csv")
write_csv(fc_arima_tweet,"src/time_series_analysis_forecast/outputs/output_fc_arima_tweet.csv")
write_csv(fc_ensemble,"src/time_series_analysis_forecast/outputs/output_fc_ensemble.csv")

## Model fit output
saveRDS(fit_ets_list,"src/time_series_analysis_forecast/outputs/fit_ets_list.rds")
saveRDS(fit_arima_list,"src/time_series_analysis_forecast/outputs/fit_arima_list.rds")
saveRDS(fit_arima_tweet_list,"src/time_series_analysis_forecast/outputs/fit_arima_tweet_list.rds")
saveRDS(fit_ensemble_list,"src/time_series_analysis_forecast/outputs/fit_ensemble_list.rds")

## Combine data and all forecast
output <- data %>% mutate(model="data") %>% 
  rename(x = cases) %>% 
  bind_rows(fc_ets,fc_arima,fc_arima_tweet,fc_ensemble) %>% 
  mutate(model = factor(model, levels = c("arima","ets","arima_tweet","ensemble","data")))
saveRDS(output, file = "src/time_series_analysis_forecast/outputs/raw_forecasts_output.rds")

#### Filtered to ARIMA and cases only
filtered_output <- output %>%
  filter(model %in% c("data", "arima")) %>%
  mutate(forecast_date = as.factor(forecast_date))

arima_case_forecast <- ggplot() +
  geom_point(data = subset(filtered_output, model == "data"),
             aes(x = date, y = x), pch = 20, col = "grey") +
  geom_ribbon(data = subset(filtered_output, model == "arima"),
              aes(x = date, ymin = lower, ymax = upper,
                  fill = forecast_date), alpha = 0.25) +
  geom_point(data = subset(filtered_output, model == "arima"),
             aes(x = date, y = x, col = forecast_date, fill = forecast_date)) +
  geom_line(data = subset(filtered_output, model == "arima"),
            aes(x = date, y = mean, 
                col = forecast_date, group = forecast_date)) +
  annotate("rect", fill = NA, col = "black", 
           xmin = as.Date("2022-03-29"), 
           xmax = as.Date("2022-04-15"),
           ymin = 350,
           ymax = 10^5,
           size = 0.25) +
  annotate("segment", x = as.Date("2021-02-01"), xend = as.Date("2022-03-29"), y = 31.62278, yend = 350, colour = "black") +
  annotate("segment", x = max(filtered_output$date) + 30, xend = as.Date("2022-04-15"), y = 31.62278, yend = 350, colour = "black") +
  scale_y_log10() +
  coord_cartesian(xlim =c(as.Date("2020-04-15"), max(filtered_output$date) + 5)) +
  theme_bw() +
  theme(legend.position = "none") +
  labs(x = "Date", y = "COVID-19 Cases", col = NULL, fill = NULL,
       title = "ARIMA Forecasting of Case Incidence Using Case Data")

output_last_fc <- output %>% 
  filter(model == "data" | forecast_date == forecast_date[length(forecast_date)]) %>%
  tail(100)
arima_case_forecast_final <- ggplot() +
  geom_point(data = subset(output_last_fc, model == "data"),
             aes(x = date, y = x), pch = 20, col = "grey") +
  geom_ribbon(data = subset(output_last_fc, model == "arima"),
              aes(x = date, ymin = lower, ymax = upper,
                  fill = model), alpha = 0.25) +
  geom_point(data = subset(output_last_fc, model == "arima"),
             aes(x = date, y = x, col = model, fill = model)) +
  geom_line(data = subset(output_last_fc, model == "arima"),
            aes(x = date, y = mean, 
                col = model, fill = model, group = forecast_date)) +
  scale_y_log10() +
  theme_bw() +
  theme(legend.position = "none",
        plot.background = element_rect(colour = "black", fill="white", linewidth=0.5),
        axis.title.x = element_blank()) +
  labs(x = "", y = "COVID-19 Cases", col = NULL, fill = NULL)

arima_case_overall_plot <- arima_case_forecast +
  annotation_custom(
    ggplotGrob(arima_case_forecast_final),
    xmin = as.Date("2021-02-01"), xmax = max(filtered_output$date) + 30, ymin = -1.75, ymax = 1.5)

arima_case_overall_plot
ggsave(plot = arima_case_overall_plot,
       width = 9.5, 
       height = 6.1,
       filename = "src/time_series_analysis_forecast/outputs/new_arima_case_data_forecasting.png")

#### Filtered to ARIMA and tweets only

filtered_output <- output %>%
  filter(model %in% c("data", "arima")) %>%
  mutate(forecast_date = as.factor(forecast_date))

arima_case_forecast <- ggplot() +
  geom_point(data = subset(filtered_output, model == "data"),
             aes(x = date, y = x), pch = 20, col = "grey") +
  geom_ribbon(data = subset(filtered_output, model == "arima"),
              aes(x = date, ymin = lower, ymax = upper,
                  fill = forecast_date), alpha = 0.25) +
  geom_point(data = subset(filtered_output, model == "arima"),
             aes(x = date, y = x, col = forecast_date, fill = forecast_date)) +
  geom_line(data = subset(filtered_output, model == "arima"),
            aes(x = date, y = mean, 
                col = forecast_date, group = forecast_date)) +
  annotate("rect", fill = NA, col = "black", 
           xmin = as.Date("2022-03-29"), 
           xmax = as.Date("2022-04-15"),
           ymin = 350,
           ymax = 10^5,
           size = 0.25) +
  annotate("segment", x = as.Date("2021-02-01"), xend = as.Date("2022-03-29"), y = 31.62278, yend = 350, colour = "black") +
  annotate("segment", x = max(filtered_output$date) + 30, xend = as.Date("2022-04-15"), y = 31.62278, yend = 350, colour = "black") +
  scale_y_log10() +
  coord_cartesian(xlim =c(as.Date("2020-04-15"), max(filtered_output$date) + 5)) +
  theme_bw() +
  theme(legend.position = "none") +
  labs(x = "Date", y = "COVID-19 Cases", col = NULL, fill = NULL,
       title = "ARIMA Forecasting of Case Incidence Using Case Data")

output_last_fc <- output %>% 
  filter(model == "data" | forecast_date == forecast_date[length(forecast_date)]) %>%
  tail(100)
arima_case_forecast_final <- ggplot() +
  geom_point(data = subset(output_last_fc, model == "data"),
             aes(x = date, y = x), pch = 20, col = "grey") +
  geom_ribbon(data = subset(output_last_fc, model == "arima"),
              aes(x = date, ymin = lower, ymax = upper,
                  fill = model), alpha = 0.25) +
  geom_point(data = subset(output_last_fc, model == "arima"),
             aes(x = date, y = x, col = model, fill = model)) +
  geom_line(data = subset(output_last_fc, model == "arima"),
            aes(x = date, y = mean, 
                col = model, fill = model, group = forecast_date)) +
  scale_y_log10() +
  theme_bw() +
  theme(legend.position = "none",
        plot.background = element_rect(colour = "black", fill="white", linewidth=0.5),
        axis.title.x = element_blank()) +
  labs(x = "", y = "COVID-19 Cases", col = NULL, fill = NULL)

arima_case_overall_plot <- arima_case_forecast +
  annotation_custom(
    ggplotGrob(arima_case_forecast_final),
    xmin = as.Date("2021-02-01



### Old ###################


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
