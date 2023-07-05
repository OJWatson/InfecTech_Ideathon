library(fable)
library(tsibble)
library(feasts)
library(urca)
library(dplyr)
library(tidyverse)
source("time_series_analysis_forecast/support.R")

## The output of this script is .csv files of forecast and
## graphs of all forecasts over time and the last set of forecast based on
## all data available

## Read GB Covid-19 time series of cases data from global.health
data <- read_csv("data_covid_cases/outputs/cases_GB.csv")

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
for (i in seq_along(forecast_date)){
  training_end_date <- forecast_date[i] - 1
  fc_ets_list[[i]] <- time_series_forecast(data = data,
                                           method = 'ets',
                                           date_start = start_date,
                                           date_end = training_end_date,
                                           length = "14 days", ci = 95) %>% 
    mutate(forecast_date=forecast_date[i])
}
fc_ets <- as_tibble(bind_rows(fc_ets_list))

## Forecast using ARIMA
fc_arima_list <- list()
for (i in seq_along(forecast_date)){
  training_end_date <- forecast_date[i] - 1
  fc_arima_list[[i]] <- time_series_forecast(data = data,
                                             method = 'arima',
                                             date_start = start_date,
                                             date_end = training_end_date,
                                             length = "14 days", ci = 95) %>% 
    mutate(forecast_date=forecast_date[i])
}
fc_arima <- as_tibble(bind_rows(fc_arima_list))

## Output
## .csv output
write_csv(fc_ets,"time_series_analysis_forecast/outputs/output_fc_ets.csv")
write_csv(fc_arima,"time_series_analysis_forecast/outputs/output_fc_arima.csv")

## Combine data and all forecast
output <- data %>% mutate(model="data") %>% 
  rename(x = cases) %>% 
  bind_rows(fc_ets,fc_arima) %>% 
  mutate(model = factor(model, levels = c("arima","ets","data")))

fc_plot <- output %>% ggplot() +
  geom_ribbon(aes(x = date, ymin = lower, ymax = upper, 
                  col = model, fill = model)) +
  geom_point(aes(x = date, y = x, col = model, fill = model)) +
  geom_line(aes(x = date, y = mean, 
                col = model, fill = model, group = forecast_date)) +
  scale_y_log10() +
  theme_bw() +
  theme(legend.position = "bottom") +
  labs(x = "Date", y = "Cases", col = NULL, fill = NULL)

ggsave("time_series_analysis_forecast/outputs/fc_over_time.png")

## Last set of forecast only
output_last_fc <- output %>% 
  filter(model == "data" | forecast_date == forecast_date[length(forecast_date)])

last_fc_plot <- output_last_fc %>% ggplot() +
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





