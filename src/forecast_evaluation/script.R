library(scoringutils)
library(dplyr)
library(tidyverse)

## Read forecasts
fc_ets <- read_csv("time_series_analysis_forecast/outputs/output_fc_ets.csv")
fc_arima <- read_csv("time_series_analysis_forecast/outputs/output_fc_arima.csv")
fc_arima_tweet <- read_csv("time_series_analysis_forecast/outputs/output_fc_arima_tweet.csv")

## Read data
data <- read_csv("data_covid_cases/outputs/cases_GB.csv")

## Combine forecasts and data for scoring
fc_all <- bind_rows(fc_arima,fc_ets,fc_arima_tweet) %>% 
  pivot_longer(-c(model,date,ci_level,forecast_date),
               names_to="quantile_old",values_to="prediction") %>% 
  mutate(quantile=ifelse(quantile_old=="mean",0.5,
                  ifelse(quantile_old=="lower",0.025,0.975)),
         target_type="Cases",
         location="GB") %>% 
  dplyr::select(model,target_end_date=date,target_type,location,
                quantile,prediction,forecast_date) %>% 
  left_join(data %>% dplyr::select(target_end_date=date,true_value=cases)) %>% 
  filter(target_end_date<ymd("2022-03-29"))

## Forecasts evaluation and scoring using scoringutils package
fc_score <- fc_all %>%
  score() %>%
  summarise_scores(by = c("model", "target_type")) %>%
  summarise_scores(fun = signif, digits = 2)

## Output
## .csv output
write_csv(fc_score,"forecast_evaluation/outputs/fc_score.csv")