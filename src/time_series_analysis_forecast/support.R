## Simple forecasting using time series analysis methods
## data: data going to be used, should be formatted: 1st column: date, 2nd column: data to be forecast
## --note: date should be in the ymd format, e.g., "2020-12-25"
## method: choose either 'ets' or 'arima'
## date_start: starting date for training data
## date_end: ending date for training data
## --note: if either date_start or date_end is NULL, then use entire data for training
## len: length of forecast
## ci: confidence interval in % - currently hardcoded to 95

time_series_forecast <- function(data,method='ets',date_start=NULL,date_end=NULL,
                                 len="14 days",ci=95){
  ci <- 95
  
  # rename columns of data
  colnames(data) <- c("date","x")
  data$date <- ymd(data$date)
  
  # subset data for training
  if (is.null(date_start) | is.null(date_end)){
    date_start <- ymd(min(data$date))
    date_end <- ymd(max(data$date))
  }
  data_training <- tibble(data.frame(date=seq(date_start,date_end,"day"))) %>% 
    left_join(data) %>% tsibble
  
  # train time series model
  if (method=='ets'){
    fit <- data_training %>%
      model(
        ets = ETS(log(x + 1) ~ trend("A"))
      )
  } else if (method=='arima'){
    fit <- data_training %>%
      model(
        arima = ARIMA(log(x + 1))
      )
  } else {
    break('Please specify method as either ets or arima')
  }
  
  # make forecast from time series model
  fc <- fit %>%
    forecast(h = len) %>%
    hilo(level = ci) %>% 
    # unpack_hilo(cols = as.name(paste0(ci,'%'))) %>%
    unpack_hilo(`95%`) %>% # currently hardcoded
    dplyr::select(model=.model,date,mean=.mean,
                  lower=paste0(ci,'%_lower'),upper=paste0(ci,'%_upper')) %>%
    mutate(ci_level=ci)
  
  return(list(fc=fc,fit=fit))
}

## Forecasting using time series analysis methods - adding tweets data
## data: data going to be used, should be formatted: 1st column: date, 2nd column: data to be forecast
## --note: date should be in the ymd format, e.g., "2020-12-25"
## tweet_data: daily summary of tweets data
## method: choose 'arima'
## date_start: starting date for training data
## date_end: ending date for training data
## --note: if either date_start or date_end is NULL, then use entire data for training
## len: length of forecast
## ci: confidence interval in % - currently hardcoded to 95

time_series_tweets_forecast <- function(data,tweet_data,method='arima',date_start=NULL,date_end=NULL,
                                        len="14 days",ci=95){
  ci <- 95
  
  # rename columns of data
  colnames(data) <- c("date","x")
  data$date <- ymd(data$date)
  tweet_data$date <- ymd(tweet_data$date)
  tweet_data <- tweet_data %>% replace(is.na(.), 0)
  
  # combine cases and tweets data
  data <- data %>% left_join(tweet_data) %>% replace(is.na(.), 0)
  
  # subset data for training
  if (is.null(date_start) | is.null(date_end)){
    date_start <- ymd(min(data$date))
    date_end <- ymd(max(data$date))
  }
  data_training <- tibble(data.frame(date=seq(date_start,date_end,"day"))) %>% 
    left_join(data) %>% tsibble
  
  # find optimal lags for each of the tweets topic variables: max of 14 days
  n_col_tweets <- ncol(tweet_data) - 1
  optimal_lag <- list()
  for (i in seq_len(n_col_tweets)){
    x <- data_training[,2]
    y <- data_training[,2+i]
    optimal_lag[[i]] <- which.max(sapply(1:14, 
                                         function(j) cor(y, lag(x, j), use = "complete")))
  }
  
  # replace covariates with lagged covariates for training data
  data_training_lag <- data_training
  for (i in seq_len(n_col_tweets)){
    data_training_lag[,2+i] <- lag(data_training[,2+i],n = optimal_lag[[i]])
  }
  
  data_training_lag <- data_training_lag %>% replace(is.na(.), 0)
  
  # create n-days ahead of covariates for forecast
  dur_forecast_days <- round(as.numeric(as.duration(len))/86400)
  forecast_window <- seq(date_end+1,date_end+dur_forecast_days,"days")
  
  # forecast window outside current data
  forecast_window_outside <- forecast_window[forecast_window>max(data$date)]
  
  # data for forecast: for forecast create additional data for n-days ahead
  if (length(forecast_window_outside) > 0){
    data_add_for_forecast <- as_tibble(data.frame(date=forecast_window_outside,
                                                  matrix(0,nrow=length(forecast_window_outside),
                                                         ncol=ncol(data)-1)))
    colnames(data_add_for_forecast) <- colnames(data)
    data_add_for_forecast <- data_add_for_forecast %>% 
      mutate(date=ymd(date)) %>% tsibble
    data_for_forecast <- bind_rows(data,data_add_for_forecast)
  } else {
    data_for_forecast <- data
  }
  
  data_lag <- data_for_forecast
  for (i in seq_len(n_col_tweets)){
    data_lag[,2+i] <- lag(data_for_forecast[,2+i],optimal_lag[[i]])
  }
  data_forecast_lag <- data_lag %>% filter(date %in% forecast_window) %>% tsibble
  
  # train time series model
  if (method=='arima'){
    fit <- data_training_lag %>%
      model(
        arima_tweet = ARIMA(log(x + 1) ~ topic_0 + topic_1 + topic_2 + topic_3 + topic_4 + 
                              topic_5 + topic_6 + topic_7 + topic_8 + topic_9)
      )
  } else {
    break('Please specify method as arima')
  }
  
  # make forecast from time series model
  fc <- fit %>%
    forecast(data_forecast_lag) %>%
    hilo(level = ci) %>% 
    # unpack_hilo(cols = as.name(paste0(ci,'%'))) %>%
    unpack_hilo(`95%`) %>% # currently hardcoded
    dplyr::select(model=.model,date,mean=.mean,
                  lower=paste0(ci,'%_lower'),upper=paste0(ci,'%_upper')) %>% 
    mutate(ci_level=ci)
  
  return(list(fc=fc,fit=fit))
}

## Forecasting using ensemble methods: ets, arima, arima_tweet
## data: data going to be used, should be formatted: 1st column: date, 2nd column: data to be forecast
## --note: date should be in the ymd format, e.g., "2020-12-25"
## tweet_data: daily summary of tweets data
## date_start: starting date for training data
## date_end: ending date for training data
## --note: if either date_start or date_end is NULL, then use entire data for training
## len: length of forecast
## ci: confidence interval in % - currently hardcoded to 95

time_series_ensemble_forecast <- function(data,tweet_data,date_start=NULL,date_end=NULL,
                                          len="14 days",ci=95){
  ci <- 95
  
  # rename columns of data
  colnames(data) <- c("date","x")
  data$date <- ymd(data$date)
  tweet_data$date <- ymd(tweet_data$date)
  tweet_data <- tweet_data %>% replace(is.na(.), 0)
  
  # combine cases and tweets data
  data <- data %>% left_join(tweet_data) %>% replace(is.na(.), 0)
  
  # subset data for training
  if (is.null(date_start) | is.null(date_end)){
    date_start <- ymd(min(data$date))
    date_end <- ymd(max(data$date))
  }
  data_training <- tibble(data.frame(date=seq(date_start,date_end,"day"))) %>% 
    left_join(data) %>% tsibble
  
  # find optimal lags for each of the tweets topic variables: max of 14 days
  n_col_tweets <- ncol(tweet_data) - 1
  optimal_lag <- list()
  for (i in seq_len(n_col_tweets)){
    x <- data_training[,2]
    y <- data_training[,2+i]
    optimal_lag[[i]] <- which.max(sapply(1:14, 
                                         function(j) cor(y, lag(x, j), use = "complete")))
  }
  
  # replace covariates with lagged covariates for training data
  data_training_lag <- data_training
  for (i in seq_len(n_col_tweets)){
    data_training_lag[,2+i] <- lag(data_training[,2+i],n = optimal_lag[[i]])
  }
  
  data_training_lag <- data_training_lag %>% replace(is.na(.), 0)
  
  # create n-days ahead of covariates for forecast
  dur_forecast_days <- round(as.numeric(as.duration(len))/86400)
  forecast_window <- seq(date_end+1,date_end+dur_forecast_days,"days")
  
  # forecast window outside current data
  forecast_window_outside <- forecast_window[forecast_window>max(data$date)]
  
  # data for forecast: for forecast create additional data for n-days ahead
  if (length(forecast_window_outside) > 0){
    data_add_for_forecast <- as_tibble(data.frame(date=forecast_window_outside,
                                                  matrix(0,nrow=length(forecast_window_outside),
                                                         ncol=ncol(data)-1)))
    colnames(data_add_for_forecast) <- colnames(data)
    data_add_for_forecast <- data_add_for_forecast %>% 
      mutate(date=ymd(date)) %>% tsibble
    data_for_forecast <- bind_rows(data,data_add_for_forecast)
  } else {
    data_for_forecast <- data
  }
  
  data_lag <- data_for_forecast
  for (i in seq_len(n_col_tweets)){
    data_lag[,2+i] <- lag(data_for_forecast[,2+i],optimal_lag[[i]])
  }
  data_forecast_lag <- data_lag %>% filter(date %in% forecast_window) %>% tsibble
  
  # train time series model
  fit <- data_training_lag %>%
    model(
      ets = ETS(log(x + 1) ~ trend("A")),
      arima = ARIMA(log(x + 1)),
      arima_tweet = ARIMA(log(x + 1) ~ topic_0 + topic_1 + topic_2 + topic_3 + topic_4 + 
                            topic_5 + topic_6 + topic_7 + topic_8 + topic_9)
    )
  
  # make forecast from time series model
  fc <- fit %>%
    forecast(data_forecast_lag) %>%
    summarise(
      x = dist_mixture(x[1], x[2], x[3], weights=c(1/3,1/3,1/3))
    ) %>%
    mutate(.model="ensemble") %>% 
    hilo(level = ci) %>%
    unpack_hilo(`95%`) %>% mutate(.mean=mean(x)) %>% 
    dplyr::select(model=.model,date,mean=.mean,
                  lower=paste0(ci,'%_lower'),upper=paste0(ci,'%_upper')) %>% 
    mutate(ci_level=ci)
  
  return(list(fc=fc,fit=fit))
}

## Create data for forecast
## data: data going to be used, should be formatted: 1st column: date, 2nd column: data to be forecast
## --note: date should be in the ymd format, e.g., "2020-12-25"
## tweet_data: daily summary of tweets data
## date_start: starting date for training data
## date_end: ending date for training data
## --note: if either date_start or date_end is NULL, then use entire data for training
## len: length of forecast
## ci: confidence interval in % - currently hardcoded to 95
create_data_for_forecast <- function(data,tweet_data,date_start=NULL,date_end=NULL,
                                     len="14 days"){
  
  # rename columns of data
  colnames(data) <- c("date","x")
  data$date <- ymd(data$date)
  tweet_data$date <- ymd(tweet_data$date)
  tweet_data <- tweet_data %>% replace(is.na(.), 0)
  
  # combine cases and tweets data
  data <- data %>% left_join(tweet_data) %>% replace(is.na(.), 0)
  
  # subset data for training
  if (is.null(date_start) | is.null(date_end)){
    date_start <- ymd(min(data$date))
    date_end <- ymd(max(data$date))
  }
  data_training <- tibble(data.frame(date=seq(date_start,date_end,"day"))) %>% 
    left_join(data) %>% tsibble
  
  # find optimal lags for each of the tweets topic variables: max of 14 days
  n_col_tweets <- ncol(tweet_data) - 1
  optimal_lag <- list()
  for (i in seq_len(n_col_tweets)){
    x <- data_training[,2]
    y <- data_training[,2+i]
    optimal_lag[[i]] <- which.max(sapply(1:14, 
                                         function(j) cor(y, lag(x, j), use = "complete")))
  }
  
  # replace covariates with lagged covariates for training data
  data_training_lag <- data_training
  for (i in seq_len(n_col_tweets)){
    data_training_lag[,2+i] <- lag(data_training[,2+i],n = optimal_lag[[i]])
  }
  
  data_training_lag <- data_training_lag %>% replace(is.na(.), 0)
  
  # create n-days ahead of covariates for forecast
  dur_forecast_days <- round(as.numeric(as.duration(len))/86400)
  forecast_window <- seq(date_end+1,date_end+dur_forecast_days,"days")
  
  # forecast window outside current data
  forecast_window_outside <- forecast_window[forecast_window>max(data$date)]
  
  # data for forecast: for forecast create additional data for n-days ahead
  if (length(forecast_window_outside) > 0){
    data_add_for_forecast <- as_tibble(data.frame(date=forecast_window_outside,
                                                  matrix(0,nrow=length(forecast_window_outside),
                                                         ncol=ncol(data)-1)))
    colnames(data_add_for_forecast) <- colnames(data)
    data_add_for_forecast <- data_add_for_forecast %>% 
      mutate(date=ymd(date)) %>% tsibble
    data_for_forecast <- bind_rows(data,data_add_for_forecast)
  } else {
    data_for_forecast <- data
  }
  
  data_lag <- data_for_forecast
  for (i in seq_len(n_col_tweets)){
    data_lag[,2+i] <- lag(data_for_forecast[,2+i],optimal_lag[[i]])
  }
  data_forecast_lag <- data_lag %>% filter(date %in% forecast_window) %>% tsibble
  
  return(data_forecast_lag)
}
