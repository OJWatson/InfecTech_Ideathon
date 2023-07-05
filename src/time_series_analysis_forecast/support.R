## Simple forecasting using time series analysis methods
## data: data going to be used, should be formatted: 1st column: date, 2nd column: data to be forecast
## --note: date should be in the ymd format, e.g., "2020-12-25"
## method: choose either 'ets' or 'arima'
## date_start: starting date for training data
## date_end: ending date for training data
## --note: if either date_start or date_end is NULL, then use entire data for training
## length: length of forecast
## ci: confidence interval in %

time_series_forecast <- function(data,method='ets',date_start=NULL,date_end=NULL,
                                 length="14 days",ci=95){
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
    forecast(h = length) %>%
    hilo(level = ci) %>% 
    unpack_hilo(cols = paste0(ci,'%')) %>% 
    dplyr::select(model=.model,date,mean=.mean,
                  lower=paste0(ci,'%_lower'),upper=paste0(ci,'%_upper')) %>% 
    mutate(ci_level=ci)
  
  return(fc)
}