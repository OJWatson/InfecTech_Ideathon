library(tidyverse)

#### AND let's get the JHU as well/instead as looks liek it is less susceptible to blips

download_url <- function(url) {
  tryCatch({
    tf <- tempfile()
    code <- download.file(url, tf, mode = "wb")
    if (code != 0) {
      stop("Error downloading file")
    }
  },
  error = function(e) {
    stop(sprintf("Error downloading file '%s': %s, please check %s",
                 url, e$message))
  })
  return(tf)
}


## Get the worldometers data from JHU
jhu_url <- "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
jhu_tf <- download_url(jhu_url)
data <- read.csv(jhu_tf)

# format into the same style as ecdc so easy to swap back and forth
data$countryterritoryCode <- suppressWarnings(countrycode::countrycode(data$Country.Region, "country.name.en", "iso3c",
                                                                       custom_match = c(Kosovo = "KSV")))
data <- data %>% tidyr::pivot_longer(matches("X\\d"))
names(data) <- c("", "Region","lat","lon","countryterritoryCode","date","deaths")

data <- data[,c("date","deaths","countryterritoryCode","Region")]
data$date <- as.Date(data$date, format = "X%m.%d.%y")

# and into daily deaths nationally
data <- group_by(data, date, countryterritoryCode, Region) %>%
  summarise(deaths = sum(deaths, na.rm = TRUE))
data <- group_by(data, countryterritoryCode, Region) %>%
  mutate(deaths = c(0, diff(deaths)))

# now the same for cases
jhu_url <- "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
jhu_tf <- download_url(jhu_url)
cases <- read.csv(jhu_tf)

# format into the same style as ecdc so easy to swap back and forth
cases$countryterritoryCode <- suppressWarnings(countrycode::countrycode(cases$Country.Region, "country.name.en", "iso3c",
                                                                        custom_match = c(Kosovo = "KSV")))
cases <- cases %>% tidyr::pivot_longer(matches("X\\d"))
names(cases) <- c("", "Region","lat","lon","countryterritoryCode","date","cases")

cases <- cases[,c("date","cases","countryterritoryCode","Region")]
cases$date <- as.Date(cases$date, format = "X%m.%d.%y")

# and into daily cases nationally
cases <- group_by(cases, date, countryterritoryCode, Region) %>%
  summarise(cases = sum(cases, na.rm = TRUE))

cases <- group_by(cases, countryterritoryCode, Region) %>%
  mutate(cases = c(0, diff(cases)))

jhu_data <- left_join(data, cases, by = c("date", "countryterritoryCode", "Region"))
jhu_data$dateRep <- jhu_data$date

if(sum(jhu_data$deaths)>0){
  jhu_data$deaths[jhu_data$deaths < 0] <- 0
}

#spread out tanzania spike
if(jhu_data %>%
   filter(countryterritoryCode == "TZA", dateRep == "2021-10-01") %>%
   pull(deaths) == 669){
  #split up over previous and have it linearly increase
  new_values <- round(seq(0, 2/14, length.out = 14)*669)
  jhu_data <- jhu_data %>%
    mutate(
      deaths = if_else(
        countryterritoryCode == "TZA" & dateRep == "2021-10-01",
        0,
        deaths
      )
    )
  jhu_data[jhu_data$countryterritoryCode == "TZA" & (jhu_data$dateRep %in%
                                                       (as.Date("2021-10-01") - 0:13)) &
             !is.na(jhu_data$countryterritoryCode),
           "deaths"] <-jhu_data[jhu_data$countryterritoryCode == "TZA" &
                                  (jhu_data$dateRep %in% (as.Date("2021-10-01") - 0:13)) &
                                  !is.na(jhu_data$countryterritoryCode),"deaths"] +
    new_values
}


# save
dir.create("analysis/data/data-derived/", recursive = TRUE)
saveRDS(jhu_data, "analysis/data/data-derived/jhu_all.rds")
