library(tidyverse)
library(httr)
library(globaldothealth)
library(here)

setwd(sprintf("%s/src/data_covid_cases", here()))
source("support.R")

## Instructions: save your API key in an untracked file named 'key.txt'
key <- readLines("key.txt")
country <- "NZ" ## using NZ for now as rate-limited for larger data sets

linelist <- get_cases(apikey = key, country = country)
cases <- get_cases_GdotH_API(linelist)

dir.create("outputs", FALSE, TRUE)

write_csv(cases, sprintf("outputs/cases_%s.csv", country))
g <- cases %>%
  ggplot(aes(x = date, y = cases)) +
  geom_line() +
  theme_bw()

ggsave(sprintf("outputs/cases_%s.png", country))
