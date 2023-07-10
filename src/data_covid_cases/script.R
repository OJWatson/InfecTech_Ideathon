library(tidyverse)
library(httr)
library(globaldothealth)
library(here)

setwd(sprintf("%s/src/data_covid_cases", here()))
source("support.R")

## Instructions: save your API key in an untracked file named 'key.txt'
key <- readLines("key.txt")
country <- "GB" ## using NZ for now as rate-limited for larger data sets

linelist <- get_cases(apikey = key, country = country)
cases <- get_cases_GdotH_API(linelist)

dir.create("outputs", FALSE, TRUE)

write_csv(cases, sprintf("outputs/cases_%s.csv", country))
g <- cases %>%
  ggplot(aes(x = date, y = cases)) +
  geom_line() +
  theme_bw() +
  labs(x = "", y = "UK Weekly Covid-19 Cases (000s)") +
  scale_x_date(date_labels = "%b %y", date_breaks = "2 month") +
  scale_y_continuous(labels = scales::unit_format(scale = 1e-3, suffix = "")) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))

ggsave(sprintf("outputs/cases_%s.png", country), width = 10, height = 8, unit = "cm")
