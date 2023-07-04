library(here)
library(tidyverse)
library(lubridate)

setwd(sprintf("%s/src/data_simulated_tweet_counts", here()))

# to run this task please save the vaccine tweets provided for the challenge as
# vax_tweets.csv within this directory

raw_data <- read.csv("vax_tweets.csv", row.names = 1) %>%
  dplyr::mutate(date = lubridate::as_date(round(lubridate::dmy_hm(date), "day")))

pattern <- "death|dead|die|dying|deceased"
data <- raw_data %>%
  dplyr::filter(!is_retweet,
                grepl(pattern, text, ignore.case = TRUE)) %>%
  dplyr::group_by(date) %>%
  dplyr::count()

dir.create("outputs", FALSE, TRUE)
write_csv(data, "outputs/data.csv")

g <- data %>%
  ggplot(aes(x = date, y = n)) +
  geom_line() +
  theme_bw()

ggsave("outputs/data.png")

