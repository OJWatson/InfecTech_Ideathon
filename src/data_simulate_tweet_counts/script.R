library(here)
library(tidyverse)
library(lubridate)
library(countrycode)
library(sf)


setwd(sprintf("%s/src/data_simulate_tweet_counts", here()))
## Instructions: save the vax_tweets.csv within this directory to run the task
raw_data <- read.csv("vax_tweets.csv", row.names = 1) %>%
  dplyr::mutate(date = lubridate::as_date(round(lubridate::dmy_hm(date), "day")))

# add common alts for popular locations
countryname_dict <- countrycode::countryname_dict %>%
  dplyr::add_row(country.name.en = "United Kingdom", country.name.alt = "UK") %>%
  dplyr::add_row(country.name.en = "United Kingdom", country.name.alt = "U.K") %>%
  dplyr::add_row(country.name.en = "United Kingdom", country.name.alt = "Scotland") %>%
  dplyr::add_row(country.name.en = "United Kingdom", country.name.alt = "Northern Ireland") %>%
  dplyr::add_row(country.name.en = "United Kingdom", country.name.alt = "Wales") %>%
  dplyr::add_row(country.name.en = "United Kingdom", country.name.alt = "Britain") %>%
  dplyr::add_row(country.name.en = "United States", country.name.alt = "[[:space:]]US") %>%
  dplyr::add_row(country.name.en = "United States", country.name.alt = "^US[[:space:]]") %>%
  dplyr::add_row(country.name.en = "United States", country.name.alt = "^US$") %>%
  dplyr::add_row(country.name.en = "United States", country.name.alt = "U.S") %>%
  dplyr::add_row(country.name.en = "United States", country.name.alt = "U.S.A") %>%
  dplyr::add_row(country.name.en = "United States", country.name.alt = "America") %>%
  dplyr::add_row(country.name.en = "United States", country.name.alt = "USA") %>%
  dplyr::add_row(country.name.en = "Palestinian Territories", country.name.alt = "Palestine")
rownames(countryname_dict) <- countryname_dict$country.name.alt

# get unique country names
country_names <- unique(countryname_dict$country.name.en)


# create regex for common alternative spellings
country_pattern <- sapply(country_names, function(x) {
  alt_names <- c(x, subset(countryname_dict, country.name.en == x, country.name.alt)[[1]])
  paste0(c(alt_names, tolower(alt_names)), collapse = "|")
})
names(country_pattern) <- country_names


# extract unique user locations
user_locations <- unique(raw_data$user_location)
length(user_locations) # 16k to assign...
# map user locations at country level - takes a while!
location_hits <- sapply(country_pattern, grepl, x = user_locations,
                        ignore.case = FALSE)

table(rowSums(location_hits)) # check how we have done - not bad, 11k still missing

# infer country, choosing most likely if non-unique
location_ranking <- sort(colSums(location_hits), decreasing = TRUE)
location_hits <- location_hits[, names(location_ranking)]
user_countries <- apply(location_hits, 1, function(x) names(which(x))[1])
names(user_countries) <- user_locations



## Round 2: include states more broadly
# make dictionary mapping state names to countries
state_dict <- data.frame(sf::st_as_sf(rnaturalearthhires::states10)) %>%
  dplyr::select(name, country.name.alt = admin) %>%
  dplyr::left_join(countryname_dict) %>%
  dplyr::mutate(country.name.en = if_else(is.na(country.name.en), country.name.alt, country.name.en))

# create state pattern
state_pattern <- sapply(country_names, function(x) {
  alt_names <- subset(state_dict, (country.name.en == x), name)
  paste0(alt_names[[1]], collapse = "|")
})
names(state_pattern) <- country_names
state_pattern <- state_pattern[nchar(state_pattern) > 0]

# Include US states in country pattern
us_states <- c(paste0("[[:space:]]", state.abb), paste0("^", state.abb, "$"))
state_pattern['United States'] <- paste0(c(state_pattern['United States'], us_states), collapse = "|")

# extract user locations that are still missing
user_locations2 <- user_locations[is.na(user_countries)]
length(user_locations2) # 11k still to go

location_hits2 <- sapply(state_pattern, grepl, x = user_locations2,
                         ignore.case = FALSE)
table(rowSums(location_hits2)) # check how we have done - 4k still missing
location_ranking2 <- location_ranking[names(location_ranking) %in% colnames(location_hits2)]
location_hits2 <- location_hits2[, names(location_ranking2)]
colSums(location_hits2)

user_countries2 <- apply(location_hits2, 1, function(x) names(which(x))[1])
names(user_countries2) <- user_locations2
user_countries[user_locations2] <- user_countries2
sum(is.na(user_countries)) # 5.5k to go

## Round 3: include cities

# get city names from maps package
# country names in maps do not match those in countrycode: need to create a mapping
maps_dict <- data.frame(maps_country_name = unique(maps::world.cities$country.etc)) %>%
  dplyr::mutate(countrycode_country_name = if_else(
    maps_country_name %in% countryname_dict$country.name.en,
    true = maps_country_name,
    false = countryname_dict[maps_country_name, "country.name.en"])) %>%
  dplyr::arrange(maps_country_name)

rownames(maps_dict) <- maps_dict$maps_country_name
maps_dict %>% # locations missing mapping are very small
  dplyr::filter(is.na(countrycode_country_name))


# map city names to countrycode countrynames
cityname_dict <- maps::world.cities %>%
  dplyr::mutate(country.name.en = maps_dict[country.etc, "countrycode_country_name"]) %>%
  dplyr::filter(nchar(name) > 3, # reduce misclassification from short strings
                pop > 3e4) # include only larger places

# extract user locations that are still missing
user_locations3 <- user_locations[is.na(user_countries)]

city_pattern <- sapply(country_names, function(x) {
  alt_names <- subset(cityname_dict, (country.name.en == x), name)
  paste0(alt_names[[1]], collapse = "|")
})

names(city_pattern) <- country_names
city_pattern <- city_pattern[nchar(city_pattern) > 0]
location_hits3 <- sapply(city_pattern, grepl, x = user_locations3,
                        ignore.case = TRUE)

table(rowSums(location_hits3)) # check how we have done - 3.8k still missing


location_ranking3 <- location_ranking[names(location_ranking) %in% colnames(location_hits3)]
location_hits3 <- location_hits3[, names(location_ranking3)]
colSums(location_hits3)


user_countries3 <- apply(location_hits3, 1, function(x) names(which(x))[1])
names(user_countries3) <- user_locations3

user_countries[user_locations3] <- user_countries3

data <- raw_data %>%
  dplyr::mutate(user_country = user_countries[user_location])

dir.create("outputs", FALSE, TRUE)
write.csv(data, "outputs/vax_tweets_with_country.csv")

g <- data.frame(user_location = names(user_countries),
                user_country = user_countries) %>%
  dplyr::count(user_country) %>%
  dplyr::filter(n > 10) %>%
  ggplot(aes(y = reorder(user_country, -n, decreasing = TRUE), x = n)) +
  geom_bar(stat = "identity") +
  theme_bw() +
  labs(x = "Number of users in data (capped at > 10)", y = "Country")

ggsave("outputs/user_country.png", g, height = 23, unit = "cm")


# to run this task please save the vaccine tweets provided for the challenge as
# vax_tweets.csv within this directory



pattern <- "death|dead|die|dying|deceased"

death_data_by_country <- data %>%
  dplyr::filter(!is_retweet,
                grepl(pattern, text, ignore.case = TRUE)) %>%
  dplyr::group_by(date, user_country) %>%
  dplyr::count()

write_csv(death_data_by_country, "outputs/death_data_by_country.csv")

death_data <- death_data_by_country%>%
  dplyr::group_by(date) %>%
  dplyr::summarise(n = sum(n))

write_csv(death_data, "outputs/death_data.csv")

g <- death_data_by_country %>%
  dplyr::filter(user_country %in% c("United States", "United Kingdom", "India")) %>%
  ggplot(aes(x = date, y = n, col = user_country)) +
  geom_line() +
  theme_bw()
ggsave("outputs/death_data_by_country.png", g)

g <- death_data %>%
  ggplot(aes(x = date, y = n)) +
  geom_line() +
  theme_bw()
ggsave("outputs/death_data.png", g)

