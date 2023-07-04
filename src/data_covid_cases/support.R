## getting and aggregating case data from global.health
get_cases_GdotH_API <- function(cases, age_range = NULL, date_min = NULL, date_max = NULL){
  cases %>%
    dplyr::select("_id","demographics.ageRange.end","demographics.ageRange.start",
                  "events.confirmed.date","events.hospitalAdmission.date",
                  "events.icuAdmission.date","events.onsetSymptoms.date",
                  "events.outcome.date","events.selfIsolation.date",
                  "events.outcome.value",
                  "location.administrativeAreaLevel1",
                  "location.administrativeAreaLevel2",
                  "location.administrativeAreaLevel3",
                  "location.country") %>%
    mutate(age_range=paste0(demographics.ageRange.start,"-",demographics.ageRange.end))

  # not all data for every country available

  # need to check event date to be used, default should be events.confirmed.date
  # need to check whether age range available in demographics.ageRange.end & demographics.ageRange.start

  # aggregating by default: by confirmed date
  if (is.null(age_range)){
    cases_aggregated <- cases %>%
      group_by(events.confirmed.date) %>%
      summarise(cases=n()) %>% ungroup() %>%
      drop_na() %>%
      dplyr::select(date=events.confirmed.date, cases)

    if (is.null(date_min)) date_min <- min(cases_aggregated$date, na.rm = TRUE)
    if (is.null(date_max)) date_max <- max(cases_aggregated$date, na.rm = TRUE)

    date_range_df <- data.frame(date = seq.Date(as.Date(date_min), as.Date(date_max), "day"))

    cases_aggregated <- date_range_df %>% left_join(cases_aggregated) %>%
      replace_na(list(cases=0))
  } else {
    cases_aggregated <- NULL
  }

  return(cases_aggregated)
}
