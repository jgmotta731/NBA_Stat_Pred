#!/usr/bin/env Rscript

# Load Libraries
library(dplyr)
library(hoopR)
library(arrow)
library(stringr)
library(lubridate)
library(tidyr)
library(data.table)
library(purrr)
library(stringr)
library(tibble)
library(parallel)
library(rvest)
library(pdftools)

bref_injuries <- function() {
  url <- "https://www.basketball-reference.com/friv/injuries.fcgi"
  pg  <- read_html(url)
  tbl <- html_element(pg, "table") |> rvest::html_table()
  as_tibble(tbl) |>
    rename(player = Player, team = Team, updated = Update, description = Description) |>
    mutate(source = url)
}

injuries_bref <- bref_injuries()

# Season string like "2025-26" based on a date
.season_for_date <- function(d) {
  d <- as.Date(d)
  start_year <- ifelse(lubridate::month(d) >= 10, lubridate::year(d), lubridate::year(d) - 1)
  sprintf("%d-%02d", start_year, (start_year + 1) %% 100)
}

# Robust date parser for nba_schedule() output
.parse_sched_date <- function(df) {
  # Try several columns; keep only Date
  gd <- suppressWarnings(lubridate::as_date(df$game_date))
  if (all(is.na(gd)) && "game_date_time_utc" %in% names(df)) {
    gd <- suppressWarnings(lubridate::as_date(df$game_date_time_utc))
  }
  if (all(is.na(gd)) && "game_date_utc" %in% names(df)) {
    gd <- suppressWarnings(lubridate::as_date(df$game_date_utc))
  }
  gd
}

# Return a contiguous block of `days` games starting from the first upcoming day with games
stats_schedule_week <- function(
    anchor_date    = NULL,  # NULL -> later of today and Oct 21 of current year
    days           = 7,
    max_days_ahead = 365,
    verbose        = TRUE
){
  # Resolve anchor (later of "today" and Oct 21 this year)
  if (is.null(anchor_date) || is.na(anchor_date)) {
    yr    <- as.integer(format(Sys.Date(), "%Y"))
    oct21 <- as.Date(sprintf("%04d-10-21", yr))
    anchor_date <- max(Sys.Date(), oct21)
  } else {
    anchor_date <- as.Date(anchor_date)
  }
  
  season_str <- .season_for_date(anchor_date)
  
  # Pull the full season schedule once (no scoreboard endpoints)
  sched <- hoopR::nba_schedule(season = season_str)
  
  # Parse dates and keep only future window [anchor_date, anchor_date + max_days_ahead]
  sched <- sched %>%
    mutate(game_date = .parse_sched_date(cur_data())) %>%
    filter(!is.na(game_date),
           game_date >= anchor_date,
           game_date <= anchor_date + max_days_ahead) %>%
    arrange(game_date)
  
  if (!nrow(sched)) {
    if (verbose) message("No upcoming games within ", max_days_ahead, " days from ", as.character(anchor_date))
    return(tibble())
  }
  
  # Find the first day with games, then take that day + next (days-1)
  first_day <- min(sched$game_date, na.rm = TRUE)
  end_day   <- first_day + days - 1
  
  out <- sched %>%
    filter(game_date >= first_day, game_date <= end_day) %>%
    transmute(
      game_date,
      home_abbr = home_team_tricode,
      away_abbr = away_team_tricode
    ) %>%
    distinct() %>%
    arrange(game_date, home_abbr, away_abbr)
  
  if (verbose) {
    msg <- sprintf("Week window: %s to %s — %d game(s)",
                   as.character(first_day), as.character(end_day), nrow(out))
    message(msg)
  }
  
  out
}

# 0 = regular season (before Apr 12), 1 = playoffs (Apr 12 .. Sep 30)
is_playoff_flag <- function(dates, playoff_day = "04-12", next_season_day = "10-01") {
  dates <- as.Date(dates)
  yr  <- as.integer(format(dates, "%Y"))
  mo  <- as.integer(format(dates, "%m"))
  season_start_year <- yr - ifelse(mo < 10L, 1L, 0L)
  playoff_start     <- as.Date(paste0(season_start_year + 1L, "-", playoff_day))
  next_season_start <- as.Date(paste0(season_start_year + 1L, "-", next_season_day))
  as.integer(dates >= playoff_start & dates < next_season_start)
}

# Usage (Obtain current week's games)
next_slate <- stats_schedule_week() %>%
  mutate(is_playoff = is_playoff_flag(game_date)) %>%
  rename(home_abbreviation = home_abbr,
         away_abbreviation = away_abbr) %>%
  select(game_date, home_abbreviation, away_abbreviation, is_playoff)

# Player Gamelogs
player_raw <- load_nba_player_box(
  seasons = 2022:as.numeric(format(Sys.Date(), "%Y"))) %>%
  select(everything(), -game_date_time, -team_id,
         -team_name, -team_location, -team_short_display_name,
         -athlete_jersey, -athlete_short_name, -athlete_headshot_href,
         -athlete_position_name, -team_uid, -team_slug, -team_logo,
         -team_color, -team_alternate_color, -opponent_team_id,
         -opponent_team_name, -opponent_team_location,
         -opponent_team_display_name, -opponent_team_logo,
         -opponent_team_color, -opponent_team_alternate_color,
         -reason) %>%
  rename(player_id = athlete_id)

# Save files
write_parquet(player_raw, "datasets/nba_gamelogs.parquet")
write_parquet(next_slate, "datasets/nba_schedule.parquet")
write_parquet(injuries_bref, "datasets/current_injuries.parquet")

# Clear memory
rm(list = intersect(ls(), c("player_raw","next_slate", "injuries_bref")))
gc()

# ------------------ Load existing caches ------------------
existing_lineups <- if (file.exists("datasets/lineup_stints.parquet")) read_parquet("datasets/lineup_stints.parquet") else tibble()
existing_onoff   <- if (file.exists("datasets/onoff_player_game.parquet")) read_parquet("datasets/onoff_player_game.parquet") else tibble()
processed_game_ids <- unique(existing_lineups$game_id)

# ------------------ Lineup Data ------------------

detect_subs <- function(pbp) {
  pbp %>%
    filter(type_text == "Substitution") %>%
    transmute(
      game_id,
      period,
      clock_total = start_game_seconds_remaining,
      team_id,
      player_in = athlete_id_1,
      player_out = athlete_id_2,
      event_num = sequence_number
    ) %>%
    arrange(game_id, clock_total, team_id, event_num) %>%
    group_by(game_id, team_id, clock_total) %>%
    summarise(
      players_in = list(unique(player_in)),
      players_out = list(unique(player_out)),
      period = max(period),
      .groups = "drop"
    )
}

get_starting_lineups <- function(pbp) {
  pbp %>%
    filter(!is.na(athlete_id_1), type_text != "Substitution") %>%
    group_by(team_id, athlete_id_1) %>%
    summarise(first_event = min(sequence_number), .groups = "drop") %>%
    arrange(team_id, first_event) %>%
    group_by(team_id) %>%
    filter(!duplicated(athlete_id_1)) %>%
    slice_head(n = 5) %>%
    summarise(starters = list(athlete_id_1), .groups = "drop") %>%
    deframe()
}

generate_lineups <- function(pbp) {
  if (!"start_game_seconds_remaining" %in% names(pbp)) return(NULL)
  pbp <- pbp %>% filter(!is.na(start_game_seconds_remaining))
  if (nrow(pbp) == 0) return(NULL)
  
  pbp <- pbp %>%
    arrange(game_id, sequence_number) %>%
    mutate(clock_total = start_game_seconds_remaining)
  
  subs <- detect_subs(pbp)
  team_lineups <- get_starting_lineups(pbp)
  
  stint_log <- list()
  last_lineup_state <- team_lineups
  game_id <- unique(pbp$game_id)
  
  starting_clock <- max(pbp$clock_total, na.rm = TRUE)
  unique_clocks <- sort(unique(c(subs$clock_total, 0)), decreasing = TRUE)
  if (starting_clock != unique_clocks[1]) {
    unique_clocks <- c(starting_clock, unique_clocks)
  }
  
  for (i in seq_along(unique_clocks)[-length(unique_clocks)]) {
    last_clock <- unique_clocks[i]
    next_clock <- unique_clocks[i + 1]
    
    clock_subs <- subs %>% filter(clock_total == next_clock)
    if (nrow(clock_subs) > 0) {
      for (team in unique(clock_subs$team_id)) {
        team_subs <- clock_subs %>% filter(team_id == team)
        players_out <- unlist(team_subs$players_out)
        players_in <- unlist(team_subs$players_in)
        
        current_lineup <- last_lineup_state[[as.character(team)]]
        current_lineup <- setdiff(current_lineup, intersect(current_lineup, players_out))
        current_lineup <- union(current_lineup, setdiff(players_in, current_lineup))
        current_lineup <- sort(current_lineup)
        
        last_lineup_state[[as.character(team)]] <- current_lineup
      }
    }
    
    if (last_clock > next_clock) {
      for (team in names(last_lineup_state)) {
        lineup <- last_lineup_state[[team]]
        if (length(lineup) != 5) next
        
        stint_events <- pbp %>%
          filter(clock_total < last_clock & clock_total >= next_clock & team_id == as.integer(team))
        
        stint_log[[length(stint_log) + 1]] <- list(
          game_id = game_id,
          start_seconds = last_clock,
          end_seconds = next_clock,
          duration_seconds = last_clock - next_clock,
          team_id = team,
          lineup = paste(lineup, collapse = "-"),
          points = sum(stint_events$score_value, na.rm = TRUE),
          fga = sum(stint_events$shooting_play & !str_detect(stint_events$type_text, "Free Throw", TRUE), na.rm = TRUE),
          fgm = sum(stint_events$shooting_play & stint_events$scoring_play & !str_detect(stint_events$type_text, "Free Throw", TRUE), na.rm = TRUE),
          fg3a = sum(stint_events$shooting_play & str_detect(tolower(stint_events$text), "three point"), na.rm = TRUE),
          fg3m = sum(stint_events$shooting_play & stint_events$scoring_play & str_detect(tolower(stint_events$text), "three point"), na.rm = TRUE),
          turnovers = sum(str_detect(tolower(stint_events$type_text), "turnover"), na.rm = TRUE),
          assists = sum(str_detect(tolower(stint_events$text), "assist"), na.rm = TRUE),
          offensive_rebounds = sum(str_detect(tolower(stint_events$type_text), "offensive rebound"), na.rm = TRUE),
          defensive_rebounds = sum(str_detect(tolower(stint_events$type_text), "defensive rebound"), na.rm = TRUE)
        )
      }
    }
  }
  
  bind_rows(stint_log)
}

worker_fn <- function(pbp) {
  tryCatch({
    generate_lineups(pbp)
  }, error = function(e) NULL)
}

# MAIN LOOP
start_time <- Sys.time()
seasons <- c(2022, 2023, 2024, 2025)
all_lineups <- list()
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
clusterExport(cl, varlist = c("worker_fn", "generate_lineups", "detect_subs", "get_starting_lineups"))
clusterEvalQ(cl, {
  library(dplyr)
  library(stringr)
  library(tibble)
})

for (season in seasons) {
  message("Processing season ", season)
  pbp_season <- load_nba_pbp(season = season)
  pbp_split <- split(pbp_season, pbp_season$game_id)
  pbp_split <- pbp_split[!(names(pbp_split) %in% processed_game_ids)]  # Skip already processed games
  
  if (length(pbp_split) == 0) next
  
  season_results <- parLapply(cl, pbp_split, worker_fn)
  for (res in season_results) {
    if (!is.null(res)) all_lineups[[length(all_lineups) + 1]] <- res
  }
}

stopCluster(cl)
end_time <- Sys.time()
message("Processing completed in ", round(end_time - start_time, 2))

if (length(all_lineups) == 0 && nrow(existing_lineups) == 0) stop("No lineups were generated. Check substitution parsing.")

new_lineups <- bind_rows(all_lineups) %>%
  mutate(
    rebounds = offensive_rebounds + defensive_rebounds,
    duration_minutes = round(duration_seconds / 60, 1)
  ) %>%
  distinct() %>%
  filter(
    str_count(lineup, "-") == 4,
    duration_seconds >= 10,
    !(game_id %in% c("401752955", "401752956", "401752957"))
  )

lineup_stints <- bind_rows(existing_lineups, new_lineups)
write_parquet(lineup_stints, "datasets/lineup_stints.parquet")

# ------------------ On/Off Data ------------------

extract_player_ids <- function(df) {
  df %>%
    separate_rows(lineup, sep = "-") %>%
    rename(player_id = lineup)
}

lineup_long <- extract_player_ids(lineup_stints)

rosters_by_game_team <- lineup_long %>%
  distinct(game_id, team_id, player_id) %>%
  group_by(game_id, team_id) %>%
  summarise(all_team_players = list(unique(player_id)), .groups = "drop")

lineup_with_rosters <- lineup_stints %>%
  left_join(rosters_by_game_team, by = c("game_id", "team_id")) %>%
  mutate(lineup = str_split(lineup, "-"))

on_stats <- lineup_with_rosters %>%
  unnest(lineup) %>%
  rename(player_id = lineup) %>%
  group_by(game_id, team_id, player_id) %>%
  summarise(across(c(points, fga, fgm, fg3a, fg3m, offensive_rebounds, defensive_rebounds, assists, turnovers), sum), .groups = "drop") %>%
  rename_with(~ paste0("on_", .), -c(game_id, team_id, player_id))

off_stats <- lineup_with_rosters %>%
  rowwise() %>%
  mutate(off_players = list(setdiff(all_team_players, lineup))) %>%
  unnest(off_players) %>%
  group_by(game_id, team_id, player_id = off_players) %>%
  summarise(across(c(points, fga, fgm, fg3a, fg3m, offensive_rebounds, defensive_rebounds, assists, turnovers), sum), .groups = "drop") %>%
  rename_with(~ paste0("off_", .), -c(game_id, team_id, player_id))

onoff_summary_new <- full_join(on_stats, off_stats, by = c("game_id", "team_id", "player_id"))
onoff_summary <- bind_rows(existing_onoff, onoff_summary_new) %>% distinct()

arrow::write_parquet(onoff_summary, "datasets/onoff_player_game.parquet")