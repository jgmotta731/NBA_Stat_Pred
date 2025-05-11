#!/usr/bin/env Rscript

library(dplyr)
library(hoopR)
library(arrow)
library(stringr)

player_raw <- load_nba_player_box(
  seasons = 2020:as.numeric(format(Sys.Date(), "%Y"))) %>%
  select(everything(), -game_date_time, -athlete_id, -team_id,
         -team_name, -team_location, -team_short_display_name,
         -athlete_jersey, -athlete_short_name, -athlete_headshot_href,
         -athlete_position_name, -team_uid, -team_slug, -team_logo,
         -team_color, -team_alternate_color, -opponent_team_id,
         -opponent_team_name, -opponent_team_location,
         -opponent_team_display_name, -opponent_team_logo,
         -opponent_team_color, -opponent_team_alternate_color,
         -reason)

schedule <- load_nba_schedule(as.numeric(format(Sys.Date(), "%Y"))) %>%
  select(home_abbreviation, away_abbreviation, game_date, season_type) %>%
  filter(game_date >= Sys.Date()) %>%
  mutate(
    home_abbreviation = case_when(
      home_abbreviation == "Heat/Hawks" ~ "MIA",
      home_abbreviation == "Mavericks/Grizzlies" ~ "MEM",
      TRUE ~ home_abbreviation),
    away_abbreviation = case_when(
      away_abbreviation == "Heat/Hawks" ~ "MIA",
      away_abbreviation == "Mavericks/Grizzlies" ~ "MEM",
      TRUE ~ away_abbreviation),
    is_playoff = ifelse(season_type %in% c(3, 5), 1, 0)
    ) %>%
  select(-season_type)

write_parquet(player_raw, "nba_gamelogs.parquet")
write_parquet(schedule, "nba_schedule.parquet")