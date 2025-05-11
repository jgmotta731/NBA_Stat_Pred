# app.R

library(shiny)
library(bslib)
library(reactable)
library(dplyr)
library(arrow)
library(htmltools)
library(stringr)
library(sortable)


# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
preds_df <- read_parquet("nba_predictions.parquet") %>%
  mutate(across(starts_with("predicted_"), ~ round(.x,1)),
         home_away = str_to_sentence(home_away)) %>%
  arrange(game_date, team_abbreviation, athlete_display_name)

metrics_df <- read_parquet("evaluation_metrics.parquet")

# Rename 'target' values: replace underscores with spaces and apply title case
metrics_df$target <- metrics_df$target %>%
  stringr::str_replace_all("_", " ") %>%
  stringr::str_to_title()

# ---------------------------------------------------
# Theme
# ---------------------------------------------------
my_theme <- bs_theme(
  version       = 5,
  bg            = "#121212",
  fg            = "#FFFFFF",
  primary       = "#007AC1",
  base_font     = font_google("Inter"),
  heading_font  = font_google("Roboto"),
  "navbar-bg"             = "#121212",
  "navbar-dark-color"     = "#FFFFFF",
  "navbar-dark-hover-color"= "#007AC1"
)

# ---------------------------------------------------
# UI
# ---------------------------------------------------
ui <- tagList(
  tags$head(
    tags$style(HTML("
      .navbar {
        min-height: 64px;
        padding-top: 3px;
        padding-bottom: 3px;
      }

      .navbar-brand {
        display: flex;
        align-items: center;
        font-size: 1.4rem;
      }

      .navbar-brand img {
        margin-right: 10px;
        height: 42px;
      }

      .navbar-nav > li > a {
        font-size: 1.0rem !important;
        font-weight: 500;
      }

      .card {
        display: flex;
        flex-direction: column;
        height: 100%;
      }

      .card-body {
        flex: 1;
      }

      .reactable .th, .reactable .td {
        color: #FFFFFF;
      }

      .reactable input[type='text'],
      .reactable select {
        color: black !important;
      }
    "))
  ),
  
  navbarPage(
    title = div(
      tags$img(src = "nba_light.png", alt = "Logo"), 
      "NBA Player Predictor"
    ),
    theme = my_theme,
    
    # --- HOME TAB ---
    tabPanel(
      "Home",
      div(
        class = "container-fluid",
        style = "padding:2rem; background-color:#1e1e1e; color:white;",
        div(
          class = "text-center mb-4",
          h1("Welcome to NBA Player Stat Predictor", class = "display-4"),
          p("Get next‑game predictions for NBA players using machine learning models.")
        ),
        fluidRow(
          div(
            class = "col-md-4", style = "padding:1rem;",
            div(
              class = "card h-100",
              style = "background:#1e1e1e; border:1px solid #007AC1;",
              div(
                class = "card-body",
                h3("Advanced Analytics", class = "card-title"),
                p("Machine‑learning models trained on historical NBA data.")
              )
            )
          ),
          div(
            class = "col-md-4", style = "padding:1rem;",
            div(
              class = "card h-100",
              style = "background:#FFFFFF; border:1px solid #007AC1; color:#000000;",
              div(
                class = "card-body",
                h3("Weekly Updates", class = "card-title text-primary"),
                p("Fresh predictions before every game, with latest data and scraped over-under odds from FanDuel via Odds API.")
              )
            )
          ),
          div(
            class = "col-md-4", style = "padding:1rem;",
            div(
              class = "card h-100",
              style = "background:#1e1e1e; border:1px solid #007AC1;",
              div(
                class = "card-body",
                h3("Key Stats", class = "card-title"),
                p("Predictions for 3‑PT FG, Rebounds, Assists, Steals, Blocks, Points.")
              )
            )
          )
        ),
        div(
          style = "text-align:center; margin-top:2rem;",
          tags$img(
            src = "nba_dark.png",
            style = "width: 80%; max-width: 515px; height: auto; margin: 0 auto; display: block;"
          )
        )
      )
    ),

    # --- PREDICTIONS TAB ---
    tabPanel(
      "Predictions",
      div(
        class = "container-fluid",
        style = "padding:0rem; background:#121212;",
        fluidRow(
          # Sidebar: Calculator + Column Selector
          column(
            width = 2,
            div(
              style = "background:#1e1e1e; border:1px solid #007AC1; padding:1rem;",
              
              # Implied Probability Calculator
              h4("Implied Probability Calculator", style = "color:#FFFFFF;"),
              textInput("american_odds", "American Odds:", "", width = "100%"),
              verbatimTextOutput("implied_prob", placeholder = TRUE),
              
              # Space
              tags$hr(style = "border-top: 1px solid #007AC1;"),
              
              # Column Selector (smaller + black text)
              div(
                style = "margin-top:1rem; color:white;",
                h4("Select Columns", style = "color:white;"),
                checkboxGroupInput(
                  inputId = "selected_columns",
                  label = NULL,
                  choices = c(
                    "Player", "Team", "Opponent", "Date", "HomeAway",
                    "3‑Point FG", "Rebounds", "Assists", "Steals", "Blocks", "Points"),
                  selected = c(
                    "Player", "Team", "Opponent", "Date", "HomeAway",
                    "3‑Point FG", "Rebounds", "Assists", "Steals", "Blocks", "Points"
                  )
                )
              )
            )
          ),
          
          # Main Table
          column(
            width = 10,
            div(
              class = "card",
              style = "background:#1e1e1e; border:1px solid #007AC1; padding:1rem;",
              h2("Weekly Player Predictions", style = "color:white;"),
              reactableOutput("predictions_table")
            )
          )
        )
      )
    ),
    
    # --- METRICS TAB ---
    tabPanel(
      "Metrics",
      div(
        class = "container-fluid",
        style = "padding:0rem; background:#121212;",
        h4("How to Interpret These Metrics", style = "color:#FFFFFF;"),
        p(
          style = "color:#DDDDDD;",
          strong("Root Mean Squared Error (RMSE):"),
          "Measures the typical size of prediction errors, giving extra weight to large mistakes. A higher RMSE suggests occasional big misses are present."
        ),
        p(
          style = "color:#DDDDDD;",
          strong("Mean Absolute Error (MAE):"),
          "Measures the average size of all prediction errors, treating every miss equally. A lower MAE means the model is consistently close, even if not perfect."
        ),
        p(
          style = "color:#DDDDDD;",
          strong("R‑squared (R²):"),
          "Proportion of variance explained. An R² of 0.5 means we capture about half of the game‑to‑game variability. Shows how much better the model is at predicting compared to simply guessing the average every time."
        ),
        p(
          style = "color:#DDDDDD;",
          strong("Pinball Loss (τ = 0.1):"),
          "Measures how well the model predicts the 10th percentile. A lower value means the model is effectively avoiding over-predictions, making it useful for conservative forecasting."
        ),
        p(
          style = "color:#DDDDDD;",
          strong("Pinball Loss (τ = 0.9):"),
          "Measures how well the model predicts the 90th percentile. A lower value here means the model is good at avoiding under-predictions, useful for capturing upper-bound expectations."
        ),
        reactableOutput("metrics_table"),
        div(
          style = "text-align:center; margin-top:2rem; margin-bottom:2rem;",
          tags$img(
            src = "nba_dark.png",
            style = "max-width:420px; height: auto; width:100%;"
          )
        )
      )
    )
  ),
  tags$footer(
    style = "background-color:#1e1e1e; color:#AAAAAA; text-align:center; padding:1rem; font-size:0.85rem; border-top:1px solid #007AC1;",
    HTML("
    <strong>Disclaimer:</strong> This application provides NBA player predictions for entertainment and informational purposes only. 
    There is no guarantee of accuracy, and no responsibility is assumed for financial losses or decisions made based on this data. 
    Use responsibly and follow all local laws and regulations related to sports betting.
  ")
  )
)


# ---------------------------------------------------
# Server
# ---------------------------------------------------
server <- function(input, output, session) {
  output$implied_prob <- renderText({
    req(input$american_odds)
    odds <- as.numeric(input$american_odds)
    if (is.na(odds)) return("Enter valid odds.")
    prob <- if (odds < 0) abs(odds)/(abs(odds)+100)*100 else 100/(odds+100)*100
    paste0("Implied Probability: ", round(prob,1), "%")
  })
  
  # Predictions Table
  output$predictions_table <- renderReactable({
    req(input$selected_columns) # Must have at least one selected
    
    df <- preds_df %>%
      mutate(
        Player = paste0(
          "<div style='text-align:center;'>",
          "<img src='", headshot_url, "' height='60' style='border-radius:50%;'><br>",
          athlete_display_name, "</div>"
        ),
        `3‑Point FG` = predicted_three_point_field_goals_made,
        Rebounds     = predicted_rebounds,
        Assists      = predicted_assists,
        Steals       = predicted_steals,
        Blocks       = predicted_blocks,
        Points       = predicted_points,
        Team         = team_abbreviation,
        Opponent     = opponent_team_abbreviation,
        Date         = game_date,
        HomeAway     = home_away
      )
    
    df_to_display <- df[, input$selected_columns, drop = FALSE]
    
    # Dynamically define columns only for selected ones
    col_defs <- list()
    if ("Player" %in% input$selected_columns) {
      col_defs$Player <- colDef(html = TRUE, align = "center", minWidth = 120)
    }
    if ("Team" %in% input$selected_columns) {
      col_defs$Team <- colDef(align = "center", minWidth = 60)
    }
    if ("Opponent" %in% input$selected_columns) {
      col_defs$Opponent <- colDef(align = "center", minWidth = 100)
    }
    if ("Date" %in% input$selected_columns) {
      col_defs$Date <- colDef(align = "center", minWidth = 90)
    }
    if ("HomeAway" %in% input$selected_columns) {
      col_defs$HomeAway <- colDef(name = "Home/Away", align = "center", minWidth = 110)
    }
    if ("3‑Point FG" %in% input$selected_columns) {
      col_defs[["3‑Point FG"]] <- colDef(align = "right")
    }
    if ("Rebounds" %in% input$selected_columns) {
      col_defs$Rebounds <- colDef(align = "right")
    }
    if ("Assists" %in% input$selected_columns) {
      col_defs$Assists <- colDef(align = "right")
    }
    if ("Steals" %in% input$selected_columns) {
      col_defs$Steals <- colDef(align = "right")
    }
    if ("Blocks" %in% input$selected_columns) {
      col_defs$Blocks <- colDef(align = "right")
    }
    if ("Points" %in% input$selected_columns) {
      col_defs$Points <- colDef(align = "right")
    }
    
    reactable(
      df_to_display,
      searchable      = TRUE,
      pagination = FALSE,
      filterable = TRUE,
      highlight       = TRUE,
      compact         = TRUE,
      columns = col_defs,
      theme = reactableTheme(
        style    = list(background = "#121212"),
        rowStyle = list(borderBottom = "1px solid #007AC1")
      )
    )
  })
  
  output$metrics_table <- renderReactable({
    m <- metrics_df %>%
      rename(Metric = target, RMSE = rmse, R2 = r2)
    reactable(
      m,
      pagination = FALSE,
      highlight = TRUE,
      compact = TRUE,
      columns = list(
        Metric = colDef(name="Target", align="left"),
        RMSE   = colDef(format=colFormat(digits=1), align="right"),
        mae    = colDef(name="MAE", format=colFormat(digits=1), align="right"),
        R2     = colDef(name="R²", format=colFormat(digits=2), align="right"),
        quantile_loss_under = colDef(name="Pinball Loss (Tau=0.1)", format=colFormat(digits=1), align="right"),
        quantile_loss_over = colDef(name="Pinball Loss (Tau=0.9)", format=colFormat(digits=1), align="right")
      ),
      theme = reactableTheme(
        style    = list(background = "#121212"),
        rowStyle = list(borderBottom = "1px solid #007AC1")
      )
    )
  })
}

# ---------------------------------------------------
# Run the app
# ---------------------------------------------------
shinyApp(ui, server)