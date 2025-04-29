# app.R

library(shiny)
library(bslib)
library(reactable)
library(dplyr)
library(arrow)
library(htmltools)
library(stringr)

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
metrics_df <- read_parquet("evaluation_metrics.parquet")

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
      .navbar-brand { display: flex; align-items: center; }
      .navbar-brand img { margin-right: 10px; height: 40px; }
      .reactable .th, .reactable .td { color: #FFFFFF; }
    "))
  ),
  
  navbarPage(
    title = div(
      tags$img(src="nba_light.png", alt="Logo"), "NBA Player Predictor"
    ),
    theme = my_theme,
    
    # --- HOME TAB ---
    tabPanel(
      "Home",
      div(
        class="container-fluid",
        style="padding:2rem; background-color:#1e1e1e; color:white;",
        div(
          class="text-center mb-4",
          h1("Welcome to NBA Player Stat Predictor", class="display-4"),
          p("Get next‑game predictions for NBA players using machine learning models.")
        ),
        fluidRow(
          div(
            class="col-md-4", style="padding:1rem;",
            div(
              class="card h-100",
              style="background:#1e1e1e; border:1px solid #007AC1;",
              div(class="card-body",
                  h3("Advanced Analytics", class="card-title"),
                  p("Machine‑learning models trained on historical NBA data.")
              )
            )
          ),
          div(
            class="col-md-4", style="padding:1rem;",
            div(
              class="card h-100",
              style="background:#FFFFFF; border:1px solid #007AC1; color:#000000;",
              div(class="card-body",
                  h3("Weekly Updates", class="card-title text-primary"),
                  p("Fresh predictions before every game, with latest data.")
              )
            )
          ),
          div(
            class="col-md-4", style="padding:1rem;",
            div(
              class="card h-100",
              style="background:#1e1e1e; border:1px solid #007AC1;",
              div(class="card-body",
                  h3("Key Stats", class="card-title"),
                  p("Predictions for 3‑PT FG, Rebounds, Assists, Steals, Blocks, Points.")
              )
            )
          )
        ),
        div(
          style="text-align:center; margin-top:2rem;",
          tags$img(src="nba_dark.png", style="max-width:900px; width:100%;")
        )
      )
    ),
    
    # --- PREDICTIONS TAB ---
    tabPanel(
      "Predictions",
      div(
        class="container-fluid", style="padding:2rem; background:#121212;",
        fluidRow(
          column(
            width=2,  # narrowed sidebar
            div(
              style="background:#1e1e1e; border:1px solid #007AC1; padding:1rem;",
              h4("Implied Probability Calculator", style="color:#FFFFFF;"),
              textInput("american_odds", "American Odds:", "", width="100%"),
              verbatimTextOutput("implied_prob", placeholder=TRUE)
            )
          ),
          column(
            width=10,  # expanded main area
            div(
              class="card",
              style="background:#1e1e1e; border:1px solid #007AC1; padding:1rem;",
              h2("Weekly Player Predictions", style="color:white;")
            ),
            reactableOutput("predictions_table")
          )
        )
      )
    ),
    
    # --- METRICS TAB ---
    tabPanel(
      "Metrics",
      div(
        class="container-fluid",
        style="padding:2rem; background:#121212;",
        h4("How to Interpret These Metrics", style="color: #FFFFFF;"),
        p(
          style="color:#DDDDDD;",
          strong("Root Mean Squared Error (RMSE):"),
          "On average, predictions are this many units off. E.g., an RMSE of 3 on points means we’re typically within 3 points."
        ),
        p(
          style="color:#DDDDDD;",
          strong("R‑squared (R²):"),
          "Proportion of variance explained. An R² of 0.5 means we capture about half of the game‑to‑game variability."
        ),
        reactableOutput("metrics_table"),
        div(
          style="text-align:center; margin-top:2rem;",
          tags$img(src="nba_dark.png", style="max-width:600px; width:100%;")
        )
      )
    )
  )
)

# ---------------------------------------------------
# Server
# ---------------------------------------------------
server <- function(input, output, session) {
  # --- reactive reader for the parquet file, now every 12 hours ---
  preds_df <- reactiveFileReader(
    intervalMillis = 1 * 60 * 60 * 1000,
    session        = session,
    filePath       = "nba_predictions.parquet",
    readFunc       = arrow::read_parquet
  )
  
  output$implied_prob <- renderText({
    req(input$american_odds)
    odds <- as.numeric(input$american_odds)
    if (is.na(odds)) return("Enter valid odds.")
    prob <- if (odds < 0) abs(odds)/(abs(odds)+100)*100 else 100/(odds+100)*100
    paste0("Implied Probability: ", round(prob,1), "%")
  })
  
  output$predictions_table <- renderReactable({
    df <- preds_df() %>%
      mutate(across(starts_with("predicted_"), ~ round(.x,1))) %>%
      mutate(
        Player = paste0(
          "<div style='text-align:center;'>",
          "<img src='", headshot_url, "' height='60' style='border-radius:50%;'><br>",
          athlete_display_name, "</div>"
        )
      ) %>%
      select(
        Player,
        Team     = team_abbreviation,
        Opponent = opponent_team_abbreviation,
        Date     = game_date,
        HomeAway = home_away,
        `3‑Point FG` = predicted_three_point_field_goals_made,
        Rebounds      = predicted_rebounds,
        Assists       = predicted_assists,
        Steals        = predicted_steals,
        Blocks        = predicted_blocks,
        Points        = predicted_points
      )
    
    reactable(
      df,
      searchable      = TRUE,
      filterable = TRUE,
      defaultPageSize = 7,
      defaultSorted   = "Date",
      highlight       = TRUE,
      compact         = TRUE,
      columns = list(
        Player   = colDef(html=TRUE, align="center", minWidth=120),
        Team     = colDef(align="center", minWidth=60),
        Opponent = colDef(align="center", minWidth=100),
        Date     = colDef(align="center", minWidth=90),
        HomeAway = colDef(name="Home/Away", align="center", minWidth=100),
        `3‑Point FG` = colDef(align="right"),
        Rebounds     = colDef(align="right"),
        Assists      = colDef(align="right"),
        Steals       = colDef(align="right"),
        Blocks       = colDef(align="right"),
        Points       = colDef(align="right")
      ),
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
      defaultPageSize = 6,
      highlight       = TRUE,
      compact         = TRUE,
      columns = list(
        Metric = colDef(align="left"),
        RMSE   = colDef(format=colFormat(digits=1), align="right"),
        R2     = colDef(name="R²", format=colFormat(digits=2), align="right")
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