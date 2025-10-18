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
# Load Data Helpers
# ---------------------------------------------------
# Paths via env (fallback to defaults if not set)
PREDICTIONS_DIR <- Sys.getenv("PREDICTIONS_DIR", "predictions")
METRICS_PATH    <- Sys.getenv("METRICS_PATH", "datasets/Evaluation_Metrics.parquet")

latest_predictions_path <- function(
  dir = PREDICTIONS_DIR,
  pattern = "^nba_predictions_\\d{4}-\\d{2}-\\d{2}\\.parquet$"
) {
  if (!dir.exists(dir)) return(NA_character_)
  files <- list.files(dir, full.names = TRUE, pattern = pattern)
  if (length(files) == 0) return(NA_character_)
  info <- file.info(files)
  files[which.max(info$mtime)]
}

# ---------------------------------------------------
# Theme (UNCHANGED)
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
# UI (UNCHANGED)
# ---------------------------------------------------
ui <- tagList(
  tags$head(
    tags$style(HTML("
      .navbar { min-height: 64px; padding-top: 3px; padding-bottom: 3px; }
      .navbar-brand { display: flex; align-items: center; font-size: 1.4rem; }
      .navbar-brand img { margin-right: 10px; height: 42px; }
      .navbar-nav > li > a { font-size: 1.0rem !important; font-weight: 500; }
      .card { display: flex; flex-direction: column; height: 100%; }
      .card-body { flex: 1; }
      .reactable .th, .reactable .td { color: #FFFFFF; }
      .reactable input[type='text'], .reactable select { color: black !important; }
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
          p("Get next-game predictions for NBA players using machine learning models.")
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
                p("Machine-learning models trained on historical NBA data.")
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
                p("Predictions for 3-PT FG, Rebounds, Assists, Steals, Blocks, Points.")
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
                    "3-Point FG", "Rebounds", "Assists", "Steals", "Blocks", "Points"),
                  selected = c(
                    "Player", "Team", "Opponent", "Date", "HomeAway",
                    "3-Point FG", "Rebounds", "Assists", "Steals", "Blocks", "Points"
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
              div(
                class = "text-muted",
                style = "font-size:0.9rem; margin-top:-0.5rem; margin-bottom:0.75rem;",
                HTML("<em>Note:</em> “Lower” refers to the 10th percentile and “Upper” refers to the 90th percentile.")
              ),
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
        
        # Mean-based accuracy
        p(
          style = "color:#DDDDDD;",
          strong("Root Mean Squared Error (RMSE):"),
          "Measures the typical size of prediction errors, giving extra weight to large mistakes. ",
          "A higher RMSE suggests occasional big misses are present."
        ),
        p(
          style = "color:#DDDDDD;",
          strong("Mean Absolute Error (MAE):"),
          "Measures the average size of all prediction errors, treating every miss equally. ",
          "A lower MAE means the model is consistently close, even if not perfect."
        ),
        
        # Variance explained
        p(
          style = "color:#DDDDDD;",
          strong("R-squared (R²):"),
          "Proportion of variance explained. An R² of 0.5 means we capture about half of the ",
          "game-to-game variability. Shows how much better the model is at predicting compared ",
          "to simply guessing the average every time."
        ),
        
        # Pinball Loss
        p(
          style = "color:#DDDDDD;",
          strong("Pinball Loss (τ = 0.1):"),
          "Measures how well the model predicts the 10th percentile. ",
          "A lower value means the model is effectively avoiding over-predictions, ",
          "making it useful for conservative forecasting."
        ),
        p(
          style = "color:#DDDDDD;",
          strong("Pinball Loss (τ = 0.5):"),
          "Measures how well the model predicts the 50th percentile (median). ",
          "A lower value indicates the median predictions are closely aligned with actual outcomes."
        ),
        p(
          style = "color:#DDDDDD;",
          strong("Pinball Loss (τ = 0.9):"),
          "Measures how well the model predicts the 90th percentile. ",
          "A lower value here means the model is good at avoiding under-predictions, ",
          "useful for capturing upper-bound expectations."
        ),
        
        # Prediction interval coverage
        p(
          style = "color:#DDDDDD;",
          strong("80% Prediction Interval Coverage:"),
          "Share of games where the actual value fell within the model’s 80% prediction interval (10th to 90th percentile). ",
          "Values near 80% indicate well-calibrated intervals; much higher suggests intervals are too wide, ",
          "much lower suggests they’re too narrow."
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
  # --- Reactive: Implied probability (UNCHANGED) ---
  output$implied_prob <- renderText({
    req(input$american_odds)
    odds <- as.numeric(input$american_odds)
    if (is.na(odds)) return("Enter valid odds.")
    prob <- if (odds < 0) abs(odds)/(abs(odds)+100)*100 else 100/(odds+100)*100
    paste0("Implied Probability: ", round(prob,1), "%")
  })
  
  # --- Reactive: Predictions via reactivePoll() ---
  # Poll every 10 minutes for either a NEW file (date rollover) or a modified mtime
  preds <- reactivePoll(
    600000,   # 10 minutes in ms
    session,
    checkFunc = function() {
      path <- latest_predictions_path()
      if (is.na(path)) return("")        # nothing to load yet
      fi <- file.info(path)
      # Return "path mtime size" so changes in file or contents trigger updates.
      paste(path, as.numeric(fi$mtime), fi$size)
    },
    valueFunc = function() {
      path <- latest_predictions_path()
      validate(need(!is.na(path), sprintf("No predictions parquet found yet in '%s'.", PREDICTIONS_DIR)))
      # try-catch in case file is mid-write when polled
      tryCatch({
        arrow::read_parquet(path) |>
          dplyr::mutate(
            dplyr::across(where(is.numeric), ~ round(.x, 1)),
            home_away = stringr::str_to_sentence(home_away)
          ) |>
          dplyr::arrange(game_date, team_abbreviation, athlete_display_name)
      }, error = function(e) {
        validate(need(FALSE, paste("Failed to read predictions:", conditionMessage(e))))
      })
    }
  )
  
  # --- Reactive: Metrics via reactiveFileReader() ---
  # Auto-reloads when datasets/Evaluation_Metrics.parquet is updated
  metrics <- reactiveFileReader(
    86400000, # daily
    session,
    filePath = "datasets/Evaluation_Metrics.parquet",
    readFunc = function(fp) {
      arrow::read_parquet(fp)
    }
  )
  
  # --- Predictions Table ---
  output$predictions_table <- renderReactable({
    req(input$selected_columns)
    
    df <- preds() %>%
      dplyr::mutate(
        Player = paste0(
          "<div style='text-align:center;'>",
          "<img src='", headshot_url, "' height='60' style='border-radius:50%;'><br>",
          athlete_display_name, "</div>"
        ),
        Team     = team_abbreviation,
        Opponent = opponent_team_abbreviation,
        Date     = game_date,
        HomeAway = home_away,
        
        # --- 3-Point FG ---
        `3-Point FG (Mean)`        = three_point_field_goals_made_mean,
        `3-Point FG (Median)`      = three_point_field_goals_made_median,
        `3-Point FG (Lower)`       = three_point_field_goals_made_lower,
        `3-Point FG (Upper)`       = three_point_field_goals_made_upper,
        `3-Point FG (Pred Std)`    = three_point_field_goals_made_std_pred,
        `3-Point FG (Epi Std)`     = three_point_field_goals_made_std_epistemic,
        `3-Point FG (Ale Std)`     = three_point_field_goals_made_std_aleatoric,
        `3-Point FG (Std80 Lower)` = three_point_field_goals_made_std80_lower,
        `3-Point FG (Std80 Upper)` = three_point_field_goals_made_std80_upper,
        `3-Point FG (PI80 Width)`  = three_point_field_goals_made_pi80_width,
        
        # --- Rebounds ---
        `Rebounds (Mean)`        = rebounds_mean,
        `Rebounds (Median)`      = rebounds_median,
        `Rebounds (Lower)`       = rebounds_lower,
        `Rebounds (Upper)`       = rebounds_upper,
        `Rebounds (Pred Std)`    = rebounds_std_pred,
        `Rebounds (Epi Std)`     = rebounds_std_epistemic,
        `Rebounds (Ale Std)`     = rebounds_std_aleatoric,
        `Rebounds (Std80 Lower)` = rebounds_std80_lower,
        `Rebounds (Std80 Upper)` = rebounds_std80_upper,
        `Rebounds (PI80 Width)`  = rebounds_pi80_width,
        
        # --- Assists ---
        `Assists (Mean)`        = assists_mean,
        `Assists (Median)`      = assists_median,
        `Assists (Lower)`       = assists_lower,
        `Assists (Upper)`       = assists_upper,
        `Assists (Pred Std)`    = assists_std_pred,
        `Assists (Epi Std)`     = assists_std_epistemic,
        `Assists (Ale Std)`     = assists_std_aleatoric,
        `Assists (Std80 Lower)` = assists_std80_lower,
        `Assists (Std80 Upper)` = assists_std80_upper,
        `Assists (PI80 Width)`  = assists_pi80_width,
        
        # --- Steals ---
        `Steals (Mean)`        = steals_mean,
        `Steals (Median)`      = steals_median,
        `Steals (Lower)`       = steals_lower,
        `Steals (Upper)`       = steals_upper,
        `Steals (Pred Std)`    = steals_std_pred,
        `Steals (Epi Std)`     = steals_std_epistemic,
        `Steals (Ale Std)`     = steals_std_aleatoric,
        `Steals (Std80 Lower)` = steals_std80_lower,
        `Steals (Std80 Upper)` = steals_std80_upper,
        `Steals (PI80 Width)`  = steals_pi80_width,
        
        # --- Blocks ---
        `Blocks (Mean)`        = blocks_mean,
        `Blocks (Median)`      = blocks_median,
        `Blocks (Lower)`       = blocks_lower,
        `Blocks (Upper)`       = blocks_upper,
        `Blocks (Pred Std)`    = blocks_std_pred,
        `Blocks (Epi Std)`     = blocks_std_epistemic,
        `Blocks (Ale Std)`     = blocks_std_aleatoric,
        `Blocks (Std80 Lower)` = blocks_std80_lower,
        `Blocks (Std80 Upper)` = blocks_std80_upper,
        `Blocks (PI80 Width)`  = blocks_pi80_width,
        
        # --- Points ---
        `Points (Mean)`        = points_mean,
        `Points (Median)`      = points_median,
        `Points (Lower)`       = points_lower,
        `Points (Upper)`       = points_upper,
        `Points (Pred Std)`    = points_std_pred,
        `Points (Epi Std)`     = points_std_epistemic,
        `Points (Ale Std)`     = points_std_aleatoric,
        `Points (Std80 Lower)` = points_std80_lower,
        `Points (Std80 Upper)` = points_std80_upper,
        `Points (PI80 Width)`  = points_pi80_width
      )
    
    df_to_display <- df[, input$selected_columns, drop = FALSE]
    
    col_defs <- list()
    if ("Player" %in% names(df_to_display))   col_defs$Player   <- colDef(html = TRUE, align = "center", minWidth = 120)
    if ("Team" %in% names(df_to_display))     col_defs$Team     <- colDef(align = "center", minWidth = 60)
    if ("Opponent" %in% names(df_to_display)) col_defs$Opponent <- colDef(align = "center", minWidth = 100)
    if ("Date" %in% names(df_to_display))     col_defs$Date     <- colDef(align = "center", minWidth = 90)
    if ("HomeAway" %in% names(df_to_display)) col_defs$HomeAway <- colDef(name = "Home/Away", align = "center", minWidth = 110)
    
    # Right-align any numeric stat columns the user selected
    right_align <- intersect(names(df_to_display), setdiff(names(df_to_display), c("Player","Team","Opponent","Date","HomeAway")))
    for (nm in right_align) col_defs[[nm]] <- colDef(align = "right")
    
    reactable(
      df_to_display,
      columns = col_defs,
      pagination = TRUE,
      searchable = TRUE,
      highlight  = TRUE,
      compact    = TRUE,
      defaultColDef = colDef(minWidth = 110),
      theme = reactableTheme(
        style    = list(background = "#121212"),
        rowStyle = list(borderBottom = "1px solid #007AC1")
      )
    )
  })
  
  output$metrics_table <- renderReactable({
    m <- metrics() %>%
      dplyr::filter(Suffix == "cal") %>%
      dplyr::select(
        Target, RMSE_Mean, MAE_Mean, R2, RMSE_Median, MAE_Median,
        Pinball_10, Pinball_50, Pinball_90,
        PI80_Coverage, PI80_Width,
        Below_Q10_Rate, Above_Q50_Rate, Above_Q90_Rate,
        STD80_Coverage,
        STD_Predictive_Mean, STD_Epistemic_Mean, STD_Aleatoric_Mean,
        Bias_MeanError, Uncert_Error_Corr
      )
    
    reactable(
      m,
      pagination = FALSE,
      highlight = TRUE,
      compact = TRUE,
      columns = list(
        Target = colDef(name = "Target", align = "left"),
        
        RMSE_Mean   = colDef(name = "RMSE (Mean)",   format = colFormat(digits = 1), align = "right"),
        MAE_Mean    = colDef(name = "MAE (Mean)",    format = colFormat(digits = 1), align = "right"),
        R2          = colDef(name = "R²",            format = colFormat(digits = 2), align = "right"),
        RMSE_Median = colDef(name = "RMSE (Median)", format = colFormat(digits = 1), align = "right"),
        MAE_Median  = colDef(name = "MAE (Median)",  format = colFormat(digits = 1), align = "right"),
        
        Pinball_10  = colDef(name = "Pinball Loss (q=0.10)", format = colFormat(digits = 2), align = "right"),
        Pinball_50  = colDef(name = "Pinball Loss (q=0.50)", format = colFormat(digits = 2), align = "right"),
        Pinball_90  = colDef(name = "Pinball Loss (q=0.90)", format = colFormat(digits = 2), align = "right"),
        
        PI80_Coverage   = colDef(name = "80% PI Coverage (q10–q90)", format = colFormat(percent = TRUE, digits = 0), align = "right"),
        PI80_Width      = colDef(name = "PI80 Width",                   format = colFormat(digits = 2), align = "right"),
        
        Below_Q10_Rate  = colDef(name = "Below q10 Rate", format = colFormat(percent = TRUE, digits = 0), align = "right"),
        Above_Q50_Rate  = colDef(name = "Above q50 Rate", format = colFormat(percent = TRUE, digits = 0), align = "right"),
        Above_Q90_Rate  = colDef(name = "Above q90 Rate", format = colFormat(percent = TRUE, digits = 0), align = "right"),
        
        STD80_Coverage      = colDef(name = "STD 80% Coverage (± z*std)", format = colFormat(percent = TRUE, digits = 0), align = "right"),
        STD_Predictive_Mean = colDef(name = "Mean Std (Predictive)",        format = colFormat(digits = 2), align = "right"),
        STD_Epistemic_Mean  = colDef(name = "Mean Std (Epistemic)",         format = colFormat(digits = 2), align = "right"),
        STD_Aleatoric_Mean  = colDef(name = "Mean Std (Aleatoric)",         format = colFormat(digits = 2), align = "right"),
        
        Bias_MeanError     = colDef(name = "Bias (Mean Error)",      format = colFormat(digits = 2), align = "right"),
        Uncert_Error_Corr  = colDef(name = "Uncertainty–Error Corr", format = colFormat(digits = 2), align = "right")
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