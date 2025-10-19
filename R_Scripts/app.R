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
metrics_env <- Sys.getenv("METRICS_PATH")
if (nzchar(metrics_env)) {
  METRICS_PATH <- normalizePath(metrics_env, winslash = "/", mustWork = TRUE)
} else {
  wd <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  if (basename(wd) == "R_Scripts") {
    METRICS_PATH <- normalizePath(file.path(wd, "..", "datasets", "Evaluation_Metrics.parquet"),
                                  winslash = "/", mustWork = TRUE)
  } else {
    METRICS_PATH <- normalizePath(file.path(wd, "datasets", "Evaluation_Metrics.parquet"),
                                  winslash = "/", mustWork = TRUE)
  }
}

pred_dir_env <- Sys.getenv("PREDICTIONS_DIR")
if (nzchar(pred_dir_env)) {
  PREDICTIONS_DIR <- normalizePath(pred_dir_env, winslash = "/", mustWork = TRUE)
} else {
  wd <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  # If running the app from R_Scripts/, go up one level
  if (basename(wd) == "R_Scripts") {
    PREDICTIONS_DIR <- normalizePath(file.path(wd, "..", "predictions"), winslash = "/", mustWork = TRUE)
  } else {
    PREDICTIONS_DIR <- normalizePath(file.path(wd, "predictions"), winslash = "/", mustWork = TRUE)
  }
}

today_str  <- format(Sys.Date(), "%Y-%m-%d")
latest_prediction_path <- file.path(PREDICTIONS_DIR, paste0("nba_predictions_", today_str, ".parquet"))

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
        h2("Model Metrics", style = "color:#FFFFFF;"),
        reactableOutput("metrics_table"),
        div(
          style = "text-align:center; margin-top:2rem; margin-bottom:2rem;",
          tags$img(
            src = "nba_dark.png",
            style = "max-width:420px; height: auto; width:100%;"
          )
        )
      )
    ),
    # --- GUIDE TAB (glossary / how to interpret) ---
    tabPanel(
      "Guide",
      div(
        class = "container-fluid",
        style = "padding:2rem; background:#121212; color:#FFFFFF;",
        
        h2("How to Interpret Predictions & Metrics"),
        p(class = "text-muted",
          style = "color:#BBBBBB;",
          "This page explains all prediction columns and evaluation metrics used in the app."
        ),
        tags$hr(style = "border-top: 1px solid #007AC1;"),
        
        # =========================
        # Predicted Stats
        # =========================
        h3("Predicted Stats"),
        p("Each stat is predicted for the player's next game."),
        tags$ul(
          tags$li(tags$b("3-Point FG"), ": Predicted three-pointers made (not attempted)."),
          tags$li(tags$b("Rebounds"), ": Total rebounds (offensive + defensive)."),
          tags$li(tags$b("Assists"), ": Recorded assists."),
          tags$li(tags$b("Steals"), ": Recorded steals."),
          tags$li(tags$b("Blocks"), ": Recorded blocks."),
          tags$li(tags$b("Points"), ": Total points scored.")
        ),
        
        h4("Uncertainty Columns (per stat)"),
        tags$ul(
          tags$li(tags$b("Mean"), ": Expected value of the stat."),
          tags$li(tags$b("Median"), ": 50th percentile (middle of the distribution). ",
                  "If Mean and Median differ a lot, the distribution may be skewed."),
          
          tags$li(tags$b("Lower (q10)"), ": 10th percentile. ",
                  tags$em("Closer to Mean → "), "tighter, less downside risk. ",
                  tags$em("Much lower than Mean → "), "more downside risk / uncertainty."),
          
          tags$li(tags$b("Upper (q90)"), ": 90th percentile. ",
                  tags$em("Much higher than Mean → "), "larger upside / uncertainty. ",
                  tags$em("Closer to Mean → "), "tighter, less upside spread."),
          
          tags$li(tags$b("Pred Std"), ": Total predictive std (", tags$em("epistemic + aleatoric"), "). ",
                  tags$em("High: "), "outcomes may vary widely (be cautious). ",
                  tags$em("Low: "), "outcomes expected to be consistent (more confidence)."),
          
          tags$li(tags$b("Epi Std"), ": Epistemic uncertainty (what the model doesn’t know yet). ",
                  tags$em("High: "), "limited/new context (role change, injuries, small samples). ",
                  tags$em("Low: "), "model is well-trained for this context."),
          
          tags$li(tags$b("Ale Std"), ": Aleatoric uncertainty (inherent randomness). ",
                  tags$em("High: "), "stat/matchup is volatile (e.g., stocks like blocks/steals). ",
                  tags$em("Low: "), "intrinsically steadier environment."),
          
          tags$li(tags$b("Std80 Lower / Std80 Upper"),
                  ": 80% interval via ±1.2816 × Pred Std. ",
                  tags$em("Wider → "), "more uncertainty; ",
                  tags$em("Narrower → "), "more confidence."),
          
          tags$li(tags$b("PI80 Width"),
                  ": Width of the 10–90% interval (Upper − Lower). ",
                  tags$em("High: "), "bigger range of plausible outcomes; ",
                  tags$em("Low: "), "tighter expectation.")
        ),
        
        tags$hr(style = "border-top: 1px solid #007AC1; margin: 2rem 0;"),
        
        # =========================
        # Metrics
        # =========================
        h3("Metrics"),
        p("Computed on historical data to judge calibration and accuracy."),
        tags$ul(
          tags$li(tags$b("RMSE (Mean)"),
                  ": Root Mean Squared Error using mean predictions. ",
                  tags$em("Lower is better; punishes big misses.")),
          
          tags$li(tags$b("MAE (Mean)"),
                  ": Mean Absolute Error using mean predictions. ",
                  tags$em("Lower is better; typical miss size.")),
          
          tags$li(tags$b("R²"),
                  ": Variance explained vs a constant baseline. ",
                  tags$em("Higher is better.")),
          
          tags$li(tags$b("RMSE / MAE (Median)"),
                  ": Same error metrics but for median predictions. ",
                  tags$em("Lower is better.")),
          
          tags$li(tags$b("Pinball Loss (q=0.10 / 0.50 / 0.90)"),
                  ": Quantile loss at 10th / 50th / 90th percentiles. ",
                  tags$em("Lower is better; quantile accuracy.")),
          
          tags$li(tags$b("80% PI Coverage (q10–q90)"),
                  ": Share of actuals inside the 10–90% interval. ",
                  tags$em("Desirable ≈ 80%; "),
                  "much higher → intervals too wide, much lower → too narrow."),
          
          tags$li(tags$b("PI80 Width"),
                  ": Average width of the 10–90% interval. ",
                  tags$em("Lower is tighter"), " — but balance with Coverage (avoid under-covering)."),
          
          tags$li(tags$b("Below q10 / Above q50 / Above q90 Rates"),
                  ": Empirical tail frequencies. ",
                  tags$em("Targets ≈ 10% / 50% / 10%; "),
                  "persistent deviation → miscalibration."),
          
          tags$li(tags$b("STD 80% Coverage (± z·std)"),
                  ": Coverage when intervals use ±1.2816 × Pred Std. ",
                  tags$em("≈ 80% indicates std is well-scaled.")),
          
          tags$li(tags$b("Mean Std (Predictive / Epistemic / Aleatoric)"),
                  ": Averages of std components across games. ",
                  tags$em("Lower → "), "more confidence; ",
                  tags$em("Higher → "), "more uncertainty (check calibration)."),
          
          tags$li(tags$b("Bias (Mean Error)"),
                  ": Average (prediction − actual). ",
                  tags$em("Closer to 0 is better; sign shows over/under-prediction.")),
          
          tags$li(tags$b("Uncertainty–Error Corr"),
                  ": Correlation between predicted uncertainty and absolute error. ",
                  tags$em("Positive is desirable: "),
                  "the model is more uncertain when it tends to miss more.")
        ),
        
        tags$hr(style = "border-top: 1px solid #007AC1; margin: 2rem 0;"),
        
        h4("Quick Tips"),
        tags$ul(
          tags$li("For conservative plays, focus on ", tags$b("Lower (q10)"), " and a small ", tags$b("PI80 Width"), "."),
          tags$li("If ", tags$b("Pred Std"), " is high, check whether it’s driven by ",
                  tags$b("Epi Std"), " (limited/shifted data) or ",
                  tags$b("Ale Std"), " (inherent volatility)."),
          tags$li("Good calibration: Coverage near 80% and tail rates near 10% / 50% / 10%.")
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
    600000, session,
    checkFunc = function() {
      if (!file.exists(latest_prediction_path)) "" else {
        fi <- file.info(latest_prediction_path)
        paste(as.numeric(fi$mtime), fi$size)
      }
    },
    valueFunc = function() {
      validate(need(file.exists(latest_prediction_path),
                    paste("Missing file:", latest_prediction_path)))
      arrow::read_parquet(latest_prediction_path) |>
        dplyr::mutate(
          dplyr::across(where(is.numeric), ~ round(.x, 1)),
          home_away = stringr::str_to_sentence(home_away)
        ) |>
        dplyr::arrange(game_date, team_abbreviation, athlete_display_name)
    }
  )

  
  # --- Reactive: Metrics via reactiveFileReader() ---
  # Auto-reloads when datasets/Evaluation_Metrics.parquet is updated
  metrics <- reactiveFileReader(
    86400000, # daily
    session,
    filePath = METRICS_PATH,
    readFunc = function(fp) arrow::read_parquet(fp)
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
    
    # 1) Expand group labels to actual columns by prefix match
    meta <- c("Player","Team","Opponent","Date","HomeAway")
    selected_meta   <- intersect(input$selected_columns, meta)
    selected_stats  <- setdiff(input$selected_columns, meta)
    
    # Find all columns that start with "<label> (" (e.g., "3-Point FG (Mean)")
    stat_cols <- unlist(lapply(
      selected_stats,
      function(lbl) grep(paste0(lbl, " ("), names(df), value = TRUE, fixed = TRUE)
    ), use.names = FALSE)
    
    display_cols <- c(selected_meta, stat_cols)
    req(length(display_cols) > 0)  # don’t render if nothing chosen
    
    df_to_display <- df[, display_cols, drop = FALSE]
    
    # 2) Column defs — keep yours, this just right-aligns numeric stats
    col_defs <- list()
    if ("Player"   %in% names(df_to_display)) col_defs$Player   <- colDef(html = TRUE, align = "center", minWidth = 120)
    if ("Team"     %in% names(df_to_display)) col_defs$Team     <- colDef(align = "center", minWidth = 60)
    if ("Opponent" %in% names(df_to_display)) col_defs$Opponent <- colDef(align = "center", minWidth = 100)
    if ("Date"     %in% names(df_to_display)) col_defs$Date     <- colDef(align = "center", minWidth = 90)
    if ("HomeAway" %in% names(df_to_display)) col_defs$HomeAway <- colDef(name = "Home/Away", align = "center", minWidth = 110)
    
    numeric_cols <- names(df_to_display)[vapply(df_to_display, is.numeric, logical(1))]
    for (nm in setdiff(numeric_cols, c("Team","Date"))) col_defs[[nm]] <- colDef(align = "right")
    
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
      dplyr::mutate(
        Target = dplyr::recode(
          Target,
          "three_point_field_goals_made" = "3-Point FG",
          "rebounds" = "Rebounds",
          "assists"  = "Assists",
          "steals"   = "Steals",
          "blocks"   = "Blocks",
          "points"   = "Points"
          )
        ) %>%
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
        Uncert_Error_Corr  = colDef(name = "Uncertainty-Error Corr", format = colFormat(digits = 2), align = "right", minWidth = 110)
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