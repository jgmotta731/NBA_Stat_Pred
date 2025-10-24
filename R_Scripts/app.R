# app.R

library(shiny)
library(bslib)
library(reactable)
library(dplyr)
library(arrow)
library(htmltools)
library(stringr)
library(httr)    # for HEAD/GET + write_disk

# ==============================
# Remote data endpoints (fixed)
# ==============================
PREDICTIONS_URL <- "https://github.com/jgmotta731/NBA_Stat_Pred/raw/refs/heads/main/predictions.parquet"
METRICS_URL     <- "https://github.com/jgmotta731/NBA_Stat_Pred/raw/refs/heads/main/Evaluation_Metrics.parquet"

# Small helper: robust GET → temp file → read_parquet
.read_remote_parquet <- function(url, timeout_sec = 30L) {
  tmp <- tempfile(fileext = ".parquet")
  r <- GET(url, write_disk(tmp, overwrite = TRUE), timeout(timeout_sec))
  if (http_error(r)) stop(sprintf("HTTP %s while fetching %s", status_code(r), url))
  arrow::read_parquet(tmp)
}

# ==============================
# Theme
# ==============================
my_theme <- bs_theme(
  version       = 5,
  bg            = "#121212",
  fg            = "#FFFFFF",
  primary       = "#007AC1",
  base_font     = font_google("Inter"),
  heading_font  = font_google("Roboto"),
  "navbar-bg"               = "#121212",
  "navbar-dark-color"       = "#FFFFFF",
  "navbar-dark-hover-color" = "#007AC1"
)

PRIMARY_BLUE <- "#007AC1"

# ==============================
# UI
# ==============================
ui <- tagList(
  tags$head(
    tags$style(HTML('
      .navbar { min-height: 64px; padding-top: 3px; padding-bottom: 3px; }
      .navbar-brand { display: flex; align-items: center; font-size: 1.4rem; }
      .navbar-brand img { margin-right: 10px; height: 42px; }
      .navbar-nav > li > a { font-size: 1.0rem !important; font-weight: 500; }
      .card { display: flex; flex-direction: column; height: 100%; }
      .card-body { flex: 1; }
      .reactable .th, .reactable .td { color: #FFFFFF; }
      .reactable input[type="text"], .reactable select { color: black !important; }

      /* Cells compact; headers can wrap so names stay readable */
      .reactable .rt-td { white-space: nowrap; }
      .reactable .rt-th { white-space: normal; line-height: 1.2; }
      .reactable .rt-thead .rt-th { padding-top: 8px; padding-bottom: 8px; }
      .reactable .rt-th, .reactable .rt-th .rt-resizable-header-content { overflow: visible; }

      /* No hover highlight */
      .reactable .rt-tr:hover .rt-td { background: transparent !important; }

      /* ===== Table fills the card; scroll only when needed ===== */
      #predictions_wrap { position: relative; overflow-x: auto; width: 100%; }
      #predictions_wrap .reactable,
      #predictions_wrap .html-widget,
      #predictions_wrap .rt-table {
        min-width: 100%;        /* fill when few columns */
        width: max-content;     /* grow to content when many columns */
      }

      /* ===== Sticky toolbar (search + pagination) pinned to top-right of scrollport ===== */
      #predictions_wrap .sticky-toolbar {
        position: sticky;
        top: 0;
        left: 0;
        right: 0;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 10px;
        padding: 6px 6px 0 6px;
        z-index: 5;
        background: linear-gradient(90deg, rgba(18,18,18,0) 0%, rgba(18,18,18,0.9) 40%, rgba(18,18,18,1) 100%);
      }
      #predictions_wrap .sticky-toolbar .form-control,
      #predictions_wrap .sticky-toolbar .form-select {
        height: 32px;
        line-height: 32px;
        padding-top: 2px;
        padding-bottom: 2px;
        border: 1px solid #007AC1;
        box-shadow: none;
      }
      #predictions_wrap .sticky-toolbar .pager {
        display: flex;
        align-items: center;
        gap: 6px;
      }
      #predictions_wrap .sticky-toolbar .pager .btn {
        padding: 2px 8px;
        line-height: 1.2;
        border-color: #007AC1;
        color: #FFFFFF;
      }
      #predictions_wrap .sticky-toolbar .pager .btn:hover {
        background: rgba(0,122,193,0.2);
      }
      #predictions_wrap .sticky-toolbar .page-info {
        color: #BBBBBB;
        font-size: 0.9rem;
        padding: 0 2px;
      }
    '))
  ),
  
  navbarPage(
    title = div(tags$img(src = "nba_light.png", alt = "Logo"), "NBA Player Predictor"),
    theme = my_theme,
    
    # --- HOME ---
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
          div(class = "col-md-4", style = "padding:1rem;",
              div(class = "card h-100", style = "background:#1e1e1e; border:1px solid #007AC1;",
                  div(class="card-body", h3("Advanced Analytics", class="card-title"),
                      p("Machine-learning models trained on historical NBA data.")
                  ))),
          div(class = "col-md-4", style = "padding:1rem;",
              div(class = "card h-100", style = "background:#FFFFFF; border:1px solid #007AC1; color:#000000;",
                  div(class="card-body", h3("Weekly Updates", class="card-title text-primary"),
                      p("Fresh predictions before every game, with latest data and scraped over-under odds from FanDuel via Odds API.")
                  ))),
          div(class = "col-md-4", style = "padding:1rem;",
              div(class = "card h-100", style = "background:#1e1e1e; border:1px solid #007AC1;",
                  div(class="card-body", h3("Key Stats", class="card-title"),
                      p("Predictions for 3-PT FG, Rebounds, Assists, Steals, Blocks, Points.")
                  )))
        ),
        div(style = "text-align:center; margin-top:2rem;",
            tags$img(src="nba_dark.png",
                     style="width: 80%; max-width: 515px; height: auto; margin: 0 auto; display: block;"))
      )
    ),
    
    # --- PREDICTIONS ---
    tabPanel(
      "Predictions",
      div(
        class = "container-fluid",
        style = "padding:0rem; background:#121212;",
        fluidRow(
          # Sidebar
          column(
            width = 2,
            div(
              style = "background:#1e1e1e; border:1px solid #007AC1; padding:1rem;",
              h4("Implied Probability Calculator", style = "color:#FFFFFF;"),
              textInput("american_odds", "American Odds:", "", width = "100%"),
              verbatimTextOutput("implied_prob", placeholder = TRUE),
              tags$hr(style = "border-top: 1px solid #007AC1;"),
              div(
                style = "margin-top:1rem; color:white;",
                h4("Select Columns", style = "color:white;"),
                checkboxGroupInput(
                  inputId = "selected_columns",
                  label = NULL,
                  choices = c("Player","Team","Opponent","Date","HomeAway",
                              "Points","Rebounds","Assists","Steals","Blocks","3PM"),
                  selected = c("Player","Team","Opponent","Date","HomeAway",
                               "Points","Rebounds","Assists","Steals","Blocks","3PM")
                )
              )
            )
          ),
          # Main table
          column(
            width = 10,
            div(
              class = "card",
              style = "background:#1e1e1e; border:1px solid #007AC1; padding:1rem;",
              h2("Weekly Player Predictions", style = "color:white;"),
              div(
                id = "predictions_wrap",
                # Sticky toolbar: search + page controls
                div(class = "sticky-toolbar",
                    textInput("pred_search", NULL, placeholder = "Search players, teams, stats…", width = "260px"),
                    div(class="pager",
                        actionButton("page_prev", "‹", class = "btn btn-sm btn-outline-primary"),
                        numericInput("page_num", NULL, value = 1, min = 1, step = 1, width = "80px"),
                        span(class="page-info", "of"),
                        textOutput("page_max", container = span),
                        actionButton("page_next", "›", class = "btn btn-sm btn-outline-primary"),
                        selectInput("page_size", NULL,
                                    choices = c(5,10,15,20,25,100), selected = 10, width = "90px")
                    )
                ),
                reactableOutput("predictions_table")
              )
            )
          )
        )
      )
    ),
    
    # --- METRICS ---
    tabPanel(
      "Metrics",
      div(
        class = "container-fluid",
        style = "padding:0rem; background:#121212;",
        h2("Model Metrics", style = "color:#FFFFFF;"),
        reactableOutput("metrics_table"),
        div(style = "text-align:center; margin-top:2rem; margin-bottom:2rem;",
            tags$img(src = "nba_dark.png", style = "max-width:420px; height: auto; width:100%;"))
      )
    ),
    
    # --- GUIDE ---
    tabPanel(
      "Guide",
      div(
        class = "container-fluid",
        style = "padding:2rem; background:#121212; color:#FFFFFF;",
        h2("How to Interpret Predictions & Metrics"),
        p(class = "text-muted", style = "color:#BBBBBB;",
          "This page explains all prediction columns and evaluation metrics used in the app."
        ),
        tags$hr(style = "border-top: 1px solid #007AC1;"),
        
        h3("Predicted Stats"),
        p("Each stat is predicted for the player's next game."),
        tags$ul(
          tags$li(tags$b("3PM")),
          tags$li(tags$b("Rebounds")),
          tags$li(tags$b("Assists")),
          tags$li(tags$b("Steals")),
          tags$li(tags$b("Blocks")),
          tags$li(tags$b("Points"))
        ),
        
        h4("Uncertainty Columns (per stat)"),
        tags$ul(
          tags$li(tags$b("Mean"), ": Expected value of the stat — the model’s best overall estimate."),
          tags$li(tags$b("Median"), ": 50th percentile — midpoint prediction where half of outcomes are above and half below."),
          tags$li(tags$b("Lower (q10)"), ": 10th percentile — closer to mean ⇒ tighter, much lower ⇒ more downside risk."),
          tags$li(tags$b("Upper (q90)"), ": 90th percentile — much higher than mean ⇒ more upside spread."),
          tags$li(tags$b("PI80 Width"), ": 80% predictive interval width — distance between 10th and 90th percentiles. High ⇒ more uncertainty; Low ⇒ tighter expectation."),
          tags$li(tags$b("Pred Std"), ": Predicted standard deviation — spread and total uncertainty. High ⇒ wide outcomes; Low ⇒ tight range."),
          tags$li(tags$b("Epi Std"), ": Epistemic uncertainty — what the model doesn’t know (context/data limits). High ⇒ new context; Low ⇒ unfamiliar situation."),
          tags$li(tags$b("Ale Std"), ": Aleatoric uncertainty — inherent randomness (shooting variance, pace, foul trouble). High ⇒ volatile; Low ⇒ consistent."),
          tags$li(tags$b("Std80 Lower / Std80 Upper"), ": Mean ± 1.28 × Pred Std — 80% interval based on the predicted standard deviation. Wider ⇒ more uncertainty; narrower ⇒ tighter outcomes"),
        ),
        
        h5("Examples of low vs. high values (by stat)"),
        p("Use these as ballpark cutoffs; compare players in similar roles."),
        tags$ul(
          tags$li(
            tags$b("Points"),
            tags$ul(
              tags$li("Pred Std: low ≤ 3 pts, high ≥ 6 pts"),
              tags$li("Epi Std: low ≤ 2 pts, high ≥ 4 pts"),
              tags$li("Ale Std: low ≤ 2 pts, high ≥ 4 pts"),
              tags$li("Std80 width (Std80 Upper − Std80 Lower): narrow ≤ 6 pts, wide ≥ 12 pts")
            )
          ),
          tags$li(
            tags$b("Rebounds"),
            tags$ul(
              tags$li("Pred Std: low ≤ 2, high ≥ 4"),
              tags$li("Epi Std: low ≤ 1, high ≥ 2"),
              tags$li("Ale Std: low ≤ 1.5, high ≥ 3"),
              tags$li("Std80 width: narrow ≤ 3, wide ≥ 6")
            )
          ),
          tags$li(
            tags$b("Assists"),
            tags$ul(
              tags$li("Pred Std: low ≤ 1.5, high ≥ 3"),
              tags$li("Epi Std: low ≤ 0.8, high ≥ 1.5"),
              tags$li("Ale Std: low ≤ 1.0, high ≥ 2.0"),
              tags$li("Std80 width: narrow ≤ 2.5, wide ≥ 5")
            )
          ),
          tags$li(
            tags$b("3PM"),
            tags$ul(
              tags$li("Pred Std: low ≤ 0.5, high ≥ 1.0"),
              tags$li("Epi Std: low ≤ 0.3, high ≥ 0.6"),
              tags$li("Ale Std: low ≤ 0.4, high ≥ 0.8"),
              tags$li("Std80 width: narrow ≤ 1.0, wide ≥ 2.0")
            )
          ),
          tags$li(
            tags$b("Steals"),
            tags$ul(
              tags$li("Pred Std: low ≤ 0.4, high ≥ 0.8"),
              tags$li("Epi Std: low ≤ 0.2, high ≥ 0.4"),
              tags$li("Ale Std: low ≤ 0.3, high ≥ 0.6"),
              tags$li("Std80 width: narrow ≤ 0.8, wide ≥ 1.6")
            )
          ),
          tags$li(
            tags$b("Blocks"),
            tags$ul(
              tags$li("Pred Std: low ≤ 0.4, high ≥ 0.8"),
              tags$li("Epi Std: low ≤ 0.2, high ≥ 0.4"),
              tags$li("Ale Std: low ≤ 0.3, high ≥ 0.6"),
              tags$li("Std80 width: narrow ≤ 0.8, wide ≥ 1.6")
            )
          )
        ),
        
        tags$hr(style = "border-top: 1px solid #007AC1; margin: 2rem 0;"),
        
        h3("Metrics"),
        p("Computed on historical data to judge accuracy and calibration."),
        tags$ul(
          tags$li(tags$b("RMSE (Mean)"), ": Lower is better; penalizes big misses."),
          tags$li(tags$b("MAE (Mean)"),  ": Lower is better; typical miss size."),
          tags$li(tags$b("R²"),          ": Higher is better."),
          tags$li(tags$b("RMSE/MAE (Median)"), ": Same metrics for median predictions."),
          tags$li(tags$b("Pinball Loss (q=0.10/0.50/0.90)"), ": Lower is better; quantile errors."),
          tags$li(tags$b("80% PI Coverage (q10–q90)"), ": ≈80% is on target (higher ⇒ too wide; lower ⇒ too narrow)."),
          tags$li(tags$b("PI80 Width"), ": Balance against coverage—narrow is good if coverage stays near 80%."),
          tags$li(tags$b("Below q10 / Above q50 / Above q90"), ": Aiming ≈10% / 50% / 10%; large deviations indicate miscalibration."),
          tags$li(tags$b("STD 80% Coverage (± z·std)"), ": ≈80% suggests the spread is well-scaled."),
          tags$li(tags$b("Mean Std (Predictive/Epistemic/Aleatoric)"), ": Lower generally indicates tighter, more confident predictions (watch coverage)."),
          tags$li(tags$b("Bias (Mean Error)"), ": Closer to 0 is better (sign shows over/under)."),
          tags$li(tags$b("Uncertainty–Error Corr"), ": Positive is preferred—larger errors happen when uncertainty is higher.")
        ),
        
        tags$hr(style = "border-top: 1px solid #007AC1; margin: 2rem 0;"),
        
        h4("Quick Tips"),
        tags$ul(
          tags$li(HTML(paste0("For conservative plays, focus on ", tags$b("Lower (q10)"), " and small ", tags$b("PI80 Width"), "."))),
          tags$li(HTML(paste0("If ", tags$b("Pred Std"), " is high, check whether it's driven by ", tags$b("Epi Std"), " or ", tags$b("Ale Std"), "."))),
          tags$li("Good calibration is where about 80% of actual results fall within the 80% interval, and outcomes fall below q10, above q50, and above q90 roughly 10%/50%/10% of the time."),
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

# ==============================
# Server
# ==============================
server <- function(input, output, session) {
  
  # Implied probability calc
  output$implied_prob <- renderText({
    req(input$american_odds)
    odds <- suppressWarnings(as.numeric(input$american_odds))
    if (is.na(odds)) return("Enter valid odds.")
    prob <- if (odds < 0) abs(odds)/(abs(odds)+100)*100 else 100/(odds+100)*100
    paste0("Implied Probability: ", round(prob, 1), "%")
  })
  
  # --------------------------
  # Predictions via URL
  # --------------------------
  preds <- reactive({
    validate(need(nzchar(PREDICTIONS_URL), "Predictions URL not set"))
    df <- .read_remote_parquet(PREDICTIONS_URL)
    df |>
      dplyr::mutate(
        dplyr::across(where(is.numeric), ~ round(.x, 1)),
        home_away = stringr::str_to_sentence(home_away)
      ) |>
      dplyr::arrange(game_date, team_abbreviation, athlete_display_name)
  })
  
  # --------------------------
  # Metrics via URL
  # --------------------------
  metrics <- reactive({
    validate(need(nzchar(METRICS_URL), "Metrics URL not set"))
    .read_remote_parquet(METRICS_URL)
  })
  
  # --------------------------
  # Predictions table helpers
  # --------------------------
  `%||%` <- function(a, b) if (!is.null(a)) a else b
  strip_html <- function(x) gsub("<[^>]*>", "", x)
  slug <- function(x) tolower(gsub("[^a-z0-9]+", "-", x))
  
  make_table_df <- function(df_raw) {
    if (!"headshot_url" %in% names(df_raw)) df_raw$headshot_url <- ""
    required_id_cols <- c("athlete_display_name","team_abbreviation",
                          "opponent_team_abbreviation","game_date","home_away")
    missing_ids <- setdiff(required_id_cols, names(df_raw))
    validate(need(length(missing_ids) == 0,
                  paste("Predictions missing required columns:", paste(missing_ids, collapse=", "))))
    
    df_raw %>%
      mutate(
        Player = paste0(
          "<div style='text-align:center;'>",
          "<img src='", headshot_url, "' height='60' style='border-radius:50%;'><br>",
          athlete_display_name, "</div>"
        ),
        Team     = team_abbreviation,
        Opponent = opponent_team_abbreviation,
        Date     = game_date,
        HomeAway = home_away,
        
        # Points
        `Points (Mean)`        = points_mean,
        `Points (Median)`      = points_median,
        `Points (Lower)`       = points_lower,
        `Points (Upper)`       = points_upper,
        `Points (PI80 Width)`  = points_pi80_width,
        `Points (Pred Std)`    = points_std_pred,
        `Points (Epi Std)`     = points_std_epistemic,
        `Points (Ale Std)`     = points_std_aleatoric,
        `Points (Std80 Lower)` = points_std80_lower,
        `Points (Std80 Upper)` = points_std80_upper,
        
        # Rebounds
        `Rebounds (Mean)`        = rebounds_mean,
        `Rebounds (Median)`      = rebounds_median,
        `Rebounds (Lower)`       = rebounds_lower,
        `Rebounds (Upper)`       = rebounds_upper,
        `Rebounds (PI80 Width)`  = rebounds_pi80_width,
        `Rebounds (Pred Std)`    = rebounds_std_pred,
        `Rebounds (Epi Std)`     = rebounds_std_epistemic,
        `Rebounds (Ale Std)`     = rebounds_std_aleatoric,
        `Rebounds (Std80 Lower)` = rebounds_std80_lower,
        `Rebounds (Std80 Upper)` = rebounds_std80_upper,
        
        # Assists
        `Assists (Mean)`        = assists_mean,
        `Assists (Median)`      = assists_median,
        `Assists (Lower)`       = assists_lower,
        `Assists (Upper)`       = assists_upper,
        `Assists (PI80 Width)`  = assists_pi80_width,
        `Assists (Pred Std)`    = assists_std_pred,
        `Assists (Epi Std)`     = assists_std_epistemic,
        `Assists (Ale Std)`     = assists_std_aleatoric,
        `Assists (Std80 Lower)` = assists_std80_lower,
        `Assists (Std80 Upper)` = assists_std80_upper,
        
        # Steals
        `Steals (Mean)`        = steals_mean,
        `Steals (Median)`      = steals_median,
        `Steals (Lower)`       = steals_lower,
        `Steals (Upper)`       = steals_upper,
        `Steals (PI80 Width)`  = steals_pi80_width,
        `Steals (Pred Std)`    = steals_std_pred,
        `Steals (Epi Std)`     = steals_std_epistemic,
        `Steals (Ale Std)`     = steals_std_aleatoric,
        `Steals (Std80 Lower)` = steals_std80_lower,
        `Steals (Std80 Upper)` = steals_std80_upper,
        
        # Blocks
        `Blocks (Mean)`        = blocks_mean,
        `Blocks (Median)`      = blocks_median,
        `Blocks (Lower)`       = blocks_lower,
        `Blocks (Upper)`       = blocks_upper,
        `Blocks (PI80 Width)`  = blocks_pi80_width,
        `Blocks (Pred Std)`    = blocks_std_pred,
        `Blocks (Epi Std)`     = blocks_std_epistemic,
        `Blocks (Ale Std)`     = blocks_std_aleatoric,
        `Blocks (Std80 Lower)` = blocks_std80_lower,
        `Blocks (Std80 Upper)` = blocks_std80_upper,
        
        # 3PM
        `3PM (Mean)`        = three_point_field_goals_made_mean,
        `3PM (Median)`      = three_point_field_goals_made_median,
        `3PM (Lower)`       = three_point_field_goals_made_lower,
        `3PM (Upper)`       = three_point_field_goals_made_upper,
        `3PM (PI80 Width)`  = three_point_field_goals_made_pi80_width,
        `3PM (Pred Std)`    = three_point_field_goals_made_std_pred,
        `3PM (Epi Std)`     = three_point_field_goals_made_std_epistemic,
        `3PM (Ale Std)`     = three_point_field_goals_made_std_aleatoric,
        `3PM (Std80 Lower)` = three_point_field_goals_made_std80_lower,
        `3PM (Std80 Upper)` = three_point_field_goals_made_std80_upper
      ) %>%
      select(
        Player, Team, Opponent, Date, HomeAway,
        `Points (Mean)`, `Points (Lower)`, `Points (Median)`, `Points (Upper)`,
        `Points (PI80 Width)`, `Points (Pred Std)`, `Points (Epi Std)`,
        `Points (Ale Std)`, `Points (Std80 Lower)`, `Points (Std80 Upper)`,
        `Rebounds (Mean)`, `Rebounds (Lower)`, `Rebounds (Median)`, `Rebounds (Upper)`,
        `Rebounds (PI80 Width)`, `Rebounds (Pred Std)`, `Rebounds (Epi Std)`,
        `Rebounds (Ale Std)`, `Rebounds (Std80 Lower)`, `Rebounds (Std80 Upper)`,
        `Assists (Mean)`, `Assists (Lower)`, `Assists (Median)`, `Assists (Upper)`,
        `Assists (PI80 Width)`, `Assists (Pred Std)`, `Assists (Epi Std)`,
        `Assists (Ale Std)`, `Assists (Std80 Lower)`, `Assists (Std80 Upper)`,
        `Steals (Mean)`, `Steals (Lower)`, `Steals (Median)`, `Steals (Upper)`,
        `Steals (PI80 Width)`, `Steals (Pred Std)`, `Steals (Epi Std)`,
        `Steals (Ale Std)`, `Steals (Std80 Lower)`, `Steals (Std80 Upper)`,
        `Blocks (Mean)`, `Blocks (Lower)`, `Blocks (Median)`, `Blocks (Upper)`,
        `Blocks (PI80 Width)`, `Blocks (Pred Std)`, `Blocks (Epi Std)`,
        `Blocks (Ale Std)`, `Blocks (Std80 Lower)`, `Blocks (Std80 Upper)`,
        `3PM (Mean)`, `3PM (Lower)`, `3PM (Median)`, `3PM (Upper)`,
        `3PM (PI80 Width)`, `3PM (Pred Std)`, `3PM (Epi Std)`,
        `3PM (Ale Std)`, `3PM (Std80 Lower)`, `3PM (Std80 Upper)`
      )
  }
  
  # all-column search (case-insensitive), ignores HTML in Player
  filter_by_query <- function(df, q) {
    q <- tolower(trimws(q %||% ""))
    if (q == "") return(df)
    hits <- apply(df, 1, function(row) {
      any(grepl(q, tolower(strip_html(as.character(row))), fixed = TRUE))
    })
    df[hits, , drop = FALSE]
  }
  
  compute_show_cols <- function(df_all, selected_labels) {
    meta <- c("Player","Team","Opponent","Date","HomeAway")
    selected_meta  <- intersect(selected_labels, meta)
    selected_stats <- setdiff(selected_labels, meta)
    stat_cols <- unlist(lapply(
      selected_stats,
      function(lbl) {
        patt <- paste0("^", gsub("([\\[\\]\\(\\)\\+\\?\\^\\$\\\\\\|\\{\\}])","\\\\\\1", lbl), " \\(")
        grep(patt, names(df_all), value = TRUE)
      }
    ), use.names = FALSE)
    c(selected_meta, stat_cols)
  }
  
  build_col_defs <- function(df_any) {
    defs <- setNames(vector("list", length(names(df_any))), names(df_any))
    for (nm in names(df_any)) {
      cls <- paste0("col-", slug(nm))
      defs[[nm]] <- colDef(
        html     = identical(nm, "Player"),
        align    = if (is.numeric(df_any[[nm]])) "right" else "center",
        minWidth = if (nm == "Player") 200 else 110,  # can stretch (no maxWidth caps)
        name     = if (nm == "HomeAway") "Home/Away" else NULL,
        class    = cls,
        headerClass = cls,
        sticky   = if (nm == "Player") "left" else NULL
      )
    }
    defs
  }
  
  # ---- Sticky pagination helpers (compat: no pageSize in updateReactable) ----
  page_size_rv <- reactiveVal(10L)     # source of truth for defaultPageSize
  table_redraw <- reactiveVal(0L)      # bump to force re-render when size changes
  
  filtered_df <- reactive({
    filter_by_query(make_table_df(preds()), input$pred_search)
  })
  
  total_pages <- reactive({
    sz <- as.numeric(page_size_rv())
    n  <- nrow(filtered_df())
    max(1L, ceiling(n / sz))
  })
  
  output$page_max <- renderText({
    as.character(total_pages())
  })
  
  # --------------------------
  # Render predictions table
  # --------------------------
  output$predictions_table <- renderReactable({
    table_redraw()  # depend so page size changes re-render safely
    
    df_all <- filtered_df()
    
    labels <- input$selected_columns
    if (is.null(labels) || !length(labels)) {
      labels <- c("Player","Team","Opponent","Date","HomeAway",
                  "Points","Rebounds","Assists","Steals","Blocks","3PM")
    }
    show_cols <- compute_show_cols(df_all, labels)
    df_show   <- df_all[, show_cols, drop = FALSE]
    col_defs  <- build_col_defs(df_show)
    
    reactable(
      df_show,
      columns   = c(list(.selection = colDef(show = FALSE)), col_defs),  # hide the radio dot
      selection = "single",
      onClick   = "select",
      highlight = FALSE,
      pagination = TRUE,
      defaultPageSize = as.numeric(page_size_rv()),
      showPageSizeOptions = FALSE,  # custom sticky control
      showPageInfo        = FALSE,
      showPagination      = FALSE,
      searchable = FALSE,           # using the sticky search
      compact    = TRUE,
      fullWidth  = TRUE,
      defaultColDef = colDef(minWidth = 110),
      theme = reactableTheme(
        cellPadding = "6px 8px",
        style    = list(background = "#121212", color = "#FFFFFF"),
        rowStyle = list(borderBottom = "1px solid #007AC1"),
        rowSelectedStyle = list(
          backgroundColor = "rgba(0, 122, 193, 0.18)",
          borderLeft      = "3px solid #007AC1",
          "& td" = list(backgroundColor = "rgba(0, 122, 193, 0.18)")
        )
      )
    )
  })
  
  # ------ Sticky pagination controls (NO pageSize arg to updateReactable) ------
  # Page size change → re-render with new defaultPageSize, then restore page
  observeEvent(input$page_size, ignoreInit = TRUE, {
    sz <- as.numeric(input$page_size)
    n  <- nrow(isolate(filtered_df()))
    tp <- max(1L, ceiling(n / sz))
    cur_page <- min(max(1L, as.integer(isolate(input$page_num) %||% 1L)), tp)
    
    page_size_rv(sz)                 # update defaultPageSize source of truth
    table_redraw(table_redraw() + 1) # force re-render
    
    updateNumericInput(session, "page_num", value = cur_page)
    session$onFlushed(function() {
      reactable::updateReactable("predictions_table", page = cur_page)
    }, once = TRUE)
  })
  
  # Prev / Next buttons
  observeEvent(input$page_prev, {
    new_page <- max(1L, as.integer(isolate(input$page_num) %||% 1L) - 1L)
    updateNumericInput(session, "page_num", value = new_page)
    reactable::updateReactable("predictions_table", page = new_page)
  })
  observeEvent(input$page_next, {
    new_page <- min(total_pages(), as.integer(isolate(input$page_num) %||% 1L) + 1L)
    updateNumericInput(session, "page_num", value = new_page)
    reactable::updateReactable("predictions_table", page = new_page)
  })
  
  # Manual page number edit
  observeEvent(input$page_num, ignoreInit = TRUE, {
    pg <- as.integer(input$page_num); if (is.na(pg)) return()
    pg <- min(total_pages(), max(1L, pg))
    if (pg != input$page_num) updateNumericInput(session, "page_num", value = pg)
    reactable::updateReactable("predictions_table", page = pg)
  })
  
  # Re-clamp page when search text changes (row count changes)
  observeEvent(input$pred_search, ignoreInit = TRUE, {
    tp <- total_pages()
    pg <- min(tp, max(1L, as.integer(isolate(input$page_num) %||% 1L)))
    if (pg != isolate(input$page_num)) {
      updateNumericInput(session, "page_num", value = pg)
      reactable::updateReactable("predictions_table", page = pg)
    }
  })
  
  # Keep current page after column set changes (table re-render happens)
  observeEvent(input$selected_columns, ignoreInit = TRUE, {
    cur_page <- as.integer(isolate(input$page_num) %||% 1L)
    session$onFlushed(function() {
      reactable::updateReactable("predictions_table", page = cur_page)
    }, once = TRUE)
  })
  
  # --------------------------
  # Metrics table
  # --------------------------
  output$metrics_table <- renderReactable({
    m_raw <- metrics()
    validate(need(all(c("Target","Suffix") %in% names(m_raw)), "Metrics file missing required columns (Target/Suffix)."))
    
    m <- m_raw %>%
      dplyr::filter(Suffix == "cal") %>%
      dplyr::mutate(
        Target = dplyr::recode(
          Target,
          "three_point_field_goals_made" = "3PM",
          "rebounds" = "Rebounds",
          "assists"  = "Assists",
          "steals"   = "Steals",
          "blocks"   = "Blocks",
          "points"   = "Points",
          .default = Target
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
      compact = TRUE,
      highlight = FALSE,
      columns = list(
        Target = colDef(name = "Target", align = "left"),
        RMSE_Mean   = colDef(name = "RMSE (Mean)",   format = colFormat(digits = 1), align = "right"),
        MAE_Mean    = colDef(name = "MAE (Mean)",    format = colFormat(digits = 1), align = "right"),
        R2          = colDef(name = "R\u00B2",        format = colFormat(digits = 2), align = "right"),
        RMSE_Median = colDef(name = "RMSE (Median)", format = colFormat(digits = 1), align = "right"),
        MAE_Median  = colDef(name = "MAE (Median)",  format = colFormat(digits = 1), align = "right"),
        Pinball_10  = colDef(name = "Pinball Loss (q=0.10)", format = colFormat(digits = 2), align = "right"),
        Pinball_50  = colDef(name = "Pinball Loss (q=0.50)", format = colFormat(digits = 2), align = "right"),
        Pinball_90  = colDef(name = "Pinball Loss (q=0.90)", format = colFormat(digits = 2), align = "right"),
        PI80_Coverage   = colDef(name = "80% PI Coverage (q10–q90)", format = colFormat(percent = TRUE, digits = 0), align = "right"),
        PI80_Width      = colDef(name = "PI80 Width", format = colFormat(digits = 2), align = "right"),
        Below_Q10_Rate  = colDef(name = "Below q10 Rate", format = colFormat(percent = TRUE, digits = 0), align = "right"),
        Above_Q50_Rate  = colDef(name = "Above q50 Rate", format = colFormat(percent = TRUE, digits = 0), align = "right"),
        Above_Q90_Rate  = colDef(name = "Above q90 Rate", format = colFormat(percent = TRUE, digits = 0), align = "right"),
        STD80_Coverage      = colDef(name = "STD 80% Coverage (± z*std)", format = colFormat(percent = TRUE, digits = 0), align = "right"),
        STD_Predictive_Mean = colDef(name = "Mean Std (Predictive)", format = colFormat(digits = 2), align = "right"),
        STD_Epistemic_Mean  = colDef(name = "Mean Std (Epistemic)",  format = colFormat(digits = 2), align = "right"),
        STD_Aleatoric_Mean  = colDef(name = "Mean Std (Aleatoric)",  format = colFormat(digits = 2), align = "right"),
        Bias_MeanError     = colDef(name = "Bias (Mean Error)",      format = colFormat(digits = 2), align = "right"),
        Uncert_Error_Corr  = colDef(name = "Uncertainty-Error Corr", format = colFormat(digits = 2), align = "right", minWidth = 110)
      ),
      theme = reactableTheme(
        style    = list(background = "#121212", color = "#FFFFFF"),
        rowStyle = list(borderBottom = "1px solid #007AC1")
      )
    )
  })
}

# ==============================
# Run the app
# ==============================
shinyApp(ui, server)
