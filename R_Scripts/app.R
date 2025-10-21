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
  if (http_error(r)) {
    status <- status_code(r)
    stop(sprintf("HTTP %s while fetching %s", status, url))
  }
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
  "navbar-bg"              = "#121212",
  "navbar-dark-color"      = "#FFFFFF",
  "navbar-dark-hover-color"= "#007AC1"
)

# ==============================
# UI
# ==============================
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
  ")),
    # JS: inject CSS to hide/show columns by class (no re-render, pagination stays)
    tags$script(HTML("
    Shiny.addCustomMessageHandler('rt-hide-cols-css', function(msg){
      var styleId = 'hide-cols-' + msg.id;
      var node = document.getElementById(styleId);
      if (!node) {
        node = document.createElement('style');
        node.id = styleId;
        document.head.appendChild(node);
      }
      node.textContent = msg.css || '';
    });
  "))
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
                              "Points","Rebounds","Assists","Steals","Blocks","3-Point FG"),
                  selected = c("Player","Team","Opponent","Date","HomeAway",
                               "Points","Rebounds","Assists","Steals","Blocks","3-Point FG")
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
              reactableOutput("predictions_table")
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
          "This page explains all prediction columns and evaluation metrics used in the app."),
        tags$hr(style = "border-top: 1px solid #007AC1;"),
        
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
          tags$li(tags$b("Median"), ": 50th percentile."),
          tags$li(tags$b("Lower (q10)"), ": 10th percentile — closer to mean ⇒ tighter, much lower ⇒ more downside risk."),
          tags$li(tags$b("Upper (q90)"), ": 90th percentile — much higher than mean ⇒ more upside spread."),
          tags$li(tags$b("PI80 Width"), ": (Upper − Lower). High ⇒ big plausible range; Low ⇒ tight expectation."),
          tags$li(tags$b("Pred Std"), ": Predictive std (epistemic + aleatoric). High ⇒ wide outcomes; Low ⇒ consistent."),
          tags$li(tags$b("Epi Std"), ": What the model doesn’t know (context/data limits). High ⇒ new/shifted context."),
          tags$li(tags$b("Ale Std"), ": Inherent randomness. High ⇒ volatile stat/matchup."),
          tags$li(tags$b("Std80 Lower / Std80 Upper"), ": ±1.2816 × Pred Std (≈80%). Wider ⇒ more uncertainty.")
        ),
        
        tags$hr(style = "border-top: 1px solid #007AC1; margin: 2rem 0;"),
        
        h3("Metrics"),
        p("Computed on historical data to judge calibration and accuracy."),
        tags$ul(
          tags$li(tags$b("RMSE (Mean)"), ": Lower is better; punishes big misses."),
          tags$li(tags$b("MAE (Mean)"),  ": Lower is better; typical miss size."),
          tags$li(tags$b("R²"),          ": Higher is better."),
          tags$li(tags$b("RMSE/MAE (Median)"), ": Same metrics for median predictions."),
          tags$li(tags$b("Pinball Loss (q=0.10/0.50/0.90)"), ": Lower is better; quantile accuracy."),
          tags$li(tags$b("80% PI Coverage (q10–q90)"), ": ≈80% is ideal (wider ⇒ too wide; lower ⇒ too narrow)."),
          tags$li(tags$b("PI80 Width"), ": Lower is tighter—balance with coverage."),
          tags$li(tags$b("Below q10 / Above q50 / Above q90"), ": Targets ≈10% / 50% / 10%; deviations ⇒ miscalibration."),
          tags$li(tags$b("STD 80% Coverage (± z·std)"), ": ≈80% ⇒ std well-scaled."),
          tags$li(tags$b("Mean Std (Predictive/Epistemic/Aleatoric)"), ": Lower ⇒ more confidence (watch calibration)."),
          tags$li(tags$b("Bias (Mean Error)"), ": Closer to 0 is better (sign = over/under)."),
          tags$li(tags$b("Uncertainty–Error Corr"), ": Positive desired—model is uncertain when it tends to miss.")
        ),
        
        tags$hr(style = "border-top: 1px solid #007AC1; margin: 2rem 0;"),
        
        h4("Quick Tips"),
        tags$ul(
          tags$li("For conservative plays, focus on ", tags$b("Lower (q10)"), " and small ", tags$b("PI80 Width"), "."),
          tags$li("If ", tags$b("Pred Std"), " is high, check whether it's driven by ", tags$b("Epi Std"), " or ", tags$b("Ale Std"), "."),
          tags$li("Good calibration: coverage near 80% and tail rates near 10%/50%/10%.")
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
  
  # Implied probability calc (unchanged)
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
  # Predictions table
  # --------------------------
  # Build full table (all columns present); used by both render and the column-toggle logic
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
        
        # 3-Point FG
        `3-Point FG (Mean)`        = three_point_field_goals_made_mean,
        `3-Point FG (Median)`      = three_point_field_goals_made_median,
        `3-Point FG (Lower)`       = three_point_field_goals_made_lower,
        `3-Point FG (Upper)`       = three_point_field_goals_made_upper,
        `3-Point FG (PI80 Width)`  = three_point_field_goals_made_pi80_width,
        `3-Point FG (Pred Std)`    = three_point_field_goals_made_std_pred,
        `3-Point FG (Epi Std)`     = three_point_field_goals_made_std_epistemic,
        `3-Point FG (Ale Std)`     = three_point_field_goals_made_std_aleatoric,
        `3-Point FG (Std80 Lower)` = three_point_field_goals_made_std80_lower,
        `3-Point FG (Std80 Upper)` = three_point_field_goals_made_std80_upper
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
        `3-Point FG (Mean)`, `3-Point FG (Lower)`, `3-Point FG (Median)`, `3-Point FG (Upper)`,
        `3-Point FG (PI80 Width)`, `3-Point FG (Pred Std)`, `3-Point FG (Epi Std)`,
        `3-Point FG (Ale Std)`, `3-Point FG (Std80 Lower)`, `3-Point FG (Std80 Upper)`
      )
  }
  
  # Slugify to build safe CSS classes from column names
  slug <- function(x) tolower(gsub("[^a-z0-9]+", "-", x))
  
  # Build col defs once with per-column classes (cells + headers)
  build_col_defs <- function(df_all) {
    defs <- setNames(vector("list", length(names(df_all))), names(df_all))
    for (nm in names(df_all)) {
      cls <- paste0("col-", slug(nm))
      defs[[nm]] <- colDef(
        html = identical(nm, "Player"),
        align = if (is.numeric(df_all[[nm]])) "right" else "center",
        minWidth = if (nm == "Player") 120 else 110,
        name = if (nm == "HomeAway") "Home/Away" else NULL,
        class = cls,        # applies to cells
        headerClass = cls   # applies to header
      )
    }
    defs
  }
  
  # Compute which columns to show from the checkbox labels
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
  
  # Build the CSS string to hide unselected columns
  make_hide_css <- function(all_cols, show_cols) {
    hide <- setdiff(all_cols, show_cols)
    if (!length(hide)) return("")
    sel <- paste(paste0(".", paste0("col-", slug(hide))), collapse = ",")
    paste0(sel, "{display:none !important;}")
  }
  
  # Render once with ALL columns; hide/show via CSS so pagination stays put
  output$predictions_table <- renderReactable({
    df_all   <- make_table_df(preds())
    col_defs <- build_col_defs(df_all)
    
    # initial table render (all columns present)
    tbl <- reactable(
      df_all,
      columns = col_defs,
      pagination = TRUE,
      defaultPageSize   = 10,
      showPageSizeOptions = TRUE,
      pageSizeOptions     = c(5, 10, 15, 20, 25, 100),
      showPagination = TRUE,
      showPageInfo  = TRUE,
      searchable = TRUE,
      highlight  = TRUE,
      compact    = TRUE,
      defaultColDef = colDef(minWidth = 110),
      theme = reactableTheme(
        style    = list(background = "#121212"),
        rowStyle = list(borderBottom = "1px solid #007AC1")
      )
    )
    
    # IMPORTANT: do NOT depend on input$selected_columns here.
    # Just read once to initialize visibility, without creating a dependency.
    init_labels <- isolate(input$selected_columns)
    if (is.null(init_labels) || !length(init_labels)) {
      init_labels <- c("Player","Team","Opponent","Date","HomeAway",
                       "Points","Rebounds","Assists","Steals","Blocks","3-Point FG")
    }
    show_cols <- compute_show_cols(df_all, init_labels)
    css_text  <- make_hide_css(names(df_all), show_cols)
    
    # Apply CSS directly after render (no re-render, no page reset)
    htmlwidgets::onRender(
      tbl,
      sprintf(
        "function(el,x){
         var id = 'predictions_table';
         var css = %s;
         var styleId = 'hide-cols-' + id;
         var node = document.getElementById(styleId);
         if (!node) {
           node = document.createElement('style');
           node.id = styleId;
           document.head.appendChild(node);
         }
         node.textContent = css || '';
       }",
        jsonlite::toJSON(css_text, auto_unbox = TRUE)
      )
    )
  })
  
  # When the user changes the selector: inject new CSS (no re-render)
  observeEvent(input$selected_columns, ignoreInit = TRUE, {
    df_all  <- make_table_df(preds())
    show    <- compute_show_cols(df_all, input$selected_columns)
    css_txt <- make_hide_css(names(df_all), show)
    session$sendCustomMessage("rt-hide-cols-css", list(
      id  = "predictions_table",
      css = css_txt
    ))
  })
  
  # When the parquet refreshes: table re-renders; reapply CSS once
  observeEvent(preds(), ignoreInit = TRUE, {
    df_all  <- make_table_df(preds())
    show    <- compute_show_cols(df_all, isolate(input$selected_columns))
    css_txt <- make_hide_css(names(df_all), show)
    session$onFlushed(function() {
      session$sendCustomMessage("rt-hide-cols-css", list(
        id  = "predictions_table",
        css = css_txt
      ))
    }, once = TRUE)
  })
  
  # --------------------------
  # Metrics table (unchanged)
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
        style    = list(background = "#121212"),
        rowStyle = list(borderBottom = "1px solid #007AC1")
      )
    )
  })
}

# ==============================
# Run the app
# ==============================
shinyApp(ui, server)