#!/usr/bin/env Rscript
# =============================================================================
# NBA Data Auto-Updater Launcher
# Runs update_nba_data.R only during the NBA season.
# Includes Windows-safe logging, locking, and diagnostics.
# =============================================================================

# --- Paths and files ---------------------------------------------------------
proj     <- "C:/Users/jgmot/NBA_Prediction_Tool"
rscript  <- "C:/Program Files/R/R-4.4.1/bin/Rscript.exe"
main_r   <- file.path(proj, "R_Scripts", "update_nba_data.R")
log_dir  <- file.path(proj, "logs")
lockfile <- file.path(log_dir, "data_refresh.lock")

# --- Season window (month/day only) ------------------------------------------
in_season <- function(today = Sys.Date()) {
  m <- as.integer(format(today, "%m"))
  d <- as.integer(format(today, "%d"))
  ((m == 10 && d >= 21) || m %in% c(11, 12)) || 
    (m %in% c(1, 2, 3, 4, 5) || (m == 6 && d <= 22))
}

# --- Skip if out of season ---------------------------------------------------
if (!in_season()) {
  message("Out of NBA season — skipping data refresh.")
  quit(status = 0)
}

# --- Ensure logs directory exists --------------------------------------------
if (!dir.exists(log_dir)) dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

# --- Lock file (prevent overlapping runs, treat <30 min as active) -----------
if (file.exists(lockfile)) {
  age_min <- as.numeric(difftime(Sys.time(), file.info(lockfile)$mtime, units = "mins"))
  if (!is.na(age_min) && age_min < 30) {
    message("Lock present — another refresh is likely running. Exiting.")
    quit(status = 0)
  } else {
    message("Stale lock detected — continuing.")
  }
}

ok <- try(file.create(lockfile), silent = TRUE)
if (inherits(ok, "try-error") || !ok) {
  message("Could not create lock file — check permissions.")
  quit(status = 1)
}
on.exit(unlink(lockfile, force = TRUE), add = TRUE)

# --- Set working directory ---------------------------------------------------
setwd(proj)

# --- Log setup ---------------------------------------------------------------
ts <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
logfile <- file.path(log_dir, paste0("data_refresh_", ts, ".log"))
latest  <- file.path(log_dir, "data_refresh_latest.log")

# --- Remove old "latest" log if exists ---------------------------------------
if (file.exists(latest)) file.remove(latest)

# --- Build shell command (safe on Windows) -----------------------------------
cmd <- paste(
  shQuote(rscript),
  shQuote(main_r),
  ">", shQuote(logfile), "2>&1"
)

# --- Run update_nba_data.R and capture status --------------------------------
res <- tryCatch({
  shell(cmd, wait = TRUE)
}, error = function(e) {
  writeLines(paste("Launcher error:", conditionMessage(e)), con = logfile)
  1L
})

# --- Copy to "latest" log ----------------------------------------------------
file.copy(logfile, latest, overwrite = TRUE)

# --- Exit with status for Task Scheduler (0 = success) -----------------------
if (is.null(res)) res <- 0L

if (res != 0L) {
  message("update_nba_data.R failed (exit code ", res, ").")
  quit(status = res)
}

message("update_nba_data.R completed successfully.")
quit(status = 0)