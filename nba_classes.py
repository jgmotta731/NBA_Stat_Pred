import pandas as pd
import numpy as np
from datetime import date, timedelta
from unidecode import unidecode
from nba_api.stats.static import players
import gc
from joblib import load, Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FullPredictionPipeline(BaseEstimator):
    def __init__(self):
        self.targets = ['three_point_field_goals_made', 'rebounds', 'assists', 'steals', 'blocks', 'points']
        self.rolling_cols = [
            'field_goals_made', 'field_goals_attempted', 'three_point_field_goals_made',
            'three_point_field_goals_attempted', 'free_throws_made', 'free_throws_attempted',
            'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists', 'steals', 'blocks',
            'turnovers', 'fouls', 'points', 'minutes', 'plus_minus'
        ]
        self.targets2 = [
            'minutes', 'field_goals_attempted', 'field_goals_made',
            'free_throws_attempted', 'free_throws_made',
            'three_point_field_goals_attempted', 'team_score'
        ]

    def safe_divide(self, numerator, denominator, eps=1e-3):
        return numerator / (denominator + eps)

    def load_models(self):
        self.model_primary = load("nba_base_model.joblib")
        self.model_secondary = load("nba_secondary_model.joblib")
        self.model_cls = load("nba_clf_model.joblib")
        self.model_meta = load("nba_meta_model.joblib")
        self.preprocessor = load("preprocessor_pipeline.joblib")

    def add_generic_missing_flag(self, df):
        nulls = df.isnull().sum(axis=1)
        df["generic_missing_flag"] = (nulls > 0).astype(np.int32)
        return df

    def feature_engineering(self, gamelogs):
        gamelogs = gamelogs.copy()
        gamelogs[gamelogs.select_dtypes('float64').columns] = gamelogs.select_dtypes('float64').apply(pd.to_numeric, downcast='float')
        gamelogs[gamelogs.select_dtypes('int64').columns] = gamelogs.select_dtypes('int64').apply(pd.to_numeric, downcast='integer')
        gamelogs = gamelogs[gamelogs["season"] >= 2022].copy()
        gamelogs = gamelogs.dropna(subset=self.targets).copy()
        player_game_counts = gamelogs.groupby("athlete_display_name").size()
        valid_players = player_game_counts[player_game_counts >= 20].index
        gamelogs = gamelogs[gamelogs["athlete_display_name"].isin(valid_players)].copy()
        gamelogs = gamelogs[gamelogs["minutes"] >= 15].reset_index(drop=True)
        gc.collect()

        gamelogs = gamelogs.sort_values(['game_date', 'athlete_display_name', 'team_abbreviation']).reset_index(drop=True)

        if 'plus_minus' in gamelogs.columns:
            gamelogs['plus_minus'] = (
                gamelogs['plus_minus'].astype(str)
                .str.replace(r'^\+', '', regex=True)
                .replace('None', np.nan)
                .astype(np.float32)
            )

        for col in ['starter', 'ejected', 'team_winner', 'is_playoff']:
            if col in gamelogs.columns:
                gamelogs[col] = gamelogs[col].fillna(False).astype(np.int32)

        gamelogs = gamelogs.sort_values(['game_date', 'season', 'athlete_display_name']).reset_index(drop=True)
        gamelogs['is_playoff'] = gamelogs['season_type'].isin([3, 5]).astype(np.int32)
        gamelogs['game_date'] = pd.to_datetime(gamelogs['game_date'])
        gamelogs['days_since_last_game'] = gamelogs.groupby('athlete_display_name')['game_date'].diff().dt.days.astype(np.float32)

        def compute_rolling_features(col):
            result = pd.DataFrame(index=gamelogs.index)
            for window in [5, 20]:
                def apply_rolling(series):
                    shifted = series.shift(1)
                    roll_mean = shifted.rolling(window=window, min_periods=1).mean()
                    roll_std = shifted.rolling(window=window, min_periods=1).std()
                    shift2 = series.shift(2)
                    roll_shift2_mean = shift2.rolling(window=window, min_periods=1).mean()
                    ewm_mean = series.shift(1).ewm(span=window, adjust=False).mean()
                    return pd.DataFrame({
                        f'{col}_rolling{window}': roll_mean,
                        f'{col}_rolling_std{window}': roll_std,
                        f'{col}_rolling_zscore{window}': (shifted - roll_mean) / (roll_std + 1e-6),
                        f'{col}_momentum{window}': shifted - roll_shift2_mean,
                        f'{col}_rolling_cv{window}': roll_std / (roll_mean + 1e-6),
                        f'{col}_consistency_index_rolling{window}': roll_mean / (roll_std + 1e-6),
                        f'{col}_ewm_rolling{window}': ewm_mean
                    }, index=series.index)

                rolled = gamelogs[[col, 'athlete_display_name']].groupby(
                    'athlete_display_name', group_keys=False)[col]\
                    .apply(apply_rolling).reset_index(level=0, drop=True)

                rolled[f'{col}_global_rolling_rank{window}'] = rolled[f'{col}_rolling{window}']\
                    .groupby(gamelogs['season']).rank(method="average", pct=True).astype(np.float32)

                result = pd.concat([result, rolled], axis=1)
            return result

        rolling_results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(compute_rolling_features)(col) for col in self.rolling_cols)
        gamelogs = pd.concat([gamelogs] + rolling_results, axis=1)

        def compute_trend_stats_for_player(name, group, col, window):
            group = group.sort_values("game_date")
            values = group[col].shift(1).to_numpy()
            indices = group.index.to_numpy()
            slopes = np.full(len(values), np.nan, dtype=np.float32)
            intercepts = np.full(len(values), np.nan, dtype=np.float32)
            r2_scores = np.full(len(values), np.nan, dtype=np.float32)
            for i in range(window - 1, len(values)):
                y = values[i - window + 1:i + 1]
                x = np.arange(len(y)).reshape(-1, 1)
                mask = ~np.isnan(y)
                if np.count_nonzero(mask) >= 3 and not np.allclose(y[mask], y[mask][0]):
                    try:
                        model = LinearRegression().fit(x[mask], y[mask])
                        y_pred = model.predict(x[mask])
                        slopes[i] = model.coef_[0]
                        intercepts[i] = model.intercept_
                        r2_scores[i] = r2_score(y[mask], y_pred)
                    except Exception:
                        continue
            return indices, slopes, intercepts, r2_scores

        for window in [5, 20]:
            for col in self.targets:
                results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
                    delayed(compute_trend_stats_for_player)(name, group, col, window)
                    for name, group in gamelogs.groupby('athlete_display_name'))
                slope_vals = np.full(len(gamelogs), np.nan, dtype=np.float32)
                intercept_vals = np.full(len(gamelogs), np.nan, dtype=np.float32)
                r2_vals = np.full(len(gamelogs), np.nan, dtype=np.float32)
                for indices, slopes, intercepts, r2s in results:
                    slope_vals[indices] = slopes
                    intercept_vals[indices] = intercepts
                    r2_vals[indices] = r2s
                gamelogs[f'{col}_trend_slope_{window}'] = slope_vals
                gamelogs[f'{col}_trend_intercept_{window}'] = intercept_vals
                gamelogs[f'{col}_trend_r2_{window}'] = r2_vals

        def compute_opponent_team_rolling(window):
            df = pd.DataFrame(index=gamelogs.index)
            for group_col, stat_col, label in [
                ("opponent_team_abbreviation", "team_score", "opponent_points_allowed"),
                ("opponent_team_abbreviation", "three_point_field_goals_made", "opponent_3pm_allowed"),
                ("opponent_team_abbreviation", "field_goals_made", "opponent_fgm_allowed"),
                ("opponent_team_abbreviation", "rebounds", "opponent_rebounds_allowed"),
                ("team_abbreviation", "team_score", "team_score"),
                ("team_abbreviation", "rebounds", "team_rebounds"),
                ("team_abbreviation", "turnovers", "team_turnovers"),
                ("team_abbreviation", "three_point_field_goals_made", "team_3pm"),
                ("team_abbreviation", "three_point_field_goals_attempted", "team_3pa"),
                ("team_abbreviation", "assists", "team_assists"),
            ]:
                df[f"{label}_rolling{window}"] = (
                    gamelogs[[stat_col, group_col]]
                    .groupby(group_col)[stat_col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                ).astype(np.float32)

            df[f"opponent_defense_rank_rolling{window}"] = (
                df[f"opponent_points_allowed_rolling{window}"]
                .groupby(gamelogs["season"])
                .rank(method="average", pct=True)
            ).astype(np.float32)

            df[f"team_offense_rank_rolling{window}"] = (
                df[f"team_score_rolling{window}"]
                .groupby(gamelogs["season"])
                .rank(method="average", pct=True)
            ).astype(np.float32)

            df[f'assist_to_turnover_rolling{window}'] = self.safe_divide(
                gamelogs[f'assists_rolling{window}'],
                gamelogs[f'turnovers_rolling{window}']
            ).astype(np.float32)

            df[f'orb_pct_rolling{window}'] = self.safe_divide(
                gamelogs[f'offensive_rebounds_rolling{window}'],
                gamelogs[f'rebounds_rolling{window}']
            ).astype(np.float32)

            df[f'fg_pct_rolling{window}'] = self.safe_divide(
                gamelogs[f'field_goals_made_rolling{window}'],
                gamelogs[f'field_goals_attempted_rolling{window}']
            ).clip(0, 1).astype(np.float32)

            df[f'ft_pct_rolling{window}'] = self.safe_divide(
                gamelogs[f'free_throws_made_rolling{window}'],
                gamelogs[f'free_throws_attempted_rolling{window}']
            ).clip(0, 1).astype(np.float32)

            df[f'three_pt_pct_rolling{window}'] = self.safe_divide(
                gamelogs[f'three_point_field_goals_made_rolling{window}'],
                gamelogs[f'three_point_field_goals_attempted_rolling{window}']
            ).clip(0, 1).astype(np.float32)

            df[f'usage_proxy_rolling{window}'] = self.safe_divide(
                gamelogs[f'field_goals_attempted_rolling{window}'] +
                gamelogs[f'free_throws_attempted_rolling{window}'] +
                gamelogs[f'turnovers_rolling{window}'],
                gamelogs[f'minutes_rolling{window}']
            ).astype(np.float32)

            return df

        team_results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(compute_opponent_team_rolling)(window) for window in [5, 20])
        gamelogs = pd.concat([gamelogs] + team_results, axis=1)

        def compute_expanding_features(col):
            df = pd.DataFrame(index=gamelogs.index)
            sorted_gamelogs = gamelogs.sort_values('game_date')
            group = sorted_gamelogs.groupby(['athlete_display_name', 'season'])[col]
            expanding_mean = group.transform(lambda x: x.shift(1).expanding().mean()).astype(np.float32)
            expanding_std = group.transform(lambda x: x.shift(1).expanding().std()).astype(np.float32)
            df[f'{col}_season_expanding_mean'] = expanding_mean
            df[f'{col}_season_expanding_std'] = expanding_std
            df[f'{col}_season_expanding_zscore'] = (
                (sorted_gamelogs[col].shift(1) - expanding_mean) / (expanding_std + 1e-6)
            ).astype(np.float32)
            df[f'{col}_season_expanding_cv'] = (expanding_std / (expanding_mean + 1e-6)).astype(np.float32)
            df[f'{col}_season_consistency_index'] = (expanding_mean / (expanding_std + 1e-6)).astype(np.float32)
            df[f'{col}_season_expanding_rank'] = expanding_mean.rank(method='average', pct=True).astype(np.float32)
            df[f'{col}_rolling_vs_opp5'] = (
                sorted_gamelogs
                .groupby(['athlete_display_name', 'opponent_team_abbreviation'])[col]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            ).astype(np.float32)
            return df

        expanding_results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(compute_expanding_features)(col) for col in self.rolling_cols)
        gamelogs = pd.concat([gamelogs] + expanding_results, axis=1)

        gamelogs['rolling_win_streak'] = (
            gamelogs.sort_values(['athlete_display_name', 'game_date'])
            .groupby('athlete_display_name')['team_winner']
            .transform(lambda x: x.shift(1).groupby((x.shift(1) != 1).cumsum()).cumcount())
        ).astype(np.int32)

        gamelogs['rolling_loss_streak'] = (
            gamelogs.sort_values(['athlete_display_name', 'game_date'])
            .groupby('athlete_display_name')['team_winner']
            .transform(lambda x: x.shift(1).groupby((x.shift(1) != 0).cumsum()).cumcount())
        ).astype(np.int32)

        return gamelogs
    
    def transform(self, X, schedule):
        if not hasattr(self, 'model_meta'):
            self.load_models()

        X = self.feature_engineering(X)
        X = self.add_generic_missing_flag(X)

        # ---------------------------------------------------
        # Feature Selection
        # ---------------------------------------------------
        available_cols = set(X.columns)
        lagged_rolling_features = [col for col in X.columns
                                   if 'rolling' in col or 'trend' in col or 'lag' in col 
                                   or 'expanding' in col or 'momentum' in col]

        numeric_features = lagged_rolling_features + ['days_since_last_game']
        categorical_features = ["home_away", "athlete_position_abbreviation", "is_playoff"]
        features = numeric_features + categorical_features

        def add_generic_missing_flag(X_df, feature_list):
            X_df = X_df.copy()
            X_df['was_missing'] = X_df[feature_list].isna().any(axis=1).astype(int)
            return X_df

        X = add_generic_missing_flag(X, features)

        features = numeric_features + ['was_missing'] + categorical_features
        features = list(dict.fromkeys(features))
        numeric_features = list(dict.fromkeys(numeric_features))
        categorical_features = list(dict.fromkeys(categorical_features))

        features += ['cluster_label', 'was_missing']
        categorical_features += ['cluster_label', 'was_missing']

        self.features = list(dict.fromkeys(features))
        self.numeric_features = list(dict.fromkeys(numeric_features))
        self.categorical_features = list(dict.fromkeys(categorical_features))

        # schedule is now passed as an argument
        latest_players = (
            X[X['season'] == 2025]
            .sort_values('game_date')
            .groupby('athlete_display_name')
            .tail(1)
        )

        schedule_clean = schedule[(schedule['home_abbreviation'] != 'TBD') & (schedule['away_abbreviation'] != 'TBD')].copy()
        schedule_clean['home_away'] = 'home'
        schedule_away = schedule_clean.copy()
        schedule_away['home_away'] = 'away'
        schedule_away[['home_abbreviation', 'away_abbreviation']] = schedule_away[['away_abbreviation', 'home_abbreviation']]
        schedule_expanded = pd.concat([schedule_clean, schedule_away], ignore_index=True)
        schedule_expanded.rename(columns={
            'home_abbreviation': 'team_abbreviation',
            'away_abbreviation': 'opponent_team_abbreviation'
        }, inplace=True)

        schedule_players = schedule_expanded.merge(
            latest_players,
            on='team_abbreviation',
            how='left'
        )
        schedule_players.dropna(subset=['athlete_display_name'], inplace=True)
        schedule_players = schedule_players.drop(columns=[col for col in schedule_players.columns if col.endswith('_y')])
        schedule_players.columns = [col[:-2] if col.endswith('_x') else col for col in schedule_players.columns]
        schedule_players['game_date'] = pd.to_datetime(schedule_players['game_date'], errors='coerce')
        self.schedule_players = schedule_players

        clusters = pd.read_parquet("player_clusters.parquet")
        X = X.merge(clusters[['athlete_display_name', 'cluster_label']], on='athlete_display_name', how='left')
        X['cluster_label'] = X['cluster_label'].fillna(-1).astype(np.int32)

        X_proc = self.preprocessor.transform(X)

        pred_cls = self.model_cls.predict(X_proc)
        pred_secondary = self.model_secondary.predict(X_proc)
        pred_bins = self.model_primary.predict(X_proc)

        X_stacked = np.hstack([X_proc, pred_bins, pred_secondary, pred_cls])
        return X_stacked

    def predict(self, X, schedule, betting_odds_df):
        

        X_stacked = self.transform(X, schedule)
        y_pred_final = self.model_meta.predict(X_stacked)

        pred_df = self.schedule_players[[
            'athlete_display_name','athlete_position_abbreviation',
            'team_abbreviation','opponent_team_abbreviation',
            'game_date','home_away'
        ]].copy()

        pred_cols = ['predicted_three_point_field_goals_made', 'predicted_rebounds', 'predicted_assists',
                     'predicted_steals', 'predicted_blocks', 'predicted_points']
        predictions_df = pd.DataFrame(y_pred_final, columns=pred_cols)
        pred_df = pd.concat([pred_df.reset_index(drop=True), predictions_df], axis=1)

        today = date.today()
        start_of_week = pd.to_datetime(today - timedelta(days=today.weekday()))
        end_of_week = start_of_week + timedelta(days=6)

        pred_df = pred_df[(pred_df['game_date'] >= start_of_week) & (pred_df['game_date'] <= end_of_week)]
        pred_df = pred_df.sort_values('game_date').drop_duplicates('athlete_display_name', keep='first')
        pred_df['game_date'] = pred_df['game_date'].dt.strftime('%Y-%m-%d')

        try:
            player_df = pd.DataFrame(players.get_active_players())
            player_df['headshot_url'] = player_df['id'].apply(
                lambda pid: f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
            )
            player_df['normalized_name'] = player_df['full_name'].apply(lambda name: unidecode(name).title())
            pred_df['normalized_name'] = pred_df['athlete_display_name'].apply(lambda name: unidecode(name).title())
            pred_df = pred_df.merge(player_df, left_on='normalized_name', right_on='normalized_name', how='left')
            pred_df = pred_df.drop(columns=['id', 'full_name', 'first_name', 'last_name', 'is_active', 'normalized_name'])
        except Exception as e:
            print("Warning: could not fetch player headshots:", e)

        pred_df[pred_cols] = pred_df[pred_cols].clip(lower=0)
        pred_df['normalized_name'] = pred_df['athlete_display_name'].str.replace(r'[^\w\s]', '', regex=True)

        # Process provided betting odds DataFrame
        df = betting_odds_df.copy()
        df['market_label'] = df['label'] + " " + df['market'].str.replace('player_', '', regex=False).str.title()

        pivot_price = df.pivot_table(index='description', 
                                     columns='market_label', 
                                     values='price', 
                                     aggfunc='first')
        pivot_point = df.pivot_table(index='description', 
                                     columns='market_label', 
                                     values='point', 
                                     aggfunc='first')

        pivot_price.columns = [f"{col} - Price" for col in pivot_price.columns]
        pivot_point.columns = [f"{col} - Point" for col in pivot_point.columns]

        pivot_df = pd.concat([pivot_price, pivot_point], axis=1)

        stats = set()
        for col in pivot_df.columns:
            if "-" in col:
                label_stat = col.split(" - ")[0]
                label, stat = label_stat.split(" ", 1)
                stats.add(stat)

        sorted_columns = ['player_name']
        for stat in sorted(stats):
            for label in ['Over', 'Under']:
                for measure in ['Price', 'Point']:
                    col_name = f"{label} {stat} - {measure}"
                    if col_name in pivot_df.columns:
                        sorted_columns.append(col_name)

        pivot_df.reset_index(inplace=True)
        pivot_df.rename(columns={'description': 'player_name'}, inplace=True)
        pivot_df = pivot_df[[col for col in sorted_columns if col in pivot_df.columns]]

        betting_odds = pivot_df
        betting_odds['player_name'] = betting_odds['player_name'].str.replace(r'[^\w\s]', '', regex=True)

        pred_df = pred_df.merge(
            betting_odds,
            left_on='normalized_name',
            right_on='player_name',
            how='inner'
        )

        pred_df = pred_df.drop(columns=['player_name', 'normalized_name'])
        pred_df.to_parquet("nba_predictions.parquet", index=False)
        print("Predictions saved to nba_predictions.parquet")

        return y_pred_final
