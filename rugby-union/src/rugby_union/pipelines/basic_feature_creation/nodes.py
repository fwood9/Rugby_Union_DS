"""
This is a boilerplate pipeline 'basic_feature_creation'
generated using Kedro 0.19.14
"""
import pandas as pd
import numpy as np
import math


def rename_competitions(df):
    mapping = {"united-rugby-championship-": "urc",
               "european-rugby-champions-cup-": "european-champions-cup",
               "premiership-rugby-": "gallagher-premiership-rugby",
               "top--": "top14", "pro-d-": "prod-12",
               "challenge-cup-": "european-challenge-cup",
               "championship-rugby-": "championship-rugby",
               "premiership-rugby-cup-": "premiership-rugby-cup"
               }
    df["competition"] = df["competition"].replace(mapping)
    return df

def create_match_ids(df):
    df = df.reset_index(drop=True)
    df['match_id'] = df.index + 1001

    return df

def create_team_ids(df):

    unique_teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))

    team_to_id = {team: idx + 20 for idx, team in enumerate(unique_teams)}

    df['home_team_id'] = df['home_team'].map(team_to_id)
    df['away_team_id'] = df['away_team'].map(team_to_id)

    return df


def compute_home_adv(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values('match_date').copy()

    # Compute margins
    df_sorted['home_margin'] = df_sorted['home_scores'] - df_sorted['away_scores']
    df_sorted['away_margin'] = df_sorted['away_scores'] - df_sorted['home_scores']

    # Rolling average of home margin: last 10 HOME games before the current one
    df_sorted['home_team_advantage'] = (
        df_sorted
        .groupby('home_team')['home_margin']
        .rolling(window=10, min_periods=1, closed='left')
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Similarly for away margin: average over last 10 AWAY games before current one
    df_sorted['away_team_disadvantage'] = (
        df_sorted
        .groupby('away_team')['away_margin']
        .rolling(window=10, min_periods=1, closed='left')
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Net home advantage metric
    df_sorted['total_home_advantage'] = (
            df_sorted['home_team_advantage'] - df_sorted['away_team_disadvantage']
    )

    return df_sorted

def compute_home_adv_last_3_games(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values('match_date').copy()

    # Rolling average of home margin: last 3 HOME games before the current one

    df_sorted['home_margin'] = df_sorted['home_scores'] - df_sorted['away_scores']
    df_sorted['away_margin'] = df_sorted['away_scores'] - df_sorted['home_scores']

    df_sorted['home_team_advantage_last_3_games'] = (
        df_sorted
        .groupby('home_team')['home_margin']
        .rolling(window=3, min_periods=1, closed='left')
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Similarly for away margin: average over last 3 AWAY games before current one
    df_sorted['away_team_disadvantage_last_3_games'] = (
        df_sorted
        .groupby('away_team')['away_margin']
        .rolling(window=3, min_periods=1, closed='left')
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Net home advantage metric
    df_sorted['total_home_advantage_last_3_games'] = (
            df_sorted['home_team_advantage_last_3_games'] - df_sorted['away_team_disadvantage_last_3_games']
    )

    return df_sorted



def stack_dfs(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Stacks (concatenates) multiple DataFrames row-wise,
    flexibly parses match_date, and handles malformed entries.
    """
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.rename(columns={'date': 'match_date'})

    # Strip whitespace and drop blank entries
    df['match_date'] = df['match_date'].astype(str).str.strip()
    df = df[df['match_date'] != ""]

    # Flexible datetime parsing
    df['match_date'] = pd.to_datetime(
        df['match_date'],
        errors='coerce',
        infer_datetime_format=True,
        dayfirst=False  # set to True if your data is D/M/Y
    )
    df = df[df['match_date'].notna()]

    df = df.sort_values(by='match_date', ascending=True).reset_index(drop=True)
    df = df.drop_duplicates(subset=['match_link'], keep='first')
    df = rename_competitions(df)

    return df



def calculate_expected_probability(home_elo, away_elo, home_advantage):
    """Classic Elo expected win probability formula."""
    return 1 / (1 + 10 ** ((away_elo - home_elo - home_advantage) / 400))

def margin_of_victory_multiplier(home_elo, away_elo, margin, max_factor=5, base_factor=1, decay=0.1, default_factor=1.5):
    """Adjusts MOV multiplier based on Elo gap."""
    margin = abs(margin)
    elo_gap = home_elo - away_elo
    if margin == 0:
        return default_factor
    if margin < 1:
        return base_factor
    # Calculate MOV multiplier with Elo gap adjustment
    q = 2.2 / ((elo_gap * 0.001) + 2.2)
    return math.log(margin + 1) * q

def compute_pre_and_post_match_elo(dataframe: pd.DataFrame, initial_rating: float = 1500, k_factor: float = 30) -> pd.DataFrame:
    """Compute pre- and post-match Elo ratings with margin-of-victory adjustment."""
    df = dataframe.sort_values('match_date').copy()
    df['total_home_advantage'] = df['total_home_advantage'].fillna(0)

    teams = pd.unique(df[['home_team', 'away_team']].values.ravel())
    elo = {team: initial_rating for team in teams}

    output = df.copy()
    for col in ['home_elo_pre_match', 'away_elo_pre_match', 'home_elo_post_match', 'away_elo_post_match']:
        output[col] = np.nan

    for idx, row in output.iterrows():
        if pd.isnull(row.get('result')) or not row.get('result'):
            continue

        home = row['home_team']
        away = row['away_team']
        home_pre = elo.get(home, initial_rating)
        away_pre = elo.get(away, initial_rating)

        output.at[idx, 'home_elo_pre_match'] = home_pre
        output.at[idx, 'away_elo_pre_match'] = away_pre

        exp_home = calculate_expected_probability(home_pre, away_pre, row['total_home_advantage'])
        exp_away = 1 - exp_home

        margin = row.get('home_margin')
        if pd.isnull(margin):
            continue

        if margin > 0:
            actual_home, actual_away = 1.0, 0.0
        elif margin < 0:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home = actual_away = 0.5

        multiplier = margin_of_victory_multiplier(home_pre, away_pre, margin)

        home_post = home_pre + k_factor * multiplier * (actual_home - exp_home)
        away_post = away_pre + k_factor * multiplier * (actual_away - exp_away)

        output.at[idx, 'home_elo_post_match'] = home_post
        output.at[idx, 'away_elo_post_match'] = away_post

        elo[home] = home_post
        elo[away] = away_post

    return output


def create_basic_features(df):

    df.drop(columns=["home_team_odds", "draw_team_odds", "away_team_odds", "home_prob_BE", "draw_prob_BE", "away_prob_BE", "home_prob", "draw_prob", "away_prob"], inplace=True)

    df = create_match_ids(df)
    df = create_team_ids(df)
    df = compute_home_adv(df)
    df = compute_home_adv_last_3_games(df)
    df = add_team_avg_score_diff(df)
    df = add_team_win_streak(df)
    df = add_fixture_score_diff_avg(df)
    df = add_fixture_home_away_score_diff_avg(df)
    df = add_score_diff_last_fixture(df)

    return df

def create_elo_and_score_difference(df):

    df["pre_match_elo_difference"] = df["home_elo_pre_match"] - df["away_elo_pre_match"]
    df["elo_avg_change_difference_last_match"] = df["home_avg_elo_change_last_match"] - df["away_avg_elo_change_last_match"]
    df["elo_avg_change_difference_last_5_matches"] = df["home_avg_elo_change_last_5_matches"] - df["away_avg_elo_change_last_5_matches"]
    df["elo_avg_change_difference_last_10_matches"] = df["home_avg_elo_change_last_10_matches"] - df["away_avg_elo_change_last_10_matches"]
    df["last_avg_match_scores_difference"] = df["home_avg_score_diff_last_1"] - df["away_avg_score_diff_last_1"]
    df["last_avg_5_match_scores_difference"] = df["home_avg_score_diff_last_5"] - df["away_avg_score_diff_last_5"]
    df["last_avg_10_match_scores_difference"] = df["home_avg_score_diff_last_10"] - df["away_avg_score_diff_last_10"]

    return df


def add_team_elo_changes(df):
    """
    Calculate average Elo change for each team over their last 1, 5, and 10 matches,
    irrespective of whether they played home or away.
    """

    # Prepare two DataFrames: one for home team, one for away team
    home_df = df[['match_date', 'home_team', 'home_elo_pre_match']].copy()
    home_df.columns = ['match_date', 'team', 'elo']

    away_df = df[['match_date', 'away_team', 'away_elo_pre_match']].copy()
    away_df.columns = ['match_date', 'team', 'elo']

    # Combine them into one DataFrame with team matches
    team_matches = pd.concat([home_df, away_df], ignore_index=True)

    # Sort by team and date
    team_matches = team_matches.sort_values(by=['team', 'match_date'])

    # Calculate Elo change per team
    team_matches['elo_change'] = team_matches.groupby('team')['elo'].diff()

    # Calculate rolling averages
    team_matches['avg_elo_change_last_match'] = team_matches.groupby('team')['elo_change'].transform(
        lambda x: x.rolling(window=1, min_periods=1).mean())
    team_matches['avg_elo_change_last_5_matches'] = team_matches.groupby('team')['elo_change'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean())
    team_matches['avg_elo_change_last_10_matches'] = team_matches.groupby('team')['elo_change'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean())

    # Now merge these averages back into the original df for both home and away teams
    # Merge for home team
    home_avg = team_matches[['match_date', 'team', 'avg_elo_change_last_match', 'avg_elo_change_last_5_matches', 'avg_elo_change_last_10_matches']]
    home_avg.columns = ['match_date', 'home_team', 'home_avg_elo_change_last_match', 'home_avg_elo_change_last_5_matches', 'home_avg_elo_change_last_10_matches']
    df = df.merge(home_avg, on=['match_date', 'home_team'], how='left')

    # Merge for away team
    away_avg = team_matches[['match_date', 'team', 'avg_elo_change_last_match', 'avg_elo_change_last_5_matches', 'avg_elo_change_last_10_matches']]
    away_avg.columns = ['match_date', 'away_team', 'away_avg_elo_change_last_match', 'away_avg_elo_change_last_5_matches', 'away_avg_elo_change_last_10_matches']
    df = df.merge(away_avg, on=['match_date', 'away_team'], how='left')

    return df

def add_team_avg_score_diff(df):
    """
    Calculate average score difference (margin) for each team over their last 1, 5, and 10 matches,
    excluding the current match (i.e., only historical matches).

    Assumes df has:
    - 'match_date'
    - 'home_team'
    - 'away_team'
    - 'home_margin'
    - 'away_margin'
    """

    # Prepare DataFrames for home and away teams with their score differences
    home_df = df[['match_date', 'home_team', 'home_margin']].copy()
    home_df.columns = ['match_date', 'team', 'score_diff']

    away_df = df[['match_date', 'away_team', 'away_margin']].copy()
    away_df.columns = ['match_date', 'team', 'score_diff']

    # Combine home and away
    team_matches = pd.concat([home_df, away_df], ignore_index=True)

    # Sort by team and match_date
    team_matches = team_matches.sort_values(by=['team', 'match_date'])

    # Calculate rolling averages of score_diff for last 1, 5, and 10 matches, excluding current match by shifting
    team_matches['avg_score_diff_last_1'] = team_matches.groupby('team')['score_diff'].transform(
        lambda x: x.shift().rolling(window=1, min_periods=1).mean())
    team_matches['avg_score_diff_last_5'] = team_matches.groupby('team')['score_diff'].transform(
        lambda x: x.shift().rolling(window=5, min_periods=1).mean())
    team_matches['avg_score_diff_last_10'] = team_matches.groupby('team')['score_diff'].transform(
        lambda x: x.shift().rolling(window=10, min_periods=1).mean())

    # Merge rolling averages back to original df for home teams
    home_avg = team_matches[['match_date', 'team', 'avg_score_diff_last_1', 'avg_score_diff_last_5', 'avg_score_diff_last_10']]
    home_avg.columns = ['match_date', 'home_team', 'home_avg_score_diff_last_1', 'home_avg_score_diff_last_5', 'home_avg_score_diff_last_10']
    df = df.merge(home_avg, on=['match_date', 'home_team'], how='left')

    # Merge rolling averages back to original df for away teams
    away_avg = team_matches[['match_date', 'team', 'avg_score_diff_last_1', 'avg_score_diff_last_5', 'avg_score_diff_last_10']]
    away_avg.columns = ['match_date', 'away_team', 'away_avg_score_diff_last_1', 'away_avg_score_diff_last_5', 'away_avg_score_diff_last_10']
    df = df.merge(away_avg, on=['match_date', 'away_team'], how='left')

    return df


def add_team_win_streak(df):
    """
    Adds home and away win streaks for each match using the 'winner' column.
    Memory-efficient version.

    Assumes df includes:
    - 'match_date'
    - 'home_team'
    - 'away_team'
    - 'winner' column with values: 'Home', 'Away', 'Draw'
    """

    df = df.sort_values('match_date').copy()
    df['home_win_streak'] = 0
    df['away_win_streak'] = 0

    # Dictionary to track win streaks per team
    streaks = {}

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        winner = row['winner']

        # Get current win streaks (default to 0)
        home_streak = streaks.get(home, 0)
        away_streak = streaks.get(away, 0)

        # Assign current streaks to this row BEFORE updating (historical only)
        df.at[idx, 'home_win_streak'] = home_streak
        df.at[idx, 'away_win_streak'] = away_streak

        # Update streaks after the match result
        if winner == 'Home':
            streaks[home] = home_streak + 1
            streaks[away] = 0
        elif winner == 'Away':
            streaks[away] = away_streak + 1
            streaks[home] = 0
        else:  # Draw
            streaks[home] = 0
            streaks[away] = 0

    return df

def add_fixture_score_diff_avg(df):
    """
    Adds average score difference over the last 5 matches between the two teams,
    always from the perspective of the home team in the current match.
    """
    df = df.sort_values(by='match_date').copy()
    df['avg_score_diff_last_3_between_teams'] = 0.0  # Default to 0.0

    fixture_history = {}

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        fixture_key = tuple(sorted([home, away]))

        if fixture_key not in fixture_history:
            fixture_history[fixture_key] = []

        past_results = fixture_history[fixture_key]
        last_3 = past_results[-3:]

        adjusted_diffs = []
        for past_home, past_away, past_diff in last_3:
            if past_home == home:
                adjusted_diffs.append(past_diff)
            else:
                adjusted_diffs.append(-past_diff)

        avg_diff = sum(adjusted_diffs) / len(adjusted_diffs) if adjusted_diffs else 0.0
        df.at[idx, 'avg_score_diff_last_3_between_teams'] = avg_diff

        current_diff = row['home_scores'] - row['away_scores']
        fixture_history[fixture_key].append((home, away, current_diff))

    return df

def add_fixture_home_away_score_diff_avg(df):
    """
    Adds the average score difference over the last 3 games for the same fixture
    (same home team vs same away team), excluding the current match.
    """
    df = df.sort_values(by='match_date').copy()
    df['avg_score_diff_last_3_same_fixture'] = 0.0  # Default to 0.0

    fixture_history = {}

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        fixture_key = (home, away)

        if fixture_key not in fixture_history:
            fixture_history[fixture_key] = []

        past_diffs = fixture_history[fixture_key]
        last_3 = past_diffs[-3:]

        avg_diff = sum(last_3) / len(last_3) if last_3 else 0.0
        df.at[idx, 'avg_score_diff_last_3_same_fixture'] = avg_diff

        current_diff = row['home_scores'] - row['away_scores']
        fixture_history[fixture_key].append(current_diff)

    return df

def add_score_diff_last_fixture(df):

    df = df.sort_values(by='match_date').copy()
    df['avg_score_diff_last_3_same_fixture'] = 0.0  # Default to 0.0

    fixture_history = {}

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        fixture_key = (home, away)

        if fixture_key not in fixture_history:
            fixture_history[fixture_key] = []

        past_diffs = fixture_history[fixture_key]
        last_fixt = past_diffs[-1:]

        avg_diff = sum(last_fixt) / len(last_fixt) if last_fixt else 0.0
        df.at[idx, 'score_diff_last_fixture'] = avg_diff

        current_diff = row['home_scores'] - row['away_scores']
        fixture_history[fixture_key].append(current_diff)

    return df

def create_elo_features(df):

    df = compute_pre_and_post_match_elo(df)
    df = add_team_elo_changes(df)
    df = create_elo_and_score_difference(df)

    return df