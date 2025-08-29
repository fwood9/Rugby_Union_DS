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



def stack_dfs(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Stacks (concatenates) multiple DataFrames row-wise.
    """
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.rename(columns={'date': 'match_date'}, inplace=False)
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values(by='match_date', ascending=True).reset_index(drop=True)
    df = df.drop_duplicates(subset=['match_link'], keep='first')

    df = rename_competitions(df)

    return df


def calculate_expected_probability(home_elo, away_elo, home_advantage):
    """Classic Elo expected win probability formula."""
    return 1 / (1 + 10 ** ((away_elo - home_elo - home_advantage) / 400))


def calculate_expected_margin(home_elo, away_elo, home_advantage):
    """Rescale probability to margin range [-1, +1]."""
    exp_prob = calculate_expected_probability(home_elo, away_elo, home_advantage)
    return (exp_prob - 0.5) * 2


def margin_of_victory_multiplier(margin, max_factor=10, base_factor=1, decay=0.1, default_factor=1.5):
    margin = abs(margin)
    if margin == 0:
        return default_factor
    if margin < 1:
        return base_factor
    return max_factor - (max_factor - base_factor) * math.exp(-decay * (margin - 1))



def compute_pre_and_post_match_elo(
    dataframe: pd.DataFrame,
    initial_rating: float = 1500,
    k_factor: float = 30
) -> pd.DataFrame:
    """
    Compute pre- and post-match Elo ratings with margin-of-victory adjustment always active.
    Skips matches with no valid result.
    """
    # Sort and prepare data
    df = dataframe.sort_values('match_date').copy()
    df['total_home_advantage'] = df['total_home_advantage'].fillna(0)

    # Initialize Elo ratings
    teams = pd.unique(df[['home_team', 'away_team']].values.ravel())
    elo = {team: initial_rating for team in teams}

    # Prepare output structure
    output = df.copy()
    for col in ['home_elo_pre_match', 'away_elo_pre_match',
                'home_elo_post_match', 'away_elo_post_match']:
        output[col] = np.nan

    # Iterate matches
    for idx, row in output.iterrows():
        result = row.get('result')
        if pd.isnull(result) or not result:
            continue

        home = row['home_team']
        away = row['away_team']
        home_pre = elo.get(home, initial_rating)
        away_pre = elo.get(away, initial_rating)

        output.at[idx, 'home_elo_pre_match'] = home_pre
        output.at[idx, 'away_elo_pre_match'] = away_pre

        exp_home = calculate_expected_probability(home_pre, away_pre, row['total_home_advantage'])
        exp_away = 1 - exp_home

        home_scores = row['home_scores']
        away_scores = row['away_scores']

        if home_scores > away_scores:
            actual_home, actual_away = 1.0, 0.0
        elif away_scores > home_scores:
            actual_home, actual_away = 0.0, 1.0
        else:  # draw
            actual_home = actual_away = 0.5

        margin = home_scores - away_scores
        multiplier = margin_of_victory_multiplier(margin)

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
    df = compute_pre_and_post_match_elo(df)

    return df

