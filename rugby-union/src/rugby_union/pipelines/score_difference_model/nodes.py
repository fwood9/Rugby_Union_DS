"""
This is a boilerplate pipeline 'score_difference_model'
generated using Kedro 0.19.14
"""
from prospect.models.training import train_model_ensemble
from prospect.models.training import evaluate_model_ensemble
import pandas as pd

def drop_first_games(df):
    """
    Removes matches where either team is playing their first game,
    while preserving all columns including 'train_test_split' and
    keeping a unique 'observation_id' for each row.

    Assumes 'match_date', 'home_team', and 'away_team' columns exist.
    """

    df = df.copy()
    df['match_date'] = pd.to_datetime(df['match_date'])

    # Get first appearance per team
    team_appearances = pd.concat([
        df[['home_team', 'match_date']].rename(columns={'home_team': 'team'}),
        df[['away_team', 'match_date']].rename(columns={'away_team': 'team'})
    ])

    first_match_dates = team_appearances.groupby('team')['match_date'].min().to_dict()

    df['home_team_first'] = df.apply(lambda row: row['match_date'] == first_match_dates[row['home_team']], axis=1)
    df['away_team_first'] = df.apply(lambda row: row['match_date'] == first_match_dates[row['away_team']], axis=1)

    # Filter out first games
    df_filtered = df[~(df['home_team_first'] | df['away_team_first'])].drop(
        columns=['home_team_first', 'away_team_first']
    )


    df_filtered = df_filtered.copy()
    df_filtered['observation_id'] = df_filtered.index

    return df_filtered


def train_score_difference_model(df):
    X_variables = [
        "total_home_advantage",
        "pre_match_elo_difference",
        "elo_avg_change_difference_last_match",
        "elo_avg_change_difference_last_5_matches",
        "elo_avg_change_difference_last_10_matches",
        "last_avg_match_scores_difference",
        "last_avg_5_match_scores_difference",
        "last_avg_10_match_scores_difference",
        "home_win_streak", "away_win_streak",
        "avg_score_diff_last_3_between_teams",
      #  "avg_score_diff_last_3_same_fixture",
        "total_home_advantage_last_3_games",
        "score_diff_last_fixture"
    ]

    y_variable = "home_margin"
    model_name = "forcasting_future_score_difference_model"
    model_type = "xgboost_regression"

    df = df.copy()
    df = df.dropna(subset=[y_variable] + X_variables)

    return train_model_ensemble(
        df,
        X_variables,
        y_variable,
        model_name,
        model_type,
        bag_n_splits=10,
        train_n_sets=5,
        bag_n_sets=5,
        bag_sample_with_replacement=True,
        bag_split_size=0.8,
        xgboost_cv=3,
        xgboost_learning_rate=[0.01],
        xgboost_n_estimators=[1000],
        xgboost_max_depth=[3],
        xgboost_min_child_weight=[10],
    )

def evaluate_score_difference_model(
    df: pd.DataFrame,
    xgboost_model_ensemble,
    model_error_metrics=None,
    return_train_test_predictions_df=False,
    rename_predicted_columns_dict=None):

    X_variables = [
        "total_home_advantage",
        "pre_match_elo_difference",
        "elo_avg_change_difference_last_match",
        "elo_avg_change_difference_last_5_matches",
        "elo_avg_change_difference_last_10_matches",
        "last_avg_match_scores_difference",
        "last_avg_5_match_scores_difference",
        "last_avg_10_match_scores_difference",
        "home_win_streak", "away_win_streak",
        "avg_score_diff_last_3_between_teams",
        #"avg_score_diff_last_3_same_fixture",
        "total_home_advantage_last_3_games",
        "score_diff_last_fixture"
    ]

    y_variable = "home_margin"

    df = df.copy()
    df = df.dropna(subset=[y_variable] + X_variables)

    return evaluate_model_ensemble(
        df,
        xgboost_model_ensemble,
        model_error_metrics=model_error_metrics or ["r2_score", "mean_absolute_error"],
        return_train_test_predictions_df=return_train_test_predictions_df,
        rename_predicted_columns_dict=rename_predicted_columns_dict or {}
    )




