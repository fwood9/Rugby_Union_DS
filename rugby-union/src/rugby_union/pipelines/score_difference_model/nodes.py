"""
This is a boilerplate pipeline 'score_difference_model'
generated using Kedro 0.19.14
"""
from prospect.models.training import train_model_ensemble
from prospect.models.training import evaluate_model_ensemble
import pandas as pd


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
        xgboost_learning_rate=[0.012],
        xgboost_n_estimators=[1000],
        xgboost_max_depth=[5],
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




