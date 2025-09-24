"""
This is a boilerplate pipeline 'score_difference_model'
generated using Kedro 0.19.14
"""
from prospect.models.training import train_model_ensemble
from prospect.models.training import evaluate_model_ensemble
from prospect.models.plotting import create_model_predictions_scatter_plot
import pandas as pd


def train_score_difference_model(df):
    """
        Train an ensemble XGBoost regression model to predict the home team's score margin.

        Uses engineered features related to Elo ratings, score differences, team streaks, and home advantage.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing match data with required features and target variable 'home_margin'.

        Returns
        -------
        model_ensemble
            Trained model ensemble returned by `train_model_ensemble` with specified hyperparameters.

        Notes
        -----
        - Drops rows with missing values in features or target before training.
        - Uses bagging and cross-validation for robust training.
        """
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
        "score_diff_last_fixture",
        "win_pct_last_5_diff",
        "win_pct_last_10_diff",
        "win_pct_last_20_diff",
        "win_pct_last_50_diff",
        "international_fixture_indicator",
        "top_leagues_indicator"
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
    """
        Evaluate a trained XGBoost regression ensemble on match data for home score margin prediction.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with the features and target 'home_margin' for evaluation.
        xgboost_model_ensemble : object
            Trained model ensemble to evaluate.
        model_error_metrics : list of str, optional
            Metrics to evaluate, default is ["r2_score", "mean_absolute_error"].
        return_train_test_predictions_df : bool, optional
            Whether to return DataFrame with train/test predictions, default False.
        rename_predicted_columns_dict : dict, optional
            Optional dict to rename prediction columns in returned DataFrame.

        Returns
        -------
        dict or pandas.DataFrame
            Evaluation metrics dictionary or DataFrame with predictions based on parameters.

        Notes
        -----
        - Drops rows with missing values before evaluation.
        """
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
        "score_diff_last_fixture",
        "win_pct_last_5_diff",
        "win_pct_last_10_diff",
        "win_pct_last_20_diff",
        "win_pct_last_50_diff",
        "international_fixture_indicator",
        "top_leagues_indicator",
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


def make_prediction_scatter_plot(df):
    """
    Kedro node wrapper for the existing plot function.
    """
    df = df.dropna()
    fig = create_model_predictions_scatter_plot(df, "home_margin", "home_margin_predicted")
    return fig




