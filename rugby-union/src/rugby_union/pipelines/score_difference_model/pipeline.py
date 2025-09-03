"""
This is a boilerplate pipeline 'score_difference_model'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import *
from prospect.models.plotting import create_model_coefficients_summary_plot
from prospect.models.predictions import create_model_ensemble_predictions_df

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=drop_first_games,
            inputs="complete_matches_df_with_elo_features",
            outputs="model_input_df",
            name="drop_first_games_node"
        ),
        node(
            func=train_score_difference_model,
            inputs="model_input_df",
            outputs="xgboost_regression_forecasting_future_score_difference_model_ensemble",
            name="train_score_difference_model"
        ),
        node(
            func=evaluate_score_difference_model,
            inputs=[
                "model_input_df",
                "xgboost_regression_forecasting_future_score_difference_model_ensemble",
                "params:model_error_metrics",
                "params:return_train_test_predictions_df",
                "params:rename_predicted_columns_dict"
            ],
            outputs=[
                "xgboost_regression_forecasting_future_score_difference_model_coefficients_df",
                "xgboost_regression_forecasting_future_score_difference_model_coefficients_summary_df",
                "xgboost_regression_forecasting_future_score_difference_model_predictions_df",
                "xgboost_regression_forecasting_future_score_difference_model_train_predictions_df",
                "xgboost_regression_forecasting_future_score_difference_model_test_predictions_df",
                "xgboost_regression_forecasting_future_score_difference_model_metrics_df",
                "xgboost_regression_forecasting_future_score_difference_model_train_metrics_df",
                "xgboost_regression_forecasting_future_score_difference_model_bag_metrics_df"
            ],
            name="evaluate_score_difference_model",
        ),
        node(
            func=create_model_coefficients_summary_plot,
            inputs="xgboost_regression_forecasting_future_score_difference_model_coefficients_summary_df",
            outputs="model_coefficients_summary_plot",
            name="create_model_coefficients_summary_plot_node",
        ),
        node(
            func=create_model_ensemble_predictions_df,
            inputs=["model_input_df", "xgboost_regression_forecasting_future_score_difference_model_ensemble"],
            outputs=["xgboost_regression_forecasting_future_score_difference_model_ensemble_predictions_df",
                     "xgboost_regression_forecasting_future_score_difference_model_individual_predictions_df"],
            name="create_model_ensemble_predictions_df_node",
        )
    ])
