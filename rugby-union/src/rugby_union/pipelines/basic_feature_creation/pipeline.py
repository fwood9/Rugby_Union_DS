"""
This is a boilerplate pipeline 'basic_feature_creation'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from.nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=stack_dfs,
            inputs=["scraped_rugby_odds_df_3", "scraped_rugby_odds_df_2", "scraped_rugby_odds_df_1", "scraped_rugby_odds_df_4"],
            outputs="complete_matches_df",
            name="stack_dataframes_node"
        ),
        node(
            func=create_basic_features,
            inputs="complete_matches_df",
            outputs="complete_matches_df_with_basic_features",
            name="create_features_node"
        ),
        node(
            func=create_elo_features,
            inputs="complete_matches_df_with_basic_features",
            outputs="complete_matches_df_with_elo_features",
            name="create_elo_features_node"
        ),
        node(
            func=drop_first_games,
            inputs="complete_matches_df_with_elo_features",
            outputs="model_input_df",
            name="drop_first_games_node"
        ),
    ])
