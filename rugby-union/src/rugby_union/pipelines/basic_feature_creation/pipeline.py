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
            inputs=["scraped_rugby_odds_df_w", "scraped_rugby_odds_df_1"],
            outputs="complete_matches_df",
            name="stack_dataframes_node"
        )
        ,
        node(
            func=create_features,
            inputs="complete_matches_df",
            outputs="complete_matches_with_features_df",
            name="create_features_node"
        )
    ])
