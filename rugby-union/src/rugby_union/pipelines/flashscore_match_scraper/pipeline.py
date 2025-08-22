"""
This is a boilerplate pipeline 'flashscore_match_scraper'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from.nodes import *

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            make_urls,
            inputs=["params:sport", "params:competition", "params:years", "params:countries"],
            outputs="urls",
            tags=["historic"]
        ),

        node(
            create_driver,
            inputs=None,
            outputs="driver",
            name="driver",
            tags=["historic", "weekly"]
        ),


        node(
            extract_raw_data_historic,
            inputs=["urls", "driver"],
            outputs=["raw_matches_h", "raw_odds_h"],
            name="historic_scraper",
            tags=["historic"]
        ),

        node(
            extract_raw_data_weekly,
            inputs=["params:flashscore_rugby_link", "driver"],
            outputs=["raw_matches_w", "raw_odds_w"],
            name="weekly_scraper",
            tags=["weekly"]
        ),

        node(
            clean_match_historic,
            inputs="raw_matches_h",
            outputs="match_df_h",
            name="historic_cleaner",
            tags=["historic"]
        ),

        node(
            clean_match_weekly,
            inputs="raw_matches_w",
            outputs="match_df_w",
            name="weekly_cleaner",
            tags=["weekly"]
        ),

        node(
            calc_mean_odds,
            inputs=["raw_odds_w"],
            outputs="mean_odds_df_w",
            name="weekly_mean_odds",
            tags=["weekly"]
        ),

        node(
            calc_mean_odds,
            inputs=["raw_odds_h"],
            outputs="mean_odds_df_h",
            name="historic_mean_odds",
            tags=["historic"]
        ),

        node(
            merge_dfs,
            inputs=["match_df_w", "mean_odds_df_w"],
            outputs="scraped_rugby_odds_df_w",
            name="merge_dfs_weekly",
            tags=["weekly"],
        ),

        node(
            merge_dfs,
            inputs=["match_df_h", "mean_odds_df_h"],
            outputs="scraped_rugby_odds_df_h",
            name="merge_dfs_historic",
            tags=["historic"],
        ),
    ])
