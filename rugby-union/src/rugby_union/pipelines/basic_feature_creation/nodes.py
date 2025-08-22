"""
This is a boilerplate pipeline 'basic_feature_creation'
generated using Kedro 0.19.14
"""
import pandas as pd
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
    df_calc = df.copy()
    df_calc['home_margin'] = df_calc['home_scores'] - df_calc['away_scores']
    df_calc['away_margin'] = df_calc['away_scores'] - df_calc['home_scores']

    home_adv = df_calc.groupby('home_team')['home_margin'].mean().reset_index()
    home_adv.rename(columns={'home_team': 'team', 'home_margin': 'home_team_advantage'}, inplace=True)

    away_disadv = df_calc.groupby('away_team')['away_margin'].mean().reset_index()
    away_disadv.rename(columns={'away_team': 'team', 'away_margin': 'away_team_disadvantage'}, inplace=True)

    team_avg = pd.merge(home_adv, away_disadv, on='team', how='outer').fillna(0)

    # Only merge the final home/away columns—no extras
    result = (
        df
        .merge(team_avg[['team', 'home_team_advantage']], left_on='home_team', right_on='team', how='left')
        .drop(columns=['team'])
        .merge(team_avg[['team', 'away_team_disadvantage']], left_on='away_team', right_on='team', how='left',
               suffixes=('', '_away'))
        .drop(columns=['team'])
    )
    result["total_home_advantage"] = result["home_team_advantage"] - result["away_team_disadvantage"]
    return result



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


def create_features(df):

    df = create_match_ids(df)
    df = create_team_ids(df)
    df = compute_home_adv(df)

    return df