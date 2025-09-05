"""
This is a boilerplate pipeline 'basic_feature_creation'
generated using Kedro 0.19.14
"""
import pandas as pd
import numpy as np
import math


def rename_competitions(df):
    """
    Rename competition identifiers using a predefined mapping.

    Replaces substrings in the `competition` column of the input DataFrame
    according to a fixed mapping of longer identifiers to short-form labels.
    This is useful for standardizing competition names across data sources.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least a `competition` column with
        string entries representing competition identifiers.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame, with the `competition` column modified:
        each value in `competition` that matches a key in the mapping
        is replaced with the corresponding value.

    Notes
    -----
    The replacement is done in-place on the `competition` column via
    ``DataFrame.replace(mapping)``. If you require preserving the original
    column, consider creating a copy of the DataFrame prior to calling this
    function.

    Examples
    --------
     df = pd.DataFrame({
    ...     'competition': ['united-rugby-championship-', 'top--', 'unknown-competition-']
    ... })
     rename_competitions(df)
                          competition
    0                          urc
    1                         top14
    2         unknown-competition-
    """
    mapping = {
        "united-rugby-championship-": "urc",
        "european-rugby-champions-cup-": "european-champions-cup",
        "premiership-rugby-": "gallagher-premiership-rugby",
        "top--": "top14",
        "pro-d-": "prod-12",
        "challenge-cup-": "european-challenge-cup",
        "championship-rugby-": "championship-rugby",
        "premiership-rugby-cup-": "premiership-rugby-cup",
        "serie-a-elite-": "serie-a-elite",
        "didi--": "didi10",
        "coppa-italia-": "coppa-italia",
        "league-one-": "league-one",
        "cn-honra-": "cn-honra",
        "nationale-": "nationale",
    }
    df["competition"] = df["competition"].replace(mapping)
    return df


def create_match_ids(df):
    """
    Assign sequential match IDs to each row.

    The DataFrame index is reset (dropping the existing index), and
    a new column `match_id` is added. IDs start from 1001 and
    increment by one for each row.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing match-level records. Index is used to
        generate the new `match_id` values.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with its index reset and a new column
        `match_id`.

    Examples
    --------
  df = pd.DataFrame({'home_team': ['A', 'B'], 'away_team': ['C', 'D']})
  create_match_ids(df)
      home_team away_team  match_id
    0         A         C      1001
    1         B         D      1002
    """
    df = df.reset_index(drop=True)
    df['match_id'] = df.index + 1001
    return df


def create_team_ids(df):
    """
    Map each team to a unique numeric ID and add team ID columns.

    Identifies all unique team names in the `home_team` and `away_team`
    columns, assigns each a unique ID (starting at 20), and adds two new
    columns, `home_team_id` and `away_team_id`, to the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing `home_team` and `away_team` string columns.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with additional columns:
        - `home_team_id`: integer ID for the home team
        - `away_team_id`: integer ID for the away team

    Examples
    --------
   df = pd.DataFrame({
    ...     'home_team': ['X', 'Y'],
    ...     'away_team': ['Y', 'Z']
    ... })
  create_team_ids(df)
      home_team away_team  home_team_id  away_team_id
    0         X         Y             20             21
    1         Y         Z             21             22
    """
    unique_teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))
    team_to_id = {team: idx + 20 for idx, team in enumerate(unique_teams)}
    df['home_team_id'] = df['home_team'].map(team_to_id)
    df['away_team_id'] = df['away_team'].map(team_to_id)
    return df


def stack_dfs(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Stack multiple DataFrames and normalize match_date entries.

    Concatenates input DataFrames row-wise, strips and filters out
    blank or malformed `match_date` entries, parses dates into
    datetime objects, removes duplicates by `match_link`, sorts by
    ascending date, and applies standardized competition names.

    Parameters
    ----------
    *dfs : pandas.DataFrame
        One or more DataFrames, each expected to include at least
        `match_date` (or `date`) and `match_link` columns.

    Returns
    -------
    pandas.DataFrame
        A single consolidated DataFrame with:
        - normalized and parsed `match_date` as `datetime64[ns]`,
        - duplicates removed based on `match_link`,
        - sorted entries by `match_date`,
        - `competition` values standardized via `rename_competitions`.

    Notes
    -----
    - Input columns named `date` are renamed internally to `match_date`.
    - Rows with missing or unparsable `match_date` are dropped.
    - The function depends on `rename_competitions` to standardize competition names.

    Examples
    --------
     df1 = pd.DataFrame({
    ...     'match_link': ['a', 'b'],
    ...     'date': ['2025-08-01', ''],
    ...     'competition': ['united-rugby-championship-', 'premiership-rugby-']
    ... })
     df2 = pd.DataFrame({
    ...     'match_link': ['c'],
    ...     'match_date': ['2025-08-02'],
    ...     'competition': ['top--']
    ... })
     result = stack_dfs(df1, df2)
     result[['match_link', 'match_date', 'competition']]
      match_link match_date competition
    0          a  2025-08-01         urc
    1          c  2025-08-02       top14
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
        dayfirst=False
    )
    df = df[df['match_date'].notna()]
    df = df.sort_values(by='match_date', ascending=True).reset_index(drop=True)
    df = df.drop_duplicates(subset=['match_link'], keep='first')
    df = rename_competitions(df)

    return df


def compute_home_adv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling home advantage metrics for rugby matches.

    Computes a rolling average home advantage based on the margin of victory
    over the last 10 games for each team. Special handling is applied for
    World Cup matches to zero out advantage unless the home team is the host
    for that year.

    Parameters
    ----------
    df : pandas.DataFrame
        Input match-level DataFrame with the following required columns:
        - `home_team`: str, name of the home team.
        - `away_team`: str, name of the away team.
        - `home_scores`: float or int, score of the home team.
        - `away_scores`: float or int, score of the away team.
        - `match_date`: datetime-like or str, date of the match.
        - `competition`: str, name of the competition.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following additional columns:
        - `home_margin`: Home score minus away score.
        - `away_margin`: Away score minus home score.
        - `home_team_advantage`: Rolling 10-match average of `home_margin` for home teams.
        - `away_team_disadvantage`: Rolling 10-match average of `away_margin` for away teams.
        - `match_year`: Extracted year from `match_date`.
        - `total_home_advantage`: Net advantage: `home_team_advantage` - `away_team_disadvantage`.

    Notes
    -----
    - Rolling averages use a 10-match window, excluding the current match (`closed='left'`).
    - For World Cup matches, only the host nation retains home advantage; others are set to zero.
    - Away disadvantage is always zero for World Cup matches.

    Examples
    --------
     df = pd.DataFrame({
    ...     'home_team': ['France', 'France', 'New Zealand'],
    ...     'away_team': ['England', 'Wales', 'Australia'],
    ...     'home_scores': [24, 30, 18],
    ...     'away_scores': [17, 25, 20],
    ...     'match_date': ['2023-09-01', '2023-09-10', '2011-09-09'],
    ...     'competition': ['world-cup', 'world-cup', 'world-cup']
    ... })
     compute_home_adv(df)[['home_team', 'total_home_advantage']]
        home_team  total_home_advantage
    0     France                   0.0
    1     France                  11.5
    2  New Zealand                  0.0
    """
    WORLD_CUP_HOSTS = {
        2011: 'New Zealand',
        2015: 'England',
        2019: 'Japan',
        2023: 'France'
    }
    df['match_date'] = pd.to_datetime(df['match_date'], errors='raise')
    df_sorted = df.sort_values('match_date').copy()

    # Compute raw margins
    df_sorted['home_margin'] = df_sorted['home_scores'] - df_sorted['away_scores']
    df_sorted['away_margin'] = df_sorted['away_scores'] - df_sorted['home_scores']

    # Rolling average last 10 games
    df_sorted['home_team_advantage'] = (
        df_sorted
        .groupby('home_team')['home_margin']
        .rolling(window=10, min_periods=1, closed='left')
        .mean()
        .reset_index(level=0, drop=True)
    )

    df_sorted['away_team_disadvantage'] = (
        df_sorted
        .groupby('away_team')['away_margin']
        .rolling(window=10, min_periods=1, closed='left')
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Determine if match is World Cup and get year
    df_sorted['match_year'] = df_sorted['match_date'].dt.year
    mask_wc = df_sorted['competition'] == 'world-cup'

    # Only the home team that matches the official host for that year keeps advantage
    df_sorted.loc[mask_wc, 'home_team_advantage'] = df_sorted.loc[mask_wc].apply(
        lambda row: row['home_team_advantage']
        if WORLD_CUP_HOSTS.get(row['match_year']) == row['home_team']
        else 0,
        axis=1
    )

    # Away team disadvantage always zero for World Cup
    df_sorted.loc[mask_wc, 'away_team_disadvantage'] = 0

    # Net home advantage
    df_sorted['total_home_advantage'] = (
        df_sorted['home_team_advantage'] - df_sorted['away_team_disadvantage']
    )

    return df_sorted


def compute_home_adv_last_3_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute short‑term home advantage metrics based on last 3 games.

    Calculates rolling averages of match margin over the previous
    three games for home teams and away teams (as a disadvantage). For
    World Cup matches, applies special logic to ensure only the host
    retains advantage while all away disadvantage is zeroed.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at minimum:
        - `home_team`, `away_team`: strings indicating participating teams.
        - `home_scores`, `away_scores`: numeric scores of the match.
        - `match_date`: datetime-like or string date of the match.
        - `competition`: string identifying the competition.

    Returns
    -------
    pandas.DataFrame
        A copy of the original DataFrame, sorted by `match_date`, with
        these additional columns:
        - `home_margin`: difference between home and away scores.
        - `away_margin`: difference between away and home scores.
        - `home_team_advantage_last_3_games`: mean of home margins over
          the previous three home games (excludes the current match).
        - `away_team_disadvantage_last_3_games`: mean away margins over
          previous three away games (excludes current match).
        - `match_year`: year extracted from `match_date`.
        - `total_home_advantage_last_3_games`: home advantage minus away
          disadvantage.
        Special handling for World Cup matches:
        - `home_team_advantage_last_3_games` is zeroed unless home team
          is host for that year.
        - `away_team_disadvantage_last_3_games` is always set to zero.

    Notes
    -----
    - Rolling mean uses a window of 3 games, with `min_periods=1` and
      `closed='left'` to exclude the current match.
    - Hosts for World Cups are defined in `WORLD_CUP_HOSTS` mapping.

    Examples
    --------
     df = pd.DataFrame({
    ...     'home_team': ['France', 'France', 'Japan'],
    ...     'away_team': ['England', 'Wales', 'Scotland'],
    ...     'home_scores': [24, 30, 18],
    ...     'away_scores': [17, 25, 22],
    ...     'match_date': ['2023-09-01', '2023-09-10', '2019-09-09'],
    ...     'competition': ['world-cup', 'world-cup', 'world-cup']
    ... })
     result = compute_home_adv_last_3_games(df)
     result[['home_team', 'total_home_advantage_last_3_games']]
        home_team  total_home_advantage_last_3_games
    0       France                             0.0
    1       France                             6.5
    2        Japan                             0.0
    """
    WORLD_CUP_HOSTS = {
        2011: 'New Zealand',
        2015: 'England',
        2019: 'Japan',
        2023: 'France'
    }
    df['match_date'] = pd.to_datetime(df['match_date'], errors='raise')
    df_sorted = df.sort_values('match_date').copy()

    # Compute raw margins
    df_sorted['home_margin'] = df_sorted['home_scores'] - df_sorted['away_scores']
    df_sorted['away_margin'] = df_sorted['away_scores'] - df_sorted['home_scores']

    # Rolling average over last 3 games
    df_sorted['home_team_advantage_last_3_games'] = (
        df_sorted
        .groupby('home_team')['home_margin']
        .rolling(window=3, min_periods=1, closed='left')
        .mean()
        .reset_index(level=0, drop=True)
    )

    df_sorted['away_team_disadvantage_last_3_games'] = (
        df_sorted
        .groupby('away_team')['away_margin']
        .rolling(window=3, min_periods=1, closed='left')
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Year and World Cup flag
    df_sorted['match_year'] = df_sorted['match_date'].dt.year
    mask_wc = df_sorted['competition'] == 'world-cup'

    # Apply host-based override for home advantage
    df_sorted.loc[mask_wc, 'home_team_advantage_last_3_games'] = (
        df_sorted.loc[mask_wc]
        .apply(lambda row: row['home_team_advantage_last_3_games']
               if WORLD_CUP_HOSTS.get(row['match_year']) == row['home_team']
               else 0, axis=1)
    )

    # Set away disadvantage to zero for World Cups
    df_sorted.loc[mask_wc, 'away_team_disadvantage_last_3_games'] = 0

    # Compute net advantage
    df_sorted['total_home_advantage_last_3_games'] = (
        df_sorted['home_team_advantage_last_3_games'] -
        df_sorted['away_team_disadvantage_last_3_games']
    )

    return df_sorted


def calculate_expected_probability(home_elo, away_elo, home_advantage):
    """
    Calculate the expected win probability for the home team using Elo ratings.

    Parameters
    ----------
    home_elo : float
        Elo rating of the home team.
    away_elo : float
        Elo rating of the away team.
    home_advantage : float
        Elo rating advantage applied to the home team.

    Returns
    -------
    float
        Win probability for the home team.

    Notes
    -----
    Uses the logistic Elo expected probability formula:
    1 / (1 + 10 ** ((away_elo - home_elo - home_advantage) / 400)).

    Examples
    --------
     calculate_expected_probability(1500, 1400, 0)
    0.640065
    """
    return 1 / (1 + 10 ** ((away_elo - home_elo - home_advantage) / 400))


def margin_of_victory_multiplier(home_elo, away_elo, margin, max_factor=5, base_factor=1, decay=0.1, default_factor=1.5):
    """
    Compute an Elo margin-of-victory multiplier.

    Parameters
    ----------
    home_elo : float
        Elo rating of the home team.
    away_elo : float
        Elo rating of the away team.
    margin : float
        Score differential (home minus away).
    max_factor : float, optional
        Not currently used.
    base_factor : float, optional
        Factor used when margin is positive but less than 1.
    decay : float, optional
        Not currently used.
    default_factor : float, optional
        Factor used when margin is zero.

    Returns
    -------
    float
        Multiplier based on margin and Elo rating difference.

    Notes
    -----
    Applies a logarithmic scaling for margin with an Elo-difference dampening factor.

    Examples
    --------
     margin_of_victory_multiplier(1500, 1400, 5)
    <computed value>
     margin_of_victory_multiplier(1500, 1500, 0)
    1.5
    """
    margin = abs(margin)
    elo_gap = home_elo - away_elo
    if margin == 0:
        return default_factor
    if margin < 1:
        return base_factor
    q = 2.2 / (2.2 + (elo_gap * 0.001))
    return math.log(margin + 1) * q


def compute_pre_and_post_match_elo(dataframe: pd.DataFrame,
                                   initial_rating: float = 1500,
                                   k_factor: float = 30) -> pd.DataFrame:
    """
    Compute pre- and post-match Elo ratings for teams, incorporating margin-of-victory and home advantage.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing match records, expected to include at least the following columns:
        ``match_date``, ``home_team``, ``away_team``, ``result``, ``home_margin``, and ``total_home_advantage``.
        Must be sortable by ``match_date`` and may include past match results for Elo tracking.
    initial_rating : float, optional
        Starting Elo rating assigned to all teams not previously encountered (default is 1500).
    k_factor : float, optional
        Scaling factor for Elo updates (default is 30), typically controls the weight of each match's outcome.

    Returns
    -------
    pandas.DataFrame
        A copy of the input ``dataframe`` with four additional columns:
        ``home_elo_pre_match``, ``away_elo_pre_match``, ``home_elo_post_match``, ``away_elo_post_match``.
        Pre-match Elo values correspond to the team ratings before the current match,
        and post-match ratings reflect updates after applying the Elo formula adjusted
        for margin-of-victory and home advantage.

    Notes
    -----
    - Matches with missing or falsy ``result`` entries are skipped and carry forward previous Elo ratings.
    - Margin-of-victory adjustments are calculated via ``margin_of_victory_multiplier``,
      which affects Elo updates proportionally to the observed margin.
    - ``total_home_advantage`` is incorporated into the expected probability calculation using
      ``calculate_expected_probability``.

    """
    df = dataframe.sort_values('match_date').copy()
    df['total_home_advantage'] = df['total_home_advantage'].fillna(0)

    teams = pd.unique(df[['home_team', 'away_team']].values.ravel())
    elo = {team: initial_rating for team in teams}

    output = df.copy()
    for col in ['home_elo_pre_match', 'away_elo_pre_match',
                'home_elo_post_match', 'away_elo_post_match']:
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


def add_team_elo_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average Elo rating change for each team over recent matches.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that includes, at minimum, the following columns:
        ``match_date``, ``home_team``, ``away_team``, ``home_elo_pre_match``,
        and ``away_elo_pre_match``. Must contain past match records for Elo
        computation.

    Returns
    -------
    pandas.DataFrame
        Original DataFrame augmented with six new columns:
        ``home_avg_elo_change_last_match``,
        ``home_avg_elo_change_last_5_matches``,
        ``home_avg_elo_change_last_10_matches``,
        ``away_avg_elo_change_last_match``,
        ``away_avg_elo_change_last_5_matches``,
        ``away_avg_elo_change_last_10_matches``.
        These represent the rolling average Elo rating changes for each team,
        based on their specified number of most recent matches (excluding the
        current match).

    Notes
    -----
    - Elo change is computed as the difference in a team's pre-match Elo
      between consecutive matches (across home and away appearances).
    - Rolling averages are calculated using window sizes of 1, 5, and 10,
      with a minimum of one value required (``min_periods=1``).
    - The function aligns and merges the computed rolling averages back into
      the original DataFrame separately for home and away team contexts.

    """
    # Prepare two DataFrames: one for home teams, one for away teams
    home_df = df[['match_date', 'home_team', 'home_elo_pre_match']].copy()
    home_df.columns = ['match_date', 'team', 'elo']

    away_df = df[['match_date', 'away_team', 'away_elo_pre_match']].copy()
    away_df.columns = ['match_date', 'team', 'elo']

    # Combine them into one DataFrame with team-level records
    team_matches = pd.concat([home_df, away_df], ignore_index=True).sort_values(
        by=['team', 'match_date']
    )

    # Compute Elo change per team between matches
    team_matches['elo_change'] = team_matches.groupby('team')['elo'].diff()

    # Compute rolling averages: last 1, 5, and 10 matches
    team_matches['avg_elo_change_last_match'] = team_matches.groupby('team')['elo_change'].transform(
        lambda x: x.rolling(window=1, min_periods=1).mean()
    )
    team_matches['avg_elo_change_last_5_matches'] = team_matches.groupby('team')['elo_change'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    team_matches['avg_elo_change_last_10_matches'] = team_matches.groupby('team')['elo_change'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )

    # Merge rolling averages back into original df for home teams
    home_avg = team_matches[['match_date', 'team', 'avg_elo_change_last_match',
                             'avg_elo_change_last_5_matches',
                             'avg_elo_change_last_10_matches']].copy()
    home_avg.columns = ['match_date', 'home_team',
                        'home_avg_elo_change_last_match',
                        'home_avg_elo_change_last_5_matches',
                        'home_avg_elo_change_last_10_matches']
    df = df.merge(home_avg, on=['match_date', 'home_team'], how='left')

    # Merge rolling averages back for away teams
    away_avg = team_matches[['match_date', 'team', 'avg_elo_change_last_match',
                             'avg_elo_change_last_5_matches',
                             'avg_elo_change_last_10_matches']].copy()
    away_avg.columns = ['match_date', 'away_team',
                        'away_avg_elo_change_last_match',
                        'away_avg_elo_change_last_5_matches',
                        'away_avg_elo_change_last_10_matches']
    df = df.merge(away_avg, on=['match_date', 'away_team'], how='left')

    return df


def create_elo_and_score_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the differences in Elo ratings and average score margins between home and away teams.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing match data with the following columns:
        - ``home_elo_pre_match``: Pre-match Elo rating for the home team.
        - ``away_elo_pre_match``: Pre-match Elo rating for the away team.
        - ``home_avg_elo_change_last_match``: Average Elo change for the home team in their last match.
        - ``away_avg_elo_change_last_match``: Average Elo change for the away team in their last match.
        - ``home_avg_elo_change_last_5_matches``: Average Elo change for the home team in their last 5 matches.
        - ``away_avg_elo_change_last_5_matches``: Average Elo change for the away team in their last 5 matches.
        - ``home_avg_elo_change_last_10_matches``: Average Elo change for the home team in their last 10 matches.
        - ``away_avg_elo_change_last_10_matches``: Average Elo change for the away team in their last 10 matches.
        - ``home_avg_score_diff_last_1``: Average score difference for the home team in their last match.
        - ``away_avg_score_diff_last_1``: Average score difference for the away team in their last match.
        - ``home_avg_score_diff_last_5``: Average score difference for the home team in their last 5 matches.
        - ``away_avg_score_diff_last_5``: Average score difference for the away team in their last 5 matches.
        - ``home_avg_score_diff_last_10``: Average score difference for the home team in their last 10 matches.
        - ``away_avg_score_diff_last_10``: Average score difference for the away team in their last 10 matches.

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with the following additional columns:
        - ``pre_match_elo_difference``: Difference in Elo ratings between home and away teams before the match.
        - ``elo_avg_change_difference_last_match``: Difference in average Elo change between home and away teams in their last match.
        - ``elo_avg_change_difference_last_5_matches``: Difference in average Elo change between home and away teams in their last 5 matches.
        - ``elo_avg_change_difference_last_10_matches``: Difference in average Elo change between home and away teams in their last 10 matches.
        - ``last_avg_match_scores_difference``: Difference in average score margins between home and away teams in their last match.
        - ``last_avg_5_match_scores_difference``: Difference in average score margins between home and away teams in their last 5 matches.
        - ``last_avg_10_match_scores_difference``: Difference in average score margins between home and away teams in their last 10 matches.

    Notes
    -----
    - The Elo rating differences are calculated as:
      ``pre_match_elo_difference = home_elo_pre_match - away_elo_pre_match``.
    - The average Elo change differences are calculated for the last 1, 5, and 10 matches.
    - The score margin differences are calculated for the last 1, 5, and 10 matches.
    - These differences provide insights into team performance disparities before the match.
    """
    df["pre_match_elo_difference"] = df["home_elo_pre_match"] - df["away_elo_pre_match"]
    df["elo_avg_change_difference_last_match"] = df["home_avg_elo_change_last_match"] - df["away_avg_elo_change_last_match"]
    df["elo_avg_change_difference_last_5_matches"] = df["home_avg_elo_change_last_5_matches"] - df["away_avg_elo_change_last_5_matches"]
    df["elo_avg_change_difference_last_10_matches"] = df["home_avg_elo_change_last_10_matches"] - df["away_avg_elo_change_last_10_matches"]
    df["last_avg_match_scores_difference"] = df["home_avg_score_diff_last_1"] - df["away_avg_score_diff_last_1"]
    df["last_avg_5_match_scores_difference"] = df["home_avg_score_diff_last_5"] - df["away_avg_score_diff_last_5"]
    df["last_avg_10_match_scores_difference"] = df["home_avg_score_diff_last_10"] - df["away_avg_score_diff_last_10"]

    return df


def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate basic match-related features for the provided DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing match data with columns such as:
        - 'home_team_odds', 'draw_team_odds', 'away_team_odds'
        - 'home_prob_BE', 'draw_prob_BE', 'away_prob_BE'
        - 'home_prob', 'draw_prob', 'away_prob'
        - 'home_team', 'away_team', 'match_date', etc.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame with additional columns representing various
        match-related features, including but not limited to:
        - Match IDs
        - Team IDs
        - Home advantage metrics
        - Team average score differences
        - Team win streaks
        - Fixture-specific score differences

    Notes
    -----
    This function performs the following operations:
    - Drops columns related to betting odds and probabilities.
    - Generates match and team identifiers.
    - Computes home advantage metrics.
    - Calculates team average score differences.
    - Determines team win streaks.
    - Computes fixture-specific score differences.

    The resulting DataFrame can be used for further analysis or modeling tasks.
    """
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


def create_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Elo rating-based features for the provided DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing match data with columns such as:
        - 'home_team', 'away_team'
        - 'home_elo_pre_match', 'away_elo_pre_match'
        - 'home_avg_elo_change_last_match', 'away_avg_elo_change_last_match'
        - 'home_avg_elo_change_last_5_matches', 'away_avg_elo_change_last_5_matches'
        - 'home_avg_elo_change_last_10_matches', 'away_avg_elo_change_last_10_matches'
        - 'home_avg_score_diff_last_1', 'away_avg_score_diff_last_1'
        - 'home_avg_score_diff_last_5', 'away_avg_score_diff_last_5'
        - 'home_avg_score_diff_last_10', 'away_avg_score_diff_last_10'
        - 'match_date', etc.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame with additional columns representing various
        Elo rating-based features, including but not limited to:
        - Pre- and post-match Elo ratings for home and away teams
        - Average Elo rating changes over the last 1, 5, and 10 matches
        - Differences in Elo ratings and average score margins between home and away teams

    Notes
    -----
    This function performs the following operations:
    - Computes pre- and post-match Elo ratings for both home and away teams.
    - Calculates average Elo rating changes over the last 1, 5, and 10 matches.
    - Computes differences in Elo ratings and average score margins between home and away teams.

    The resulting DataFrame can be used for further analysis or modeling tasks.
    """
    df = compute_pre_and_post_match_elo(df)
    df = add_team_elo_changes(df)
    df = create_elo_and_score_difference(df)

    return df


def add_team_avg_score_diff(df):
    """
    Calculate average score difference (margin) for each team over their last 1, 5, and 10 matches,
    excluding the current match (i.e., only historical matches).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing match data with columns:
        - 'match_date' : datetime-like, date of the match
        - 'home_team' : str, identifier of the home team
        - 'away_team' : str, identifier of the away team
        - 'home_margin' : numeric, score difference for the home team (home score - away score)
        - 'away_margin' : numeric, score difference for the away team (away score - home score)

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with added columns:
        - 'home_avg_score_diff_last_1', 'home_avg_score_diff_last_5', 'home_avg_score_diff_last_10':
          rolling averages of home team score differences over last 1, 5, and 10 matches.
        - 'away_avg_score_diff_last_1', 'away_avg_score_diff_last_5', 'away_avg_score_diff_last_10':
          rolling averages of away team score differences over last 1, 5, and 10 matches.

    Notes
    -----
    - Rolling averages exclude the current match by shifting the score difference series.
    - Both home and away performances are combined for calculating rolling averages per team.
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
    Add home and away win streak counts to the DataFrame for each match.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing match data with columns:
        - 'match_date' : datetime-like, date of the match
        - 'home_team' : str, identifier of the home team
        - 'away_team' : str, identifier of the away team
        - 'winner' : str, outcome of the match ('Home', 'Away', or 'Draw')

    Returns
    -------
    pandas.DataFrame
        The input DataFrame augmented with two new columns:
        - 'home_win_streak': int, consecutive wins of the home team prior to the match
        - 'away_win_streak': int, consecutive wins of the away team prior to the match

    Notes
    -----
    - Win streaks are calculated sequentially by match date.
    - Streak resets to zero if a team loses or draws.
    - The win streaks represent the count *before* the current match (historical).
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
    Add average score difference over the last 3 matches between the two teams from the home team's perspective.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing match data with columns:
        - 'match_date' : datetime-like, date of the match
        - 'home_team' : str, identifier of the home team
        - 'away_team' : str, identifier of the away team
        - 'home_scores' : int or float, score of the home team in the match
        - 'away_scores' : int or float, score of the away team in the match

    Returns
    -------
    pandas.DataFrame
        The input DataFrame augmented with one new column:
        - 'avg_score_diff_last_3_between_teams': float, average score difference over the last 3 matches
          between the two teams from the home team's perspective

    Notes
    -----
    - Matches are processed in chronological order.
    - Score differences are adjusted so that the perspective is always from the home team in the current match.
    - If fewer than 3 previous matches exist between the teams, averages are calculated over the available matches.
    - If no prior matches exist, the average difference is set to 0.0.
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
    Add the average score difference over the last 3 games for the same fixture
    (same home team vs same away team), excluding the current match.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing match data with columns:
        - 'match_date' : datetime-like, date of the match
        - 'home_team' : str, identifier of the home team
        - 'away_team' : str, identifier of the away team
        - 'home_scores' : numeric, score of the home team in the match
        - 'away_scores' : numeric, score of the away team in the match

    Returns
    -------
    pandas.DataFrame
        The input DataFrame augmented with one new column:
        - 'avg_score_diff_last_3_same_fixture': float, average score difference over the last 3 matches
          between the two teams from the home team's perspective, excluding the current match.

    Notes
    -----
    - Matches are processed in chronological order.
    - Score differences are calculated as `home_scores - away_scores`.
    - The perspective is always from the home team's side; negative values indicate away team dominance.
    - If fewer than 3 previous matches exist between the teams, averages are calculated over the available matches.
    - If no prior matches exist, the average difference is set to 0.0.
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
    """
    Add the score difference from the most recent match between the same home and away teams.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing match data with columns:
        - 'match_date' : datetime-like, date of the match
        - 'home_team' : str, identifier of the home team
        - 'away_team' : str, identifier of the away team
        - 'home_scores' : numeric, score of the home team in the match
        - 'away_scores' : numeric, score of the away team in the match

    Returns
    -------
    pandas.DataFrame
        The input DataFrame augmented with one new column:
        - 'score_diff_last_fixture': float, score difference from the most recent match
          between the same home and away teams, excluding the current match.

    Notes
    -----
    - Matches are processed in chronological order.
    - Score differences are calculated as `home_scores - away_scores`.
    - The perspective is always from the home team's side; negative values indicate away team dominance.
    - If no prior match exists between the teams, the score difference is set to 0.0.
    """
    df = df.sort_values(by='match_date').copy()
    df['score_diff_last_fixture'] = 0.0  # Default to 0.0

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


def drop_first_games(df):
    """
    Remove matches where either team is playing their first game.

    This function excludes matches in which either the home team or the away team
    is participating in their very first recorded match in the dataset. It preserves
    all original columns and adds a unique 'observation_id' based on the filtered DataFrame's index.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing match data with at least the following columns:
        - 'match_date' : datetime-like or string convertible to datetime
        - 'home_team' : str, identifier of the home team
        - 'away_team' : str, identifier of the away team

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame excluding matches where either team is in their first game,
        with a new 'observation_id' column unique to each row.

    Notes
    -----
    - The function assumes 'match_date' can be converted to pandas datetime.
    - The 'observation_id' is set to the filtered DataFrame's index after filtering.
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
