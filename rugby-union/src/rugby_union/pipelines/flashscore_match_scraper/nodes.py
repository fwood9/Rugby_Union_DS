"""
This is a boilerplate pipeline 'flashscore_match_scraper'
generated using Kedro 0.19.14
"""


from bs4 import BeautifulSoup
from fractions import Fraction
import numpy as np
import pandas as pd
import re
import logging
import time

from itertools import product
from selenium import webdriver
from selenium.common.exceptions import ElementClickInterceptedException, StaleElementReferenceException
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    StaleElementReferenceException,
    TimeoutException,
    NoSuchElementException
)
from selenium_stealth import stealth
from urllib.parse import urlparse
from webdriver_manager.chrome import ChromeDriverManager


def generate_flashscore_urls(sport: list, competition: list, years: list, countries:list) -> list:
    """
       Helper function to generate flashscore.co.uk result page URLs for specified sports, competitions,
       years, and countries.

       This function accepts single values or iterables for each parameter—`sport`,
       `competition`, `years`, and `countries`—and constructs all possible combinations
       of those inputs to produce corresponding Flashscore result page URLs.

       Args:
           sport (str or list/tuple of str):
               The sport or list of sports (e.g., "football", ["football", "tennis"]).
           competition (str or list/tuple of str):
               The competition name(s) (e.g., "premier-league").
           years (int, str, or list/tuple of int/str):
               The year or years to include in the URL (e.g., 2024, ["2023", "2024"]).
           countries (str, list/tuple of str, or None):
               Country slug(s), such as "england" or "spain"; if None or empty, the
               country component is omitted from the URL.

       Returns:
           list of str:
               A list of fully constructed URLs that follow one of these formats:
               - With country: "https://www.flashscore.co.uk/{sport}/{country}/{competition}-{year}/results/"
               - Without country: "https://www.flashscore.co.uk/{competition}-{year}/results/"

       Examples:
            generate_flashscore_urls("football", "championship", 2024, "england")
           ["https://www.flashscore.co.uk/football/england/championship-2024/results/"]

            generate_flashscore_urls(
           ...     ["football", "tennis"], ["grand-slam"], [2023, 2024], None
           ... )
           [
               "https://www.flashscore.co.uk/grand-slam-2023/results/",
               "https://www.flashscore.co.uk/grand-slam-2024/results/",
               "https://www.flashscore.co.uk/tennis/grand-slam-2023/results/",
               "https://www.flashscore.co.uk/tennis/grand-slam-2024/results/"
           ]
       """
    base = "https://www.flashscore.co.uk"  # Root URL for all generated links
    urls = []  # Container to collect each constructed URL

    # Normalize inputs to lists so iteration can uniformly handle single or multiple values
    sports = sport if isinstance(sport, (list, tuple)) else [sport]
    competitions = competition if isinstance(competition, (list, tuple)) else [competition]
    years = years if isinstance(years, (list, tuple)) else [years]
    # Handle None or empty for countries: produce one iteration with country omitted
    countries = countries if isinstance(countries, (list, tuple)) else ([countries] if countries else [None])

    # Iterate through all combinations via cartesian product
    for sport, competition, year, country in product(sports, competitions, years, countries):
        comp_year = competition + (f"-{year}" if year else "")  # Append year if provided

        # Construct URL including country when specified; otherwise omit that segment
        if country:
            url = f"{base}/{sport}/{country}/{comp_year}/results/"
        else:
            url = f"{base}/{comp_year}/results/"
        urls.append(url)  # Add to list of URLs

    return urls  # Return all constructed endpoints

def make_driver() -> webdriver.Chrome:
    """
    Creates and returns a Selenium Chrome WebDriver configured for stealthy headless operation.

    Configuration Details:
    - Runs Chrome in headless mode (`--headless`) with GPU disabled (`--disable-gpu`) and sandbox bypassed (`--no-sandbox`),
      ensuring compatibility and performance in environments like Linux servers or Docker
    - Disables Selenium automation flags by excluding "enable-automation" and removing automation extensions via Chrome options,
      reducing detection by anti-bot systems
    - Suppresses image loading by setting the Chrome preference `"profile.managed_default_content_settings.images": 2`,
      improving speed and bandwidth efficiency.
    - Sets the `pageLoadStrategy` capability to `"eager"` so the driver returns control when the DOM is fully loaded,
      but before other resources (like images, stylesheets) finish loading, making navigation faster
    - Automatically installs and manages the ChromeDriver executable via `ChromeDriverManager`, simplifying setup.
    - Applies Selenium‑Stealth settings:
        • Overrides browser fingerprint details like `languages`, `vendor`, `platform`, WebGL `vendor`, and `renderer`.
        • Enables `fix_hairline` to further mask headless rendering artifacts.
      These tweaks help mimic a real user browser and evade common bot-detection heuristics

    Returns:
        webdriver.Chrome: A configured, stealth-enabled headless Chrome WebDriver ready for web automation tasks.
    """

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run browser headlessly
    options.add_argument("--disable-gpu")  # Disable GPU rendering to avoid issues in headless mode
    options.add_argument("--no-sandbox")  # Required for some CI/container environments
    options.add_experimental_option("excludeSwitches", ["enable-automation"])  # Hide automation flags
    options.add_experimental_option("useAutomationExtension", False)  # Disable Selenium extension
    options.add_experimental_option(
        "prefs",
        {"profile.managed_default_content_settings.images": 2}  # Disable images to speed up loading
    )
    options.set_capability("pageLoadStrategy", "eager")  # Return once DOM is ready, before full resource load

    service = Service(ChromeDriverManager().install())  # Auto-download and manage ChromeDriver
    driver = webdriver.Chrome(service=service, options=options)

    stealth(driver,  # Apply stealth fingerprint adjustments to evade detection
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True
            )
    return driver  # Return the configured WebDriver


def clean_match_historic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize historical match-level data for analysis.

    Processes a DataFrame by parsing dates, normalizing competition names,
    extracting score components, and determining match outcomes.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least the following columns:
        - `match_date` : str
            Combined date and time, format "DD.MM.YYYY HH:MM".
        - `competition` : str
            Competition name, potentially with trailing digits or hyphens.
        - `result` : str or NaN
            Match result strings, expected to contain scores (e.g., "3-2").
        - `home_team` : str
            Name of the home team.
        - `away_team` : str
            Name of the away team.

    Returns
    -------
    pandas.DataFrame
        Original DataFrame enhanced with added and modified columns:

        - `date` : datetime.date
            Extracted date from `match_date`.
        - `time` : datetime.time
            Extracted time from `match_date`.
        - `year` : int
            Year component of the match.
        - `competition` : str
            Cleaned competition name with trailing digits/hyphens removed.
        - `home_scores` : float
            Home team score, extracted from `result`.
        - `away_scores` : float
            Away team score, extracted from `result`.
        - `winner` : {'Home', 'Away', 'Draw'} or NaN
            Outcome of the match: 'Home' if home team won,
            'Away' if away team won, 'Draw' if tied, or NaN if indeterminate.

    Notes
    -----
    - Uses `pd.to_datetime` with `errors="coerce"` to safely parse dates.
    - Scores are cast to floats for compatibility with other numeric fields.
    - Outcome determination is handled via a row-wise helper function.

    Examples
    --------
     df = pd.DataFrame({
    ...     'match_date': ['01.08.2025 18:30'],
    ...     'competition': ['PremierLeague-123'],
    ...     'result': ['2-1'],
    ...     'home_team': ['TeamA'],
    ...     'away_team': ['TeamB']
    ... })
    clean_df = clean_match_historic(df)
    clean_df[['date', 'time', 'year', 'competition', 'home_scores', 'away_scores', 'winner']]
          date      time  year    competition  home_scores  away_scores winner
    0 2025-08-01 18:30:00  2025  PremierLeague           2.0          1.0   Home
    """
    # Parse the combined date-time string into datetime, ignoring errors
    df["match_dt"] = pd.to_datetime(
        df["match_date"],
        format="%d.%m.%Y %H:%M",
        dayfirst=True,
        errors="coerce"
    )
    # Split datetime into separate `date` and `time` columns
    df["date"] = df["match_dt"].dt.date
    df["time"] = df["match_dt"].dt.time
    df["year"] = df["match_dt"].dt.year

    # Clean `competition`: remove trailing digits or hyphens
    # First strip digits, then trailing hyphens or digits
    df["competition"] = df["competition"].apply(
        lambda x: re.sub(r"(\d+|-)$", "", re.sub(r"\d+", "", x))
    )

    # Standardize format of results as 'X - Y'
    df["result"] = (
        df["result"]
        .str.extract(r'(\d+\s*-\s*\d+)', expand=False)
        .str.replace(r'\s*-\s*', ' - ', regex=True)
    )
    # Drop no longer needed date columns
    df.drop(["match_date", "match_dt"], axis=1, inplace=True)

    # Extract numeric home and away scores, casting to float
    df[['home_scores', 'away_scores']] = (
        df['result']
        .str.extract(r'(\d+)\s*-\s*(\d+)', expand=True)
        .astype(float)
    )

    # Helper to label the winner of the match
    def label_winner(row):
        if pd.isna(row['home_scores']) or pd.isna(row['away_scores']):
            return np.nan
        if row['home_scores'] > row['away_scores']:
            return 'Home'
        elif row['home_scores'] < row['away_scores']:
            return 'Away'
        else:
            return 'Draw'

    # Apply the winner labelling function row-wise
    df['winner'] = df.apply(label_winner, axis=1)

    return df

def clean_match_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize weekly match-level data for analysis.

    Parses date and time, normalizes competition names, ensures unique matches,
    extracts scores, and determines match outcomes.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least these columns:
        - `match_date` : str
            Combined date and time, format "DD.MM.YYYY HH:MM".
        - `competition` : str
            Competition field, typically with a leading label and colon.
        - `match_link` : str
            Unique identifier or URL for each match.
        - `result` : str or NaN
            Match result, expected like "2-1" to infer winner.
        - `home_team` : str
            Name of home team.
        - `away_team` : str
            Name of away team.

    Returns
    -------
    pandas.DataFrame
        Enhanced DataFrame with additional columns:
        - `date` : datetime.date
            Extracted date of the match.
        - `time` : datetime.time
            Extracted time of day the match was played.
        - `year` : int
            Year of the match.
        - `competition` : str
            Cleaned competition name (lowercased, hyphenated).
        - `home_scores` : float
            Numeric home team score.
        - `away_scores` : float
            Numeric away team score.
        - `winner` : {'Home', 'Away', 'Draw'} or NaN
            Match outcome: 'Home' if home wins, 'Away' if away wins,
            'Draw' if tied, or NaN if no valid score.

    Notes
    -----
    - Uses `pd.to_datetime` with `errors='coerce'` to handle invalid dates.
    - Ensures that each `match_link` corresponds to only one row by dropping
      duplicates.
    - Scores are extracted via a regex and cast to floats to allow mixing with
      other numeric data.

    Examples
    --------
    df = pd.DataFrame({
    ...     'match_date': ['15.08.2025 20:00', '15.08.2025 20:00'],
    ...     'competition': ['Round: Premier League', 'Round: Premier League'],
    ...     'match_link': ['link1', 'link1'],
    ...     'result': ['3-2', None],
    ...     'home_team': ['TeamA', 'TeamA'],
    ...     'away_team': ['TeamB', 'TeamB']
    ... })
    cleaned = clean_match_weekly(df)
    cleaned[['date', 'time', 'year', 'competition', 'home_scores', 'away_scores', 'winner']]
          date      time  year        competition  home_scores  away_scores winner
    0 2025-08-15 20:00:00  2025  premier-league          3.0          2.0   Home
    """
    # Parse 'match_date' into datetime; invalid parsing yields NaT
    df["match_dt"] = pd.to_datetime(
        df["match_date"],
        format="%d.%m.%Y %H:%M",
        dayfirst=True,
        errors="coerce"
    )

    # Split datetime into `date`, `time`, and `year` components
    df["date"] = df["match_dt"].dt.date
    df["time"] = df["match_dt"].dt.time
    df["year"] = df["match_dt"].dt.year

    # Simplify and normalize `competition`:
    # Take string after colon, convert to lowercase, replace spaces with hyphens
    df["competition"] = (
        df["competition"]
        .apply(lambda x: x.split(":\xa0")[-1] if isinstance(x, str) and ":\xa0" in x else x)
        .str.lower()
        .str.replace(" ", "-", regex=False)
    )

    # Ensure `match_link` is a string, then drop duplicate matches (keep last)
    df["match_link"] = df["match_link"].astype(str)
    df = df.drop_duplicates(subset=["match_link"], keep="last")

    # Standardize result format to "X - Y"
    df["result"] = (
        df["result"]
        .str.extract(r"(\d+\s*-\s*\d+)", expand=False)
        .str.replace(r"\s*-\s*", " - ", regex=True)
    )

    # Drop intermediate date columns now that parsing is complete
    df.drop(["match_date", "match_dt"], axis=1, inplace=True)

    # Extract `home_scores` and `away_scores` from `result`; convert to float
    df[["home_scores", "away_scores"]] = (
        df["result"]
        .str.extract(r"(\d+)\s*-\s*(\d+)", expand=True)
        .astype(float)
    )

    # Helper: assign winner label based on scores
    def label_winner(row):
        if pd.isna(row["home_scores"]) or pd.isna(row["away_scores"]):
            return np.nan
        if row["home_scores"] > row["away_scores"]:
            return "Home"
        elif row["home_scores"] < row["away_scores"]:
            return "Away"
        else:
            return "Draw"

    # Apply winner determination across rows
    df["winner"] = df.apply(label_winner, axis=1)

    return df

def mean_odds_calc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average decimal odds per match from various bookmakers.

    Converts fractional or decimal odds into a decimal format and then
    averages them per match.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the following columns:
        - `odds_1`, `odds_draw`, `odds_2` : str or numeric
            Odds values in fractional (e.g., "5/2"), decimal (e.g., "3.5"),
            or placeholder string "-" for missing.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per `match_link`, containing:
        - `home_team_odds`, `draw_team_odds`, `away_team_odds` : float
            Mean decimal odds for each outcome.

    Notes
    -----
    - Fractional odds strings are parsed using `Fraction` and converted
      to decimal (plus 1).
    - Placeholder "-" values are treated as missing and converted to NaN.
    - The mean is computed across bookmakers grouped via `match_link`.

    Examples
    --------
    df = pd.DataFrame({
    ...     'match_link': ['m1', 'm1', 'm2'],
    ...     'odds_1': ['2/1', '3/1', '4.0'],
    ...     'odds_draw': ['3/1', '-', '2.5'],
    ...     'odds_2': ['4/1', '5/1', '3.0']
    ... })
    mean_odds_calc(df)
      match_link  home_team_odds  draw_team_odds  away_team_odds
    0         m1             2.50             1.50             5.00
    1         m2             5.00             2.50             3.00
    """
    # Convert odds fields to decimal format
    for col in ["odds_1", "odds_draw", "odds_2"]:
        df[col + "_dec"] = (
            df[col]
            .replace("-", np.nan)
            .apply(
                lambda x: float(Fraction(x)) + 1
                if pd.notnull(x) and '/' in str(x)
                else (float(x) if pd.notnull(x) else np.nan)
            )
        )

    # Group by match and compute average decimal odds
    m = df.groupby("match_link", as_index=False)[
        ["odds_1_dec", "odds_draw_dec", "odds_2_dec"]
    ].mean()

    # Rename columns for clarity
    return m.rename(columns={
        "odds_1_dec": "home_team_odds",
        "odds_draw_dec": "draw_team_odds",
        "odds_2_dec": "away_team_odds"
    })


def merge_and_calculate(match_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge match and odds data, compute implied probabilities of outcomes.

    Combines match-level and odds data, ensures odds fields exist,
    then calculates implied probabilities with and without bookmaker edge.

    Parameters
    ----------
    match_df : pandas.DataFrame
        Match-level DataFrame containing at least a `match_link` column.
    odds_df : pandas.DataFrame
        DataFrame with precomputed odds per match:
        - `home_team_odds`, `draw_team_odds`, `away_team_odds`.

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame including:
        - Numeric odds columns.
        - Implied probabilities (`*_prob_BE`) based on 100 / odds.
        - Normalized probabilities (`*_prob`) summing to 100 across outcomes.

    Notes
    -----
    - Missing `draw_team_odds` is initialized as NaN if absent.
    - Implied probabilities before normalization represent bookmaker edge (BE).
    - Final probabilities rescaled so that sum across outcomes equals 100.

    Examples
    --------
    match_df = pd.DataFrame({'match_link': ['m1']})
    odds_df = pd.DataFrame({
    ...     'match_link': ['m1'],
    ...     'home_team_odds': [2.0],
    ...     'away_team_odds': [3.0]
    ... })
    merge_and_calculate(match_df, odds_df)
      match_link  home_team_odds  draw_team_odds  away_team_odds  home_prob_BE  draw_prob_BE  away_prob_BE  home_prob  draw_prob  away_prob
    0         m1              2.0             NaN              3.0          50.0           NaN          33.33       60.6        0.0        33.4
    """
    # Merge match and odds data on match_link
    df = pd.merge(match_df, odds_df, on="match_link", how="inner").copy()

    # Ensure draw odds column exists
    if "draw_team_odds" not in df.columns:
        df["draw_team_odds"] = np.nan

    # Convert odds to numeric, coercing invalid values to NaN
    for side in ["home", "draw", "away"]:
        col = f"{side}_team_odds"
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute implied probabilities before edge (bookmaker margin)
    for side in ["home", "draw", "away"]:
        col = f"{side}_team_odds"
        df[f"{side}_prob_BE"] = (1 / df[col] * 100).round(2)

    # Normalize so probabilities sum exactly to 100
    total_BE = df["home_prob_BE"] + df["draw_prob_BE"] + df["away_prob_BE"]
    norm_factor = 100 / total_BE
    for side in ["home", "draw", "away"]:
        df[f"{side}_prob"] = (df[f"{side}_prob_BE"] * norm_factor).round(2)

    return df

def scrape_all(urls: list, driver) -> (pd.DataFrame, pd.DataFrame):
    """
    Scrape historical match and odds data from multiple web pages.

    Iterates through a list of URLs to fetch match links, retrieves match-level
    details, and extracts bookmaker odds where available.

    Parameters
    ----------
    urls : list of str
        URLs to scrape for match summaries.
    driver : selenium.webdriver
        Selenium WebDriver used to load and interact with pages.

    Returns
    -------
    matches_df : pandas.DataFrame
        DataFrame containing match-level records including
        `match_link`, `competition`, `match_date`, `home_team`,
        `away_team`, and `result`.
    odds_df : pandas.DataFrame
        DataFrame containing bookmaker odds per match with columns:
        `match_link`, `bookmaker`, `odds_1`, `odds_draw`, `odds_2`.

    Notes
    -----
    - Uses Selenium to navigate pages and BeautifulSoup for HTML parsing.
    - Includes error handling for timeouts and Selenium exceptions.
    - Attempts to click to expand odds sections when necessary.
    - Stops scraping a URL or match when loading fails, continuing gracefully.

    Examples
    --------
    urls = ['https://example.com/competition/round1/']
    from selenium import webdriver
    driver = webdriver.Chrome()
    matches, odds = scrape_all(urls, driver)
    matches.head()
      match_link competition       match_date home_team away_team result
    0    ...         comp1           01.01.2025  TeamA    TeamB     2‑1
    odds.head()
      match_link bookmaker odds_1 odds_draw odds_2
    0    ...       Bookie1   2.5     3.0       2.8
    """
    records = []
    odds_records = []
    wait = WebDriverWait(driver, 10)

    for url in urls:
        try:
            driver.get(url)
            # Wait until match-summary links are present
            wait.until(EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "a[href$='match-summary']")))
        except (TimeoutException, WebDriverException):
            continue  # Skip this URL if loading fails

        # Parse out competition identifier from URL
        soup = BeautifulSoup(driver.page_source, "html.parser")
        competition = urlparse(url).path.strip("/").split("/")[-2]
        match_links = {a["href"] for a in soup.select("a[href$='match-summary']")}

        for link in match_links:
            try:
                driver.get(link)
                # Wait for match details to appear
                wait.until(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, ".fixedHeaderDuel__detailStatus")))
            except (TimeoutException, WebDriverException):
                continue  # Skip this match if loading fails

            page = BeautifulSoup(driver.page_source, "html.parser")
            status = page.select_one(".duelParticipant__score")
            teams = page.find_all("a", class_=re.compile(".*participantName.*"))
            home_name = teams[0].text.strip() if len(teams) >= 2 else None
            away_name = teams[1].text.strip() if len(teams) >= 2 else None
            date_el = page.select_one(".duelParticipant__startTime")

            # Save match-level data
            records.append({
                "match_link": link,
                "competition": competition,
                "match_date": date_el.text.strip() if date_el else None,
                "home_team": home_name,
                "away_team": away_name,
                "result": status.text.strip() if status else None,
            })

            # Attempt to expand odds section if needed
            try:
                driver.find_element(By.CSS_SELECTOR, ".wclOddsHeader").click()
            except Exception:
                pass  # Odds may already be visible or unavailable

            # Wait for odds to load
            try:
                wait.until(EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, ".wclOddsContent .odds")))
            except TimeoutException:
                continue  # Skip if odds not loaded

            page = BeautifulSoup(driver.page_source, "html.parser")
            for odds in page.select(".wclOddsContent .odds"):
                bm = odds.select_one(".prematchLogo")
                vals = [v.text.strip() for v in odds.select("[data-testid='wcl-oddsValue']")]

                # Construct odds record
                odds_records.append({
                    "match_link": link,
                    "bookmaker": bm["title"] if bm and bm.has_attr("title") else None,
                    "odds_1": vals[0] if len(vals) > 0 else None,
                    "odds_draw": vals[1] if len(vals) == 3 else None,
                    "odds_2": vals[2] if len(vals) == 3 else (vals[1] if len(vals) == 2 else None),
                })

    driver.quit()
    # Convert recorded lists to DataFrames
    return pd.DataFrame(records), pd.DataFrame(odds_records)


def weekly_scrape(flashscore_cricket_link: list, driver) -> (pd.DataFrame, pd.DataFrame):
    """
    Scrape match and odds data from Flashscore for the past week.

    Uses Selenium to navigate backwards through seven days of results on
    Flashscore, collecting match links and related information, then follows
    each link to retrieve match metadata and available betting odds.

    Parameters
    ----------
    flashscore_cricket_link : list of str
        List of URLs to Flashscore pages for cricket competitions.
    driver : selenium.webdriver
        Selenium WebDriver used for page navigation and interaction.

    Returns
    -------
    matches_df : pandas.DataFrame
        DataFrame with one row per match, containing:
        `match_link`, `competition`, `match_date`, `home_team`, `away_team`, `result`.
    odds_df : pandas.DataFrame
        DataFrame with bookmaker-specific odds per match including:
        `match_link`, `bookmaker`, `odds_1`, `odds_draw`, `odds_2`.

    Notes
    -----
    - Iterates backward through seven days using the 'Previous day' button.
    - Handles page dynamics and stale elements via retries.
    - Closes the Selenium driver after scraping to free resources.
    - Extracts odds only if the relevant section loads successfully.

    Examples
    --------
    urls = ['https://www.flashscore.com/cricket/competition/']
    from selenium import webdriver
    driver = webdriver.Chrome()
    matches, odds = weekly_scrape(urls, driver)
    matches.head()
      match_link competition match_date home_team away_team result
    0    /match/...       Comp1      01.01.2025  TeamA     TeamB      200-180
    odds.head()
      match_link bookmaker odds_1 odds_draw odds_2
    0    /match/...    Bookie1   1.5     2.0     2.5
    """
    wait = WebDriverWait(driver, 15)
    match_links_info = []  # Collect (competition, match_link) pairs

    for url in flashscore_cricket_link:
        driver.get(url)
        # Wait until the 'Previous day' button is clickable
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Previous day']")))

        # Navigate backward through the last seven days
        for _ in range(7):
            btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Previous day']")))
            driver.execute_script("arguments[0].scrollIntoView(true);", btn)
            try:
                btn.click()
            except ElementClickInterceptedException:
                # Fallback to JS click if needed
                driver.execute_script("arguments[0].click();", btn)
            time.sleep(0.5)  # Allow dynamic content to load

            # Retry locating match elements to avoid stale references
            for _ in range(5):
                try:
                    matches = driver.find_elements(By.CSS_SELECTOR, "div.event__match")
                    if matches:
                        break
                except StaleElementReferenceException:
                    time.sleep(0.5)
            else:
                matches = []  # Proceed if still unstable

            soup = BeautifulSoup(driver.page_source, "html.parser")
            matches = soup.find_all("div", class_="event__match")

            # Collect match links with associated competition names
            for match in matches:
                comp_div = match.find_previous("div", class_="event__titleBox")
                competition = comp_div.text.strip() if comp_div else None
                for a in match.select("a[href$='match-summary']"):
                    match_links_info.append({
                        "competition": competition,
                        "match_link": a["href"]
                    })

    records = []
    odds_records = []

    # Visit each match link to extract detailed info and odds
    for info in match_links_info:
        competition = info["competition"]
        link = info["match_link"]
        driver.get(link)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".fixedHeaderDuel__detailStatus")))
        page = BeautifulSoup(driver.page_source, "html.parser")

        # Extract match metadata
        status = page.select_one(".duelParticipant__score")
        teams = page.find_all("a", class_=re.compile(".*participantName.*"))
        home_name = teams[0].text.strip() if len(teams) >= 1 else None
        away_name = teams[1].text.strip() if len(teams) >= 2 else None
        date_el = page.select_one(".duelParticipant__startTime")

        records.append({
            "match_link": link,
            "competition": competition,
            "match_date": date_el.text.strip() if date_el else None,
            "home_team": home_name,
            "away_team": away_name,
            "result": status.text.strip() if status else None,
        })

        # Attempt to open odds section if available
        try:
            driver.find_element(By.CSS_SELECTOR, ".wclOddsHeader").click()
        except Exception:
            pass

        # Wait for and parse odds entries
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".wclOddsContent .odds")))
        page = BeautifulSoup(driver.page_source, "html.parser")
        for odds in page.select(".wclOddsContent .odds"):
            bm = odds.select_one(".prematchLogo")
            vals = [v.text.strip() for v in odds.select("[data-testid='wcl-oddsValue']")]

            odds_records.append({
                "match_link": link,
                "bookmaker": bm["title"] if bm and bm.has_attr("title") else None,
                "odds_1": vals[0] if len(vals) > 0 else None,
                "odds_draw": vals[1] if len(vals) == 3 else None,
                "odds_2": vals[2] if len(vals) == 3 else (vals[1] if len(vals) == 2 else None),
            })

    driver.quit()  # Clean up Selenium driver
    return pd.DataFrame(records), pd.DataFrame(odds_records)


def make_urls(sport: list, competition: list, years: list, countries: list):
    return generate_flashscore_urls(sport, competition, years, countries)

def create_driver():
    return make_driver()

def extract_raw_data_historic(urls: list, driver: webdriver.Chrome):
    return scrape_all(urls, driver)

def extract_raw_data_weekly(flashscore_rugby_link: list, driver: webdriver.Chrome):
    return weekly_scrape(flashscore_rugby_link, driver)

def calc_mean_odds(df):
    return mean_odds_calc(df)

def merge_dfs(matches, odds):
    return merge_and_calculate(matches, odds)