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
      Clean and standardise historical match data in a DataFrame.

      This function transforms raw match-level data to a more analysis-ready format by:
      - Parsing match dates into separate `date`, `time`, and `year` columns.
      - Normalizing the `competition` field by stripping trailing digits and hyphens.
      - Determining match outcomes by assigning a binary `winner` flag: 1 for home win, 0 for away win, NaN otherwise.

      Parameters
      ----------
      df : pandas.DataFrame
          Input DataFrame containing at least the following columns:
          - `match_date` (str): Combined date and time in format "DD.MM.YYYY HH:MM".
          - `competition` (str): Competition name, possibly with trailing digits or hyphens.
          - `result` (str or NaN): Match result containing home or away team name to infer the winner.
          - `home_team` (str): Name of the home team.
          - `away_team` (str): Name of the away team.

      Returns
      -------
      pandas.DataFrame
          The original DataFrame enhanced with the following columns:
          - `date` (datetime.date): Date component of the match.
          - `time` (datetime.time): Time component of the match.
          - `year` (int): Year when the match was played.
          - `competition` (str): Cleaned competition name.
          - `winner` (float): 1.0 if home team won, 0.0 if away team won, NaN for indeterminate.

    """
    df["match_dt"] = pd.to_datetime(df["match_date"], format="%d.%m.%Y %H:%M", dayfirst=True, errors="coerce")
    df["date"] = df["match_dt"].dt.date
    df["time"] = df["match_dt"].dt.time
    df["year"] = df["match_dt"].dt.year
    df["competition"] = df["competition"].apply(
        lambda x: re.sub(r"(\d+|-)$", "", re.sub(r"\d+", "", x))
    )
    df['result'] = df['result'] \
    .str.extract(r'(\d+\s*-\s*\d+)', expand=False) \
    .str.replace(r'\s*-\s*', ' - ', regex=True)
    df.drop(["match_date", "match_dt"], axis=1, inplace=True)

    df[['home_scores','away_scores']] = (
        df['result']
          .str.extract(r'(\d+)\s*-\s*(\d+)', expand=True)
          .astype(float)
    )
    def label_winner(row):
        if pd.isna(row['home_scores']) or pd.isna(row['away_scores']):
            return np.nan
        if row['home_scores'] > row['away_scores']:
            return 'Home'
        elif row['home_scores'] < row['away_scores']:
            return 'Away'
        else:
            return 'Draw'

    df['winner'] = df.apply(label_winner, axis=1)
    return df

def clean_match_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
      Clean and standardize weekly match data in a DataFrame.

      This function transforms raw match-level data to a more analysis-ready format by:
      - Parsing match dates into separate `date`, `time`, and `year` columns.
      - Normalizing the `competition` field by stripping trailing digits and hyphens.
      - Determining match outcomes by assigning a binary `winner` flag: 1 for home win, 0 for away win, NaN otherwise.

      Parameters
      ----------
      df : pandas.DataFrame
          Input DataFrame containing at least the following columns:
          - `match_date` (str): Combined date and time in format "DD.MM.YYYY HH:MM".
          - `competition` (str): Competition name, possibly with trailing digits or hyphens.
          - `result` (str or NaN): Match result containing home or away team name to infer the winner.
          - `home_team` (str): Name of the home team.
          - `away_team` (str): Name of the away team.

      Returns
      -------
      pandas.DataFrame
          The original DataFrame enhanced with the following columns:
          - `date` (datetime.date): Date component of the match.
          - `time` (datetime.time): Time component of the match.
          - `year` (int): Year when the match was played.
          - `competition` (str): Cleaned competition name.
          - `winner` (float): 1.0 if home team won, 0.0 if away team won, NaN for indeterminate.
    """
    df["match_dt"] = pd.to_datetime(
        df["match_date"], format="%d.%m.%Y %H:%M", dayfirst=True, errors="coerce"
    )
    df["date"] = df["match_dt"].dt.date
    df["time"] = df["match_dt"].dt.time
    df["year"] = df["match_dt"].dt.year
    # Simplify competition field: take after colon, lowercase, replace spaces with hyphens
    df["competition"] = df["competition"].apply(lambda x: x.split(":\xa0")[-1].lower().replace(" ", "-"))
    df['match_link'] = df['match_link'].astype(str)  # Ensure link is string for uniqueness checks
    df = df.drop_duplicates(subset=['match_link'], keep='first')  # Keep only first occurrence per match
    df['result'] = df['result'] \
    .str.extract(r'(\d+\s*-\s*\d+)', expand=False) \
    .str.replace(r'\s*-\s*', ' - ', regex=True)
    df.drop(["match_date", "match_dt"], axis=1, inplace=True)

    df[['home_scores','away_scores']] = (
        df['result']
          .str.extract(r'(\d+)\s*-\s*(\d+)', expand=True)
          .astype(float)
    )
    def label_winner(row):
        if pd.isna(row['home_scores']) or pd.isna(row['away_scores']):
            return np.nan
        if row['home_scores'] > row['away_scores']:
            return 'Home'
        elif row['home_scores'] < row['away_scores']:
            return 'Away'
        else:
            return 'Draw'

    df["winner"] = df.apply(label_winner, axis=1)  # Assign winner flag similarly
    return df


def mean_odds_calc(df: pd.DataFrame) -> pd.DataFrame:
    # Convert three outcomes: home, draw, away using inline lambda
    for col in ["odds_1", "odds_draw", "odds_2"]:
        df[col + "_dec"] = df[col].replace("-", np.nan).apply(
            lambda x: float(Fraction(x)) + 1
            if pd.notnull(x) and '/' in str(x)
            else (float(x) if pd.notnull(x) else np.nan)
        )

    # Compute mean over all bookmakers per match_link
    m = df.groupby("match_link", as_index=False)[
        ["odds_1_dec", "odds_draw_dec", "odds_2_dec"]
    ].mean()

    return m.rename(columns={
        "odds_1_dec": "home_team_odds",
        "odds_draw_dec": "draw_team_odds",
        "odds_2_dec": "away_team_odds"
    })

def merge_and_calculate(match_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(match_df, odds_df, on="match_link", how="inner").copy()

    if 'draw_team_odds' not in df.columns:
        df['draw_team_odds'] = np.nan

    for side in ["home", "draw", "away"]:
        col = f"{side}_team_odds"
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for side in ["home", "draw", "away"]:
        col = f"{side}_team_odds"
        df[f"{side}_prob_BE"] = (1 / df[col] * 100).round(2)

    total_BE = df["home_prob_BE"] + df["draw_prob_BE"] + df["away_prob_BE"]
    norm_factor = 100 / total_BE

    for side in ["home", "draw", "away"]:
        df[f"{side}_prob"] = (df[f"{side}_prob_BE"] * norm_factor).round(2)

    return df


def scrape_all(urls: list, driver) -> (pd.DataFrame, pd.DataFrame):
    """
    Scrape multiple URLs for historical match and odds data, returning two DataFrames.

    This function iterates over a list of URLs, retrieves match links with summaries,
    navigates to each match to extract team names, match status, and datetime, then
    optionally collects bookmaker odds including draw odds if available.

    Parameters
    ----------
    urls : list
        List of URL strings to be scraped for match links.
    driver : Selenium WebDriver
        WebDriver instance used for loading pages and content.

    Returns
    -------
    matches_df : pandas.DataFrame
        DataFrame where each row represents a scraped match.

    odds_df : pandas.DataFrame
        DataFrame of scraped betting odds, supporting 2-way or 3-way markets.
    """
    records = []
    odds_records = []
    wait = WebDriverWait(driver, 10)

    for url in urls:
        try:
            driver.get(url)
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a[href$='match-summary']")))
        except (TimeoutException, WebDriverException):
            continue  # Skip this URL if it fails to load

        soup = BeautifulSoup(driver.page_source, "html.parser")
        competition = urlparse(url).path.strip("/").split("/")[-2]
        match_links = {a["href"] for a in soup.select("a[href$='match-summary']")}

        for link in match_links:
            try:
                driver.get(link)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".fixedHeaderDuel__detailStatus")))
            except (TimeoutException, WebDriverException):
                continue  # Skip this match if page fails to load

            page = BeautifulSoup(driver.page_source, "html.parser")
            status = page.select_one(".duelParticipant__score")
            teams = page.find_all("a", class_=re.compile(".*participantName.*"))
            home_name = teams[0].text.strip() if len(teams) >= 2 else None
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

            try:
                driver.find_element(By.CSS_SELECTOR, ".wclOddsHeader").click()
            except Exception:
                pass  # odds section may already be expanded or not available

            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".wclOddsContent .odds")))
            except TimeoutException:
                continue  # No odds loaded — skip

            page = BeautifulSoup(driver.page_source, "html.parser")
            for odds in page.select(".wclOddsContent .odds"):
                bm = odds.select_one(".prematchLogo")
                vals = [v.text.strip() for v in odds.select("[data-testid='wcl-oddsValue']")]

                odds_record = {
                    "match_link": link,
                    "bookmaker": bm["title"] if bm and bm.has_attr("title") else None,
                    "odds_1": vals[0] if len(vals) > 0 else None,
                    "odds_draw": vals[1] if len(vals) == 3 else None,
                    "odds_2": vals[2] if len(vals) == 3 else (vals[1] if len(vals) == 2 else None),
                }
                odds_records.append(odds_record)

    driver.quit()
    return pd.DataFrame(records), pd.DataFrame(odds_records)





def weekly_scrape(flashscore_rugby_links: list, driver) -> (pd.DataFrame, pd.DataFrame):
    driver.implicitly_wait(5)  # Set implicit wait globally
    wait = WebDriverWait(driver, 15)
    match_links_info = []

    for url in flashscore_rugby_links:
        logging.info(f"Opening URL: {url}")
        driver.get(url)
        try:
            consent_btn = driver.find_element(By.CSS_SELECTOR, "#onetrust-button-group #onetrust-accept-btn-handler")
            consent_btn.click()
            logging.info("Cookie banner dismissed.")
            time.sleep(1)
        except NoSuchElementException:
            pass
        except ElementClickInterceptedException:
            logging.warning("Cookie banner was present but not clickable.")

        try:
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Previous day']")))
        except TimeoutException:
            logging.warning(f"'Previous day' button not found on {url}. Skipping this URL.")
            continue

        no_match_day_count = 0

        for day in range(7):
            try:
                btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Previous day']")))
                driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                try:
                    btn.click()
                except ElementClickInterceptedException:
                    driver.execute_script("arguments[0].click();", btn)
            except TimeoutException:
                logging.error(f"Could not click 'Previous day' on iteration {day + 1} for {url}")
                break

            time.sleep(0.5)

            for _ in range(5):
                try:
                    matches = driver.find_elements(By.CSS_SELECTOR, "div.event__match")
                    if matches:
                        break
                except StaleElementReferenceException:
                    time.sleep(0.5)
            else:
                matches = []

            soup = BeautifulSoup(driver.page_source, "html.parser")
            matches = soup.find_all("div", class_="event__match")

            if not matches:
                no_match_day_count += 1
                logging.info(f"No matches found on day {day + 1} for {url}")
                continue
            else:
                logging.info(f"Found {len(matches)} matches on day {day + 1} for {url}")

            for match in matches:
                comp_div = match.find_previous("div", class_="event__titleBox")
                competition = comp_div.get_text(strip=True) if comp_div else None
                for a in match.select("a[href$='match-summary']"):
                    match_links_info.append({
                        "competition": competition,
                        "match_link": a["href"]
                    })

        if no_match_day_count == 7:
            logging.warning(f"No matches found for any of the past 7 days at {url}")

    records = []
    odds_records = []

    for info in match_links_info:
        competition = info["competition"]
        link = info["match_link"]
        logging.info(f"Scraping match: {link}")

        try:
            driver.get(link)
            try:
                consent_btn = driver.find_element(By.CSS_SELECTOR, "#onetrust-button-group #onetrust-accept-btn-handler")
                consent_btn.click()
                logging.info("Cookie banner dismissed.")
                time.sleep(1)
            except NoSuchElementException:
                pass
            except ElementClickInterceptedException:
                logging.warning("Cookie banner was present but not clickable.")
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".fixedHeaderDuel__detailStatus")))
        except TimeoutException:
            logging.error(f"Match details not found for {link}. Skipping.")
            continue

        page = BeautifulSoup(driver.page_source, "html.parser")
        status = page.select_one(".duelParticipant__score")
        teams = page.find_all("a", class_=re.compile(".*participantName.*"))
        home_name = teams[0].text.strip() if len(teams) >= 2 else None
        away_name = teams[1].text.strip() if len(teams) >= 2 else None
        date_el = page.select_one(".duelParticipant__startTime")

        records.append({
            "match_link": link,
            "competition": competition,
            "match_date": date_el.get_text(strip=True) if date_el else None,
            "home_team": home_name,
            "away_team": away_name,
            "result": status.get_text(strip=True) if status else None,
        })

        try:
            driver.find_element(By.CSS_SELECTOR, ".wclOddsHeader").click()
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".wclOddsContent .odds")))
            soup2 = BeautifulSoup(driver.page_source, "html.parser")
            odds_elements = soup2.select(".wclOddsContent .odds")
            for odds in odds_elements:
                bm = odds.select_one(".prematchLogo")
                vals = [v.get_text(strip=True) for v in odds.select("[data-testid='wcl-oddsValue']")]

                odds_records.append({
                    "match_link": link,
                    "bookmaker": bm["title"] if bm and bm.has_attr("title") else None,
                    "odds_1": vals[0] if len(vals) > 0 else None,
                    "odds_draw": vals[1] if len(vals) == 3 else None,
                    "odds_2": vals[2] if len(vals) == 3 else (vals[1] if len(vals) == 2 else None),
                })
        except (NoSuchElementException, TimeoutException):
            logging.info(f"No odds available or failed to load odds for {link}")

    driver.quit()
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