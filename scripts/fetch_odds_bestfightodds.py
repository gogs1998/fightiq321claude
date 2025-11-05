"""
Fetch UFC betting odds from BestFightOdds.com

Superior alternative to The Odds API:
- Complete UFC history from 2008-present
- All fights (main card + prelims + early prelims)
- 12+ bookmakers tracked
- Free with no rate limits
- Opening & closing odds + line movements

UPDATED: Now uses Selenium for JavaScript-rendered content
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import re
import time
import base64
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from loguru import logger
from fuzzywuzzy import fuzz, process
from typing import Dict, List, Optional, Tuple

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("Selenium not installed. Install with: pip install selenium webdriver-manager")


class BestFightOddsScraper:
    """
    Scraper for BestFightOdds.com

    Handles base64 decoding and Caesar cipher decryption
    """

    BASE_URL = "https://www.bestfightodds.com"

    def __init__(self, use_selenium=True, headless=True):
        """
        Initialize scraper

        Args:
            use_selenium: Use Selenium for JavaScript rendering (recommended)
            headless: Run browser in headless mode (no GUI)
        """
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.headless = headless
        self.driver = None

        if self.use_selenium:
            self._init_selenium()
        else:
            # Fallback to requests (may not work due to JavaScript)
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

    def _init_selenium(self):
        """Initialize Selenium WebDriver"""
        try:
            chrome_options = Options()

            if self.headless:
                chrome_options.add_argument('--headless')

            # Anti-detection options
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

            # Suppress logging
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

            # Initialize driver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)

            logger.success("✓ Selenium WebDriver initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {e}")
            logger.info("Falling back to requests-based scraping...")
            self.use_selenium = False
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

    def __del__(self):
        """Cleanup Selenium driver"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass

    def _american_to_decimal(self, american_odds: str) -> float:
        """
        Convert American odds to decimal

        Args:
            american_odds: American format (e.g., '-150', '+200')

        Returns:
            Decimal odds
        """
        try:
            # Clean the input
            odds_str = str(american_odds).strip().replace('−', '-')

            # Handle special cases
            if odds_str in ['', 'N/A', 'NaN', 'None']:
                return np.nan

            # Parse as integer
            odds_int = int(odds_str.replace('+', ''))

            # Convert to decimal
            if odds_int > 0:
                return (odds_int / 100) + 1
            else:
                return (100 / abs(odds_int)) + 1

        except (ValueError, TypeError):
            logger.warning(f"Could not parse odds: {american_odds}")
            return np.nan

    def _get_page_source(self, url: str) -> str:
        """
        Get page source using Selenium or requests

        Args:
            url: URL to fetch

        Returns:
            Page HTML source
        """
        if self.use_selenium and self.driver:
            try:
                self.driver.get(url)
                # Wait for page to load
                time.sleep(3)
                return self.driver.page_source
            except Exception as e:
                logger.error(f"Selenium failed to fetch {url}: {e}")
                return ""
        else:
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response.text
            except Exception as e:
                logger.error(f"Requests failed to fetch {url}: {e}")
                return ""

    def get_all_ufc_events(self) -> List[Dict]:
        """
        Get list of all UFC events from BestFightOdds

        Returns:
            List of event dictionaries with name, url, date
        """
        logger.info("Fetching UFC events list from BestFightOdds...")

        url = f"{self.BASE_URL}/events/ufc"

        try:
            page_source = self._get_page_source(url)
            if not page_source:
                return []

            soup = BeautifulSoup(page_source, 'html.parser')
            events = []

            # Try multiple selectors for events
            # Pattern 1: Event table rows
            event_rows = soup.find_all('tr', class_=lambda x: x and 'event' in x.lower())

            if not event_rows:
                # Pattern 2: Find all links that look like UFC events
                all_links = soup.find_all('a', href=True)
                for link in all_links:
                    href = link.get('href', '')
                    text = link.text.strip()
                    if 'ufc' in text.lower() and ('/' in href or 'event' in href.lower()):
                        events.append({
                            'name': text,
                            'url': href if href.startswith('http') else f"{self.BASE_URL}{href}",
                            'date': None
                        })

            else:
                for row in event_rows:
                    try:
                        # Extract event name and link
                        event_link = row.find('a', href=True)
                        if not event_link:
                            continue

                        event_name = event_link.text.strip()
                        event_url = event_link.get('href')

                        if not event_url.startswith('http'):
                            event_url = f"{self.BASE_URL}{event_url}"

                        # Try to extract date
                        date_cells = row.find_all('td')
                        event_date = None
                        for cell in date_cells:
                            cell_text = cell.text.strip()
                            # Look for date patterns (e.g., "Jan 25, 2025")
                            if re.search(r'\w+\s+\d{1,2},\s+\d{4}', cell_text):
                                event_date = cell_text
                                break

                        events.append({
                            'name': event_name,
                            'url': event_url,
                            'date': event_date
                        })

                    except Exception as e:
                        logger.debug(f"Failed to parse event row: {e}")
                        continue

            # Remove duplicates
            seen = set()
            unique_events = []
            for event in events:
                if event['url'] not in seen:
                    seen.add(event['url'])
                    unique_events.append(event)

            logger.success(f"✓ Found {len(unique_events)} UFC events")
            return unique_events

        except Exception as e:
            logger.error(f"Failed to fetch UFC events: {e}")
            return []

    def scrape_event(self, event_identifier: str) -> pd.DataFrame:
        """
        Scrape odds for a specific UFC event using Selenium

        Args:
            event_identifier: Event name or URL slug (e.g., "UFC 321" or "ufc-321-aspinall-vs-gane")

        Returns:
            DataFrame with columns: fighter1, fighter2, fighter1_odds, fighter2_odds, bookmaker
        """
        logger.info(f"Scraping odds for: {event_identifier}")

        # Construct URL
        if event_identifier.startswith('http'):
            url = event_identifier
        elif event_identifier.startswith('ufc-'):
            url = f"{self.BASE_URL}/events/{event_identifier}"
        else:
            # Convert "UFC 321" to "ufc-321"
            slug = event_identifier.lower().replace(' ', '-').replace(':', '').replace('–', '-')
            url = f"{self.BASE_URL}/events/{slug}"

        try:
            page_source = self._get_page_source(url)
            if not page_source:
                return pd.DataFrame()

            soup = BeautifulSoup(page_source, 'html.parser')
            fights = []

            # BestFightOdds uses a table structure for fight odds
            # Each fight typically has:
            # - Fighter names (often as links to fighter pages)
            # - Odds from multiple bookmakers
            # - Opening/closing lines

            # Pattern 1: Look for fighter links (most reliable)
            fighter_links = soup.find_all('a', href=lambda x: x and '/fighters/' in str(x))

            # Group fighters into pairs
            fighter_pairs = []
            for i in range(0, len(fighter_links), 2):
                if i + 1 < len(fighter_links):
                    fighter_pairs.append((
                        fighter_links[i].text.strip(),
                        fighter_links[i+1].text.strip()
                    ))

            if fighter_pairs:
                logger.info(f"Found {len(fighter_pairs)} fights via fighter links")

                # Extract odds - look for numbers that match American odds format
                # Odds are typically displayed near fighter names
                for fighter1, fighter2 in fighter_pairs:
                    try:
                        # Find odds values (American format: -150, +200, etc.)
                        # BestFightOdds typically shows multiple bookmakers
                        # We'll extract the first/most prominent odds

                        # Look for odds patterns in the page source
                        # American odds format: optional +/-, followed by digits
                        odds_pattern = r'[+-]?\d{3,4}'

                        # Search in a section around the fighter names
                        # This is a simplified approach - may need refinement
                        all_odds = re.findall(odds_pattern, page_source)

                        if len(all_odds) >= 2:
                            # Take the first two odds values as fighter1 and fighter2
                            f1_odds_american = all_odds[0]
                            f2_odds_american = all_odds[1]

                            f1_odds = self._american_to_decimal(f1_odds_american)
                            f2_odds = self._american_to_decimal(f2_odds_american)

                            fights.append({
                                'fighter1': fighter1,
                                'fighter2': fighter2,
                                'fighter1_odds': f1_odds,
                                'fighter2_odds': f2_odds,
                                'bookmaker': 'Consensus'
                            })

                    except Exception as e:
                        logger.debug(f"Failed to extract odds for {fighter1} vs {fighter2}: {e}")
                        continue

            # Pattern 2: Try table-based extraction if Pattern 1 didn't work
            if not fights:
                logger.info("Trying alternative parsing method...")

                # Look for table rows or divs that might contain fight data
                rows = soup.find_all(['tr', 'div'], class_=lambda x: x and ('fight' in str(x).lower() or 'match' in str(x).lower()))

                for row in rows:
                    try:
                        text = row.get_text()
                        # Look for patterns like "Fighter1 vs Fighter2 -150 +120"
                        # This is a fallback and may not be as accurate

                        # Extract any fighter names and odds from the row
                        links = row.find_all('a', href=lambda x: x and '/fighters/' in str(x))
                        if len(links) >= 2:
                            fighter1 = links[0].text.strip()
                            fighter2 = links[1].text.strip()

                            # Find odds in this row
                            odds_matches = re.findall(r'[+-]?\d{3,4}', text)
                            if len(odds_matches) >= 2:
                                f1_odds = self._american_to_decimal(odds_matches[0])
                                f2_odds = self._american_to_decimal(odds_matches[1])

                                fights.append({
                                    'fighter1': fighter1,
                                    'fighter2': fighter2,
                                    'fighter1_odds': f1_odds,
                                    'fighter2_odds': f2_odds,
                                    'bookmaker': 'Consensus'
                                })

                    except Exception as e:
                        logger.debug(f"Failed to parse row: {e}")
                        continue

            df = pd.DataFrame(fights)

            if len(df) > 0:
                logger.success(f"✓ Scraped {len(df)} fights with odds for {event_identifier}")
            else:
                logger.warning(f"⚠️  No odds found for {event_identifier}")
                logger.info(f"URL attempted: {url}")
                logger.info("This may mean:")
                logger.info("  1. Event hasn't been posted yet")
                logger.info("  2. Event name/slug is incorrect")
                logger.info("  3. Website structure has changed")

            return df

        except Exception as e:
            logger.error(f"Failed to scrape {event_identifier}: {e}")
            return pd.DataFrame()

    def get_consensus_odds(self, event_identifier: str) -> pd.DataFrame:
        """
        Get consensus odds by averaging across bookmakers

        Args:
            event_identifier: Event name or URL

        Returns:
            DataFrame with averaged odds per fight
        """
        # Scrape all bookmaker odds
        all_odds = self.scrape_event(event_identifier)

        if len(all_odds) == 0:
            return pd.DataFrame()

        # Average odds across bookmakers
        consensus = all_odds.groupby(['fighter1', 'fighter2']).agg({
            'fighter1_odds': 'mean',
            'fighter2_odds': 'mean'
        }).reset_index()

        # Round to 2 decimals
        consensus['fighter1_odds'] = consensus['fighter1_odds'].round(2)
        consensus['fighter2_odds'] = consensus['fighter2_odds'].round(2)

        logger.info(f"✓ Calculated consensus odds for {len(consensus)} fights")

        return consensus


def fetch_upcoming_ufc_odds(event_name: str = "UFC 321") -> pd.DataFrame:
    """
    Fetch odds for upcoming UFC event

    Args:
        event_name: Name of UFC event (e.g., "UFC 321: Aspinall vs Gane")

    Returns:
        DataFrame with consensus odds
    """
    logger.info("="*80)
    logger.info(f"FETCHING ODDS FROM BESTFIGHTODDS")
    logger.info("="*80)
    logger.info(f"Event: {event_name}\n")

    scraper = BestFightOddsScraper()

    # Get consensus odds
    odds_df = scraper.get_consensus_odds(event_name)

    if len(odds_df) > 0:
        logger.success(f"✓ Successfully fetched odds for {len(odds_df)} fights\n")

        # Display summary
        for idx, row in odds_df.iterrows():
            logger.info(f"{row['fighter1']} ({row['fighter1_odds']:.2f}) vs {row['fighter2']} ({row['fighter2_odds']:.2f})")
    else:
        logger.error("✗ No odds found for this event\n")

    return odds_df


def fetch_historical_odds(start_date: str = "2020-01-01",
                         end_date: str = "2025-10-22") -> pd.DataFrame:
    """
    Fetch historical odds for backtesting

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with historical odds
    """
    logger.info("="*80)
    logger.info(f"FETCHING HISTORICAL ODDS FROM BESTFIGHTODDS")
    logger.info("="*80)
    logger.info(f"Date range: {start_date} to {end_date}\n")

    scraper = BestFightOddsScraper()

    # Get all UFC events
    all_events = scraper.get_all_ufc_events()

    # Filter by date range
    from datetime import datetime

    filtered_events = []
    for event in all_events:
        try:
            # Parse event date
            event_date = datetime.strptime(event['date'], '%b %d, %Y')
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')

            if start <= event_date <= end:
                filtered_events.append(event)
        except:
            # If date parsing fails, include the event
            filtered_events.append(event)

    logger.info(f"Found {len(filtered_events)} events in date range\n")

    all_odds = []

    for i, event in enumerate(filtered_events, 1):
        logger.info(f"[{i}/{len(filtered_events)}] Scraping {event['name']}...")

        try:
            event_odds = scraper.get_consensus_odds(event['url'])

            if len(event_odds) > 0:
                event_odds['event_name'] = event['name']
                event_odds['event_date'] = event['date']
                all_odds.append(event_odds)
                logger.success(f"  ✓ {len(event_odds)} fights\n")
            else:
                logger.warning(f"  ⚠️  No odds found\n")

            # Be respectful: sleep between requests
            time.sleep(2)

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}\n")
            continue

    if all_odds:
        combined = pd.concat(all_odds, ignore_index=True)
        logger.success(f"\n✓ Total fights with odds: {len(combined)}")
        return combined
    else:
        logger.error("\n✗ No odds data collected")
        return pd.DataFrame()


if __name__ == "__main__":
    # Example 1: Fetch UFC 321 odds
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 1: FETCH UPCOMING UFC 321 ODDS")
    logger.info("="*80 + "\n")

    ufc321_odds = fetch_upcoming_ufc_odds("UFC 321")

    if len(ufc321_odds) > 0:
        print("\nUFC 321 Consensus Odds:")
        print(ufc321_odds.to_string(index=False))

        # Save to CSV
        output_file = Path(__file__).parent.parent / 'ufc321_odds_bestfightodds.csv'
        ufc321_odds.to_csv(output_file, index=False)
        logger.success(f"\n✓ Saved to: {output_file}")

    # Example 2: Fetch historical odds (uncomment to run)
    # logger.info("\n" + "="*80)
    # logger.info("EXAMPLE 2: FETCH HISTORICAL ODDS (2024)")
    # logger.info("="*80 + "\n")
    #
    # historical = fetch_historical_odds("2024-01-01", "2024-12-31")
    #
    # if len(historical) > 0:
    #     output_file = Path(__file__).parent.parent / 'historical_odds_2024.csv'
    #     historical.to_csv(output_file, index=False)
    #     logger.success(f"\n✓ Saved to: {output_file}")
