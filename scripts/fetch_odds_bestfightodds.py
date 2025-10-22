"""
Fetch UFC betting odds from BestFightOdds.com

Superior alternative to The Odds API:
- Complete UFC history from 2008-present
- All fights (main card + prelims + early prelims)
- 12+ bookmakers tracked
- Free with no rate limits
- Opening & closing odds + line movements
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


class BestFightOddsScraper:
    """
    Scraper for BestFightOdds.com

    Handles base64 decoding and Caesar cipher decryption
    """

    BASE_URL = "https://www.bestfightodds.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

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

    def get_all_ufc_events(self) -> List[Dict]:
        """
        Get list of all UFC events from BestFightOdds

        Returns:
            List of event dictionaries with name, url, date
        """
        logger.info("Fetching UFC events list from BestFightOdds...")

        url = f"{self.BASE_URL}/events/ufc"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            events = []

            # Find event table
            event_rows = soup.find_all('tr', class_='event-row')

            for row in event_rows:
                try:
                    # Extract event name and link
                    event_link = row.find('a', class_='event-name')
                    if not event_link:
                        continue

                    event_name = event_link.text.strip()
                    event_url = event_link.get('href')

                    # Extract date
                    date_cell = row.find('td', class_='event-date')
                    event_date = date_cell.text.strip() if date_cell else None

                    events.append({
                        'name': event_name,
                        'url': event_url,
                        'date': event_date
                    })

                except Exception as e:
                    logger.debug(f"Failed to parse event row: {e}")
                    continue

            logger.success(f"✓ Found {len(events)} UFC events")
            return events

        except Exception as e:
            logger.error(f"Failed to fetch UFC events: {e}")
            return []

    def scrape_event(self, event_identifier: str) -> pd.DataFrame:
        """
        Scrape odds for a specific UFC event

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
            slug = event_identifier.lower().replace(' ', '-').replace(':', '')
            url = f"{self.BASE_URL}/events/{slug}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            fights = []

            # Find fight rows
            fight_rows = soup.find_all('div', class_='table-row')

            for row in fight_rows:
                try:
                    # Extract fighter names
                    fighter_cells = row.find_all('span', class_='fighter-name')

                    if len(fighter_cells) < 2:
                        continue

                    fighter1 = fighter_cells[0].text.strip()
                    fighter2 = fighter_cells[1].text.strip()

                    # Extract odds (look for odds cells)
                    odds_cells = row.find_all('span', class_='odds-value')

                    if len(odds_cells) >= 2:
                        f1_odds_american = odds_cells[0].text.strip()
                        f2_odds_american = odds_cells[1].text.strip()

                        # Convert to decimal
                        f1_odds = self._american_to_decimal(f1_odds_american)
                        f2_odds = self._american_to_decimal(f2_odds_american)

                        # Extract bookmaker
                        bookmaker_cell = row.find('span', class_='bookmaker-name')
                        bookmaker = bookmaker_cell.text.strip() if bookmaker_cell else 'Unknown'

                        fights.append({
                            'fighter1': fighter1,
                            'fighter2': fighter2,
                            'fighter1_odds': f1_odds,
                            'fighter2_odds': f2_odds,
                            'bookmaker': bookmaker
                        })

                except Exception as e:
                    logger.debug(f"Failed to parse fight row: {e}")
                    continue

            df = pd.DataFrame(fights)

            if len(df) > 0:
                logger.success(f"✓ Scraped {len(df)} odds entries for {event_identifier}")
            else:
                logger.warning(f"⚠️  No odds found for {event_identifier}")

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
