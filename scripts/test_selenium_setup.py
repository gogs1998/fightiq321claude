"""
Test script to verify Selenium and scraper setup

Run this to check if everything is configured correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from loguru import logger

def test_imports():
    """Test if required packages are installed"""
    logger.info("Testing package imports...")

    try:
        import selenium
        logger.success("✓ Selenium installed")
    except ImportError:
        logger.error("✗ Selenium not installed. Run: pip install selenium")
        return False

    try:
        from webdriver_manager.chrome import ChromeDriverManager
        logger.success("✓ webdriver-manager installed")
    except ImportError:
        logger.error("✗ webdriver-manager not installed. Run: pip install webdriver-manager")
        return False

    try:
        import requests
        import beautifulsoup4
        logger.success("✓ Web scraping libraries installed")
    except ImportError:
        logger.error("✗ Missing web scraping libraries. Run: pip install requests beautifulsoup4")
        return False

    return True


def test_chrome():
    """Test if Chrome/ChromeDriver is available"""
    logger.info("\nTesting Chrome browser...")

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager

        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        driver.get("https://www.google.com")
        logger.success("✓ Chrome/ChromeDriver working correctly")

        driver.quit()
        return True

    except Exception as e:
        logger.error(f"✗ Chrome/ChromeDriver test failed: {e}")
        logger.info("\nTo fix this:")
        logger.info("  Ubuntu/Debian: sudo apt-get install google-chrome-stable")
        logger.info("  MacOS: brew install --cask google-chrome")
        logger.info("  Windows: Download from https://www.google.com/chrome/")
        return False


def test_scraper():
    """Test the BestFightOdds scraper"""
    logger.info("\nTesting BestFightOdds scraper...")

    try:
        from scripts.fetch_odds_bestfightodds import BestFightOddsScraper, SELENIUM_AVAILABLE

        if not SELENIUM_AVAILABLE:
            logger.warning("⚠️  Selenium not available - using fallback mode")

        scraper = BestFightOddsScraper(use_selenium=SELENIUM_AVAILABLE, headless=True)

        # Test scraping (this will attempt to fetch the UFC events page)
        logger.info("Attempting to fetch UFC events list...")
        events = scraper.get_all_ufc_events()

        if events:
            logger.success(f"✓ Successfully fetched {len(events)} events")
            logger.info(f"\nSample events:")
            for event in events[:3]:
                logger.info(f"  - {event['name']} ({event.get('date', 'Date unknown')})")
            return True
        else:
            logger.warning("⚠️  No events found (may need Selenium)")
            return False

    except Exception as e:
        logger.error(f"✗ Scraper test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("="*80)
    logger.info("BESTFIGHTODDS SCRAPER - SETUP TEST")
    logger.info("="*80 + "\n")

    results = {
        'imports': test_imports(),
        'chrome': test_chrome(),
        'scraper': test_scraper()
    }

    logger.info("\n" + "="*80)
    logger.info("TEST RESULTS")
    logger.info("="*80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name.capitalize()}: {status}")

    if all(results.values()):
        logger.success("\n🎉 All tests passed! Scraper is ready to use.")
        logger.info("\nNext steps:")
        logger.info("  1. Run: python scripts/fetch_odds_bestfightodds.py")
        logger.info("  2. Run: python scripts/predict_upcoming_with_bestfightodds.py")
    else:
        logger.warning("\n⚠️  Some tests failed. Check errors above.")
        logger.info("\nSetup guide: scripts/SELENIUM_SETUP.md")


if __name__ == "__main__":
    main()
