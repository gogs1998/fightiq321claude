# Selenium Setup for BestFightOdds Scraper

## Installation Instructions

### Option 1: Local Installation (Recommended for Development)

#### On Ubuntu/Debian:
```bash
# Install Chrome
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
sudo apt-get update
sudo apt-get install -y google-chrome-stable

# Install Python dependencies
pip install selenium webdriver-manager
```

#### On MacOS:
```bash
# Install Chrome (if not already installed)
brew install --cask google-chrome

# Install Python dependencies
pip install selenium webdriver-manager
```

#### On Windows:
```powershell
# Download and install Chrome from: https://www.google.com/chrome/

# Install Python dependencies
pip install selenium webdriver-manager
```

### Option 2: Docker Installation

Create a `Dockerfile` with Chrome support:

```dockerfile
FROM python:3.10-slim

# Install Chrome and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . .

CMD ["python", "scripts/fetch_odds_bestfightodds.py"]
```

### Option 3: Use Requests Fallback (Limited Functionality)

If you can't install Chrome, the scraper will attempt to use `requests` instead:

```python
scraper = BestFightOddsScraper(use_selenium=False)
```

**Note**: BestFightOdds uses JavaScript rendering, so the fallback may not work reliably.

## Testing the Installation

Run the test script:

```bash
python scripts/test_selenium_setup.py
```

This will verify:
- ✓ Selenium is installed
- ✓ Chrome/ChromeDriver is available
- ✓ Scraper can access BestFightOdds

## Troubleshooting

### Error: "Chrome not found"
- Install Chrome using the instructions above
- Make sure Chrome is in your PATH

### Error: "ChromeDriver version mismatch"
- The `webdriver-manager` package should handle this automatically
- If issues persist, manually download ChromeDriver from: https://chromedriver.chromium.org/

### Error: "403 Forbidden" or "Access Denied"
- This means the website is blocking automated requests
- Make sure you're using Selenium (not requests)
- Try adding delays between requests
- Consider using residential proxies if needed

### Headless Mode Issues
- If headless mode fails, try running with headless=False:
  ```python
  scraper = BestFightOddsScraper(headless=False)
  ```

## Alternative: Use Existing Package

If you prefer not to set up Selenium, consider using the battle-tested package:

```bash
git clone https://github.com/DanMcInerney/BestfightoddsScraper.git
cd BestfightoddsScraper
pip install -r requirements.txt
```

Then import and use:
```python
from bestfightodds_scraper import BFOScraper
scraper = BFOScraper()
odds = scraper.scrape_event("UFC 321")
```

## Production Considerations

For production use:

1. **Run scraper on schedule** (e.g., daily at 6 AM)
2. **Cache results** to avoid repeated requests
3. **Respect rate limits** (2-3 second delays between requests)
4. **Monitor for failures** and alert if scraping fails
5. **Use headless mode** to save resources
6. **Consider proxies** if scraping at scale

## Next Steps

Once Selenium is set up:

1. Test the scraper: `python scripts/fetch_odds_bestfightodds.py`
2. Run predictions: `python scripts/predict_upcoming_with_bestfightodds.py`
3. Set up automated daily scraping (cron job or Task Scheduler)
