import requests
from bs4 import BeautifulSoup
import re
import time
import random, logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time




session = requests.Session()    

HEADERS = {
    "User-Agent": "Prasanna Balaji (prasannabalaji470@gmail.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.sec.gov/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}


def fetch_10k_html_selenium(cik: str, filing_type: str = "10-K", year: int = None):
    print(f"[INFO] Starting fetch_10k_html_selenium for CIK={cik}, filing_type={filing_type}, year={year}")
    options = Options()
    options.headless = False
    driver = webdriver.Chrome(options=options)

    try:
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={cik}&type={filing_type}&owner=exclude&count=100"
        print(f"[INFO] Navigating to filings page: {url}")
        driver.get(url)
        time.sleep(3)

        print("[INFO] Finding all filing rows on the page")
        filings = driver.find_elements(By.CSS_SELECTOR, "table.tableFile2 tr")[1:]
        print(f"[INFO] Number of filings found (including all types): {len(filings)}")

        for idx, row in enumerate(filings):
            print(f"[INFO] Processing filing row #{idx + 1}")
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) < 4:
                print("[WARN] Not enough columns, skipping row")
                continue

            filing_type_text = cells[0].text.strip()
            filing_date_text = cells[3].text.strip()
            print(f"[DEBUG] Filing type: {filing_type_text}, Filing date: {filing_date_text}")

            if filing_type_text.upper() != filing_type.upper():
                print(f"[INFO] Skipping filing because type '{filing_type_text}' != '{filing_type}'")
                continue
            if year and str(year) not in filing_date_text:
                print(f"[INFO] Skipping filing due to year mismatch (looking for {year})")
                continue

            try:
                doc_link = cells[1].find_element(By.TAG_NAME, 'a')
            except Exception:
                print("[ERROR] Could not find Documents link, skipping row")
                continue
            
            doc_href = doc_link.get_attribute("href")
            print(f"[INFO] Navigating to filing detail page: {doc_href}")
            doc_link.click()
            time.sleep(3)
            
            files = driver.find_elements(By.CSS_SELECTOR, "table.tableFile tr")[1:]
            print(f"[INFO] Number of document files listed: {len(files)}")
            for file_idx, file_row in enumerate(files):
                cols = file_row.find_elements(By.TAG_NAME, "td")
                if len(cols) < 3:
                    continue
                file_name = cols[2].text.lower()
                print(f"[DEBUG] File #{file_idx + 1}: {file_name}")
                if file_name.endswith(".htm") or file_name.endswith(".html"):
                    html_link = cols[2].find_element(By.TAG_NAME, "a")
                    html_href = html_link.get_attribute("href")
                    print(f"[INFO] Clicking HTML document link: {html_href}")
                    html_link.click()
                    time.sleep(3)

                    filing_html = driver.page_source
                    print("[INFO] Successfully fetched full HTML content of filing page")

                    return filing_html
            
            print("[WARN] No suitable HTML filing document found in filing detail, continuing to next filing")
            driver.back()
            time.sleep(2)

        print("[INFO] Completed all filings, no matching 10-K filing found")
        return None

    except Exception as err:
        print(f"[ERROR] Exception during Selenium scraping: {err}")
        return None

    finally:
        print("[INFO] Closing Selenium webdriver")
        driver.quit()

