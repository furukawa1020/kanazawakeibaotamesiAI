"""
Debug script to check why scraper is failing on known valid ID.
Target: 202446120101 (Kanazawa 2024-12-01 1R)
"""
import requests
from bs4 import BeautifulSoup
import time

def debug_scrape():
    # Known valid ID
    race_id = "202446120101"
    url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'
    
    print(f"Testing URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check Title
        title = soup.title.string if soup.title else "No Title"
        print(f"Page Title: {title}")
        
        # Check RaceName
        race_name = soup.select_one('.RaceName')
        if race_name:
            print(f"RaceName Found: {race_name.get_text().strip()}")
        else:
            print("RaceName NOT FOUND")
            # Print potentially helpful info
            print("Table count:", len(soup.find_all('table')))
            print("Classes on body:", soup.body.get('class'))
            
            # Check for redirect or login
            if "login" in response.url:
                print("Redirected to login?")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == '__main__':
    debug_scrape()
