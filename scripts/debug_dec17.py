"""
Debug Dec 17 race ID.
"""
import requests
from bs4 import BeautifulSoup

def debug():
    # Dec 17, 2024
    url = "https://nar.netkeiba.com/race/result.html?race_id=202446121701"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    print(f"Title: {soup.title.string if soup.title else 'No Title'}")
    
    race_name = soup.select_one('.RaceName')
    if race_name:
        print(f"RaceName: {race_name.get_text().strip()}")
    else:
        print("RaceName not found.")
        # Check for cancellation text
        if "中止" in response.text or "雪" in response.text:
            print("Possible cancellation detected in text.")
        
        # Check if we are on the top page (redirect)
        if "netkeiba.com/top/" in response.url:
            print(f"Redirected to: {response.url}")

if __name__ == '__main__':
    debug()
