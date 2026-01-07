"""
Test actual Kanazawa race URLs to find correct format.
"""
import requests
from bs4 import BeautifulSoup
import time

def test_race_id_formats():
    """Test different race ID formats to find what works."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    test_ids = [
        # Try different known race ID formats
        '202401072901',  # Standard: YYYYMMDD + Track + Race
        '2024010729001',  # Alternative format
        '202401072900101',  # JRA style
        
        # Try recent races (2024)
        '202412012901',  # December 2024
        '202411032901',  # November 2024
        
        # Try 2023
        '202312032901',
        '202311052901',
        
        # Try older (2020)
        '202012062901',
        '202011292901',
    ]
    
    print("Testing race ID formats...\n")
    
    for race_id in test_ids:
        url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'
        
        try:
            response = session.get(url, timeout=10)
            status = response.status_code
            
            if status == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                race_name = soup.select_one('.RaceName')
                
                if race_name and race_name.get_text().strip():
                    print(f"✓ FOUND! {race_id}")
                    print(f"  URL: {url}")
                    print(f"  Race: {race_name.get_text().strip()}")
                    print(f"  Status: {status}\n")
                    return race_id
                else:
                    print(f"  {race_id}: 200 but no race name")
            else:
                print(f"  {race_id}: Status {status}")
            
            time.sleep(1)
            
        except Exception as e:
            print(f"  {race_id}: Error - {e}")
    
    print("\nNo working race ID found. Trying to find recent races...")
    
    # Try to find race list page
    list_url = 'https://nar.netkeiba.com/top/'
    try:
        response = session.get(list_url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for any race links
            links = soup.select('a[href*="race_id="]')
            print(f"\nFound {len(links)} race links on top page")
            
            for link in links[:5]:
                print(f"  - {link.get('href')}")
                print(f"    Text: {link.get_text().strip()}")
    except:
        pass

if __name__ == '__main__':
    test_race_id_formats()
