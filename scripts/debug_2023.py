"""
Debug 2023 connectivity.
"""
import requests
import time

def debug_2023():
    # Dec 3, 2023 (Sunday)
    race_id = "202346120301"
    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    
    print(f"Testing {url}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        start = time.time()
        response = requests.get(url, headers=headers, timeout=10)
        dur = time.time() - start
        print(f"Status: {response.status_code}")
        print(f"Time: {dur:.2f}s")
        
        if response.status_code == 200:
            if "RaceName" in response.text:
                print("Content: OK (RaceName found)")
            else:
                print("Content: Warning (RaceName not found in text)")
        else:
            print("Block/Error detected.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    debug_2023()
