"""
Debug parsing of race results table.
"""
import requests
from bs4 import BeautifulSoup
import re

def debug_parsing():
    url = "https://nar.netkeiba.com/race/result.html?race_id=202446121701"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    rows = soup.select('tr')
    print(f"Total rows found: {len(rows)}")
    
    for i, row in enumerate(rows[:5]): # Check first few rows
        print(f"\n--- Row {i} ---")
        print(str(row)[:200] + "...")
        
        finish = row.select_one('span.Num, td.Num')
        print(f"  Finish: {finish.get_text().strip() if finish else 'None'}")
        
        gate = row.select_one('span.Waku, td.Waku')
        print(f"  Gate: {gate.get_text().strip() if gate else 'None'}")
        
        num = row.select_one('td.Umaban, span.Umaban')
        print(f"  Num: {num.get_text().strip() if num else 'None'}")

if __name__ == '__main__':
    debug_parsing()
