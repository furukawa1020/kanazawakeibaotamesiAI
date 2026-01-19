"""
Production Scraper for Kanazawa Horse Racing.
v2: Enhanced Logging to debug misses.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Setup logging
log_file = Path('data_collection.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def scrape_race(race_id):
    url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'
    
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Randomize sleep slightly?
        time.sleep(1.2)
        
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            logger.warning(f"  [Status {response.status_code}] {race_id}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        race_name_elem = soup.select_one('.RaceName')
        if not race_name_elem:
            # Check for specific failure indicators
            title = soup.title.string if soup.title else ""
            if "Just a moment" in title or "Cloudflare" in response.text:
                logger.error("  [BLOCKED] Cloudflare detection!")
            elif "存在しません" in response.text:
                pass # Just no race
            else:
                logger.debug(f"  [No RaceName] {race_id} (Title: {title})")
            return None
            
        race_name = race_name_elem.get_text().strip()
        
        # ... (rest of parsing logic same as before) ...
        race_info = {'race_id': race_id, 'race_name': race_name}
        
        try:
            year = race_id[:4]
            mmdd = race_id[6:10]
            race_info['date'] = f"{year}-{mmdd[:2]}-{mmdd[2:]}"
        except:
            pass
            
        data1 = soup.select_one('div.RaceData01')
        if data1:
            text = data1.get_text()
            race_info['surface'] = 'ダ' if 'ダ' in text else ( '芝' if '芝' in text else 'Other')
            m = re.search(r'(\d{3,4})m', text)
            if m: race_info['distance'] = int(m.group(1))
        
        data2 = soup.select_one('div.RaceData02')
        cond_text = data2.get_text() if data2 else ""
        if not cond_text:
             for s in soup.select('span'):
               if '天候' in s.get_text():
                   cond_text = s.get_text()
                   break
        
        if '良' in cond_text: race_info['track_condition'] = '良'
        elif '稍' in cond_text: race_info['track_condition'] = '稍重'
        elif '重' in cond_text: race_info['track_condition'] = '重'
        elif '不' in cond_text: race_info['track_condition'] = '不良'
        else: race_info['track_condition'] = '良'
        
        race_info['class'] = 'C'
        if 'A' in race_name: race_info['class'] = 'A'
        elif 'B' in race_name: race_info['class'] = 'B'
        
        results = []
        rows = soup.select('tr')
        
        for row in rows:
            try:
                rank_elem = row.select_one('.Rank')
                if not rank_elem: continue
                rank_text = rank_elem.get_text().strip()
                if not rank_text.isdigit(): continue
                
                horse = race_info.copy()
                horse['finish_position'] = int(rank_text)
                
                nums = row.select('td.Num')
                if len(nums) >= 2:
                    g_txt = nums[0].get_text().strip()
                    n_txt = nums[1].get_text().strip()
                    if g_txt.isdigit(): horse['gate'] = int(g_txt)
                    if n_txt.isdigit(): horse['horse_no'] = int(n_txt)
                else:
                    g = row.select_one('span.Waku, td.Waku')
                    n = row.select_one('td.Umaban, span.Umaban')
                    if g and g.get_text().strip().isdigit(): horse['gate'] = int(g.get_text().strip())
                    if n and n.get_text().strip().isdigit(): horse['horse_no'] = int(n.get_text().strip())
                
                h_link = row.select_one('a[href*="/horse/"]')
                if h_link:
                    horse['horse_name'] = h_link.get_text().strip()
                    m = re.search(r'horse/(\w+)', h_link.get('href', ''))
                    if m: horse['horse_id'] = m.group(1)
                
                sa = row.select_one('td.Barei, span.Barei')
                if sa:
                    sa_txt = sa.get_text().strip()
                    if len(sa_txt) >= 2:
                        horse['sex'] = sa_txt[0]
                        digits = ''.join(filter(str.isdigit, sa_txt))
                        if digits: horse['age'] = int(digits)
                
                w = row.select_one('td.Kinryo, span.Kinryo')
                if w:
                    m = re.search(r'(\d+\.?\d*)', w.get_text())
                    if m: horse['weight_carried'] = float(m.group(1))
                
                j = row.select_one('a[href*="/jockey/"]')
                if j:
                    horse['jockey_name'] = j.get_text().strip()
                    m = re.search(r'jockey/(\w+)', j.get('href', ''))
                    if m: horse['jockey_id'] = m.group(1)
                
                t = row.select_one('a[href*="/trainer/"]')
                if t:
                    horse['trainer_name'] = t.get_text().strip()
                    m = re.search(r'trainer/(\w+)', t.get('href', ''))
                    if m: horse['trainer_id'] = m.group(1)
                
                hw = row.select_one('td.Weight')
                if hw:
                    hw_txt = hw.get_text().strip()
                    m = re.search(r'(\d+)', hw_txt)
                    if m: horse['horse_weight'] = int(m.group(1))
                    m_diff = re.search(r'\(([+-]?\d+)\)', hw_txt)
                    if m_diff: horse['horse_weight_diff'] = int(m_diff.group(1))

                required = ['finish_position', 'horse_no', 'horse_id']
                if all(k in horse for k in required):
                    results.append(horse)
            except:
                continue
                
        return results if results else None
        
    except Exception as e:
        logger.error(f"  [ERROR] {race_id}: {e}")
        return None

def collect_all_data():
    start_year = 2020
    end_year = 2024
    
    logger.info(f"STARTING FULL COLLECTION v2: {end_year} -> {start_year}")
    
    active_months = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    for year in range(end_year, start_year - 1, -1):
        logger.info(f"\n{'='*40}\n YEAR: {year} \n{'='*40}")
        
        year_data = []
        races_found = 0
        
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        current_date = end_date
        
        while current_date >= start_date:
            if current_date.month in active_months and current_date.weekday() in [0, 1, 2, 6]:
                
                mmdd = current_date.strftime('%m%d')
                race1_id = f"{year}46{mmdd}01"
                
                # logger.info(f"Checking {current_date.strftime('%Y-%m-%d')}...")
                
                data = scrape_race(race1_id)
                if data:
                    logger.info(f"  [FOUND] {current_date.strftime('%Y-%m-%d')}: Scraping...")
                    year_data.extend(data)
                    races_found += 1
                    
                    for r_num in range(2, 13):
                        rid = f"{year}46{mmdd}{r_num:02d}"
                        r_data = scrape_race(rid)
                        if r_data:
                            year_data.extend(r_data)
                            races_found += 1
                            print(".", end="", flush=True)
                        else:
                            break
                    print()
                    
                    if races_found % 50 == 0:
                        temp_save(year_data, year)
                else:
                    # Silent skip to avoid clutter, but maybe print a dot?
                    # print("x", end="", flush=True)
                    pass
            
            current_date -= timedelta(days=1)
            
        if year_data:
            save_year(year_data, year)
        else:
            logger.warning(f"[NO DATA] Year {year} yielded 0 records.")

def temp_save(data, year):
    path = Path(f'data/kanazawa_{year}_partial.csv')
    pd.DataFrame(data).to_csv(path, index=False, encoding='utf-8-sig')
    logger.info(f"  (Saved partial: {len(data)} records)")

def save_year(data, year):
    path = Path(f'data/kanazawa_{year}.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False, encoding='utf-8-sig')
    logger.info(f"✓ COMPLETED {year}: Saved {len(data)} records to {path}")

if __name__ == '__main__':
    collect_all_data()
