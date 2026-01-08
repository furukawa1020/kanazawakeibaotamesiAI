"""
Direct approach to collect Kanazawa race data.
Updated with correct Kanazawa track code (46).
OPTIMIZED: Focuses specifically on collecting Dec 2024 dates FIRST.
FIXED: Table selectors for NAR result pages (Rank, Gate, HorseNo).
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log', encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def scrape_race(race_id):
    """Scrape a single race directly."""
    url = f'https://nar.netkeiba.com/race/result.html?race_id={race_id}'
    
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        time.sleep(1.0)
        
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            logger.warning(f"  Status {response.status_code} for {race_id}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        race_name_elem = soup.select_one('.RaceName')
        if not race_name_elem:
            return None
            
        race_name = race_name_elem.get_text().strip()
        if not race_name:
            return None
        
        race_info = {'race_id': race_id, 'race_name': race_name}
        
        # Date
        try:
            date_part = race_id[6:10]
            year_part = race_id[:4]
            race_info['date'] = f"{year_part}-{date_part[:2]}-{date_part[2:]}"
        except:
            pass
        
        # Surface/Distance
        data1 = soup.select_one('div.RaceData01')
        if data1:
            text = data1.get_text()
            race_info['surface'] = 'ダ' if 'ダ' in text else '芝'
            m = re.search(r'(\d{3,4})m', text)
            if m: race_info['distance'] = int(m.group(1))
        
        # Condition
        data2 = soup.select_one('div.RaceData02')
        if data2:
            text = data2.get_text()
            if '良' in text: race_info['track_condition'] = '良'
            elif '稍' in text: race_info['track_condition'] = '稍重'
            elif '重' in text: race_info['track_condition'] = '重'
            elif '不' in text: race_info['track_condition'] = '不良'
        else:
            spans = soup.select('span')
            for s in spans:
                if '天候' in s.get_text():
                    text = s.get_text()
                    if '良' in text: race_info['track_condition'] = '良'
                    elif '稍' in text: race_info['track_condition'] = '稍重'
                    elif '重' in text: race_info['track_condition'] = '重'
                    elif '不' in text: race_info['track_condition'] = '不良'
        
        race_info['class'] = 'C'
        if 'A' in race_name: race_info['class'] = 'A'
        elif 'B' in race_name: race_info['class'] = 'B'
        
        results = []
        rows = soup.select('tr')
        
        for row in rows:
            try:
                # Rank: .Rank inside .Result_Num
                rank_elem = row.select_one('.Rank')
                if not rank_elem: continue
                rank_text = rank_elem.get_text().strip()
                if not rank_text.isdigit(): continue
                
                horse = race_info.copy()
                horse['finish_position'] = int(rank_text)
                
                # Gate and HorseNo are usually in td.Num cells
                # First td.Num is Gate, Second is HorseNo
                nums = row.select('td.Num')
                if len(nums) >= 2:
                    gate_text = nums[0].get_text().strip()
                    if gate_text.isdigit(): horse['gate'] = int(gate_text)
                    
                    no_text = nums[1].get_text().strip()
                    if no_text.isdigit(): horse['horse_no'] = int(no_text)
                else:
                    # Fallback selectors
                    gate = row.select_one('span.Waku, td.Waku')
                    if gate and gate.get_text().strip().isdigit():
                        horse['gate'] = int(gate.get_text().strip())
                    
                    num = row.select_one('td.Umaban, span.Umaban')
                    if num and num.get_text().strip().isdigit():
                        horse['horse_no'] = int(num.get_text().strip())
                
                # Other info
                h_link = row.select_one('a[href*="/horse/"]')
                if h_link:
                    horse['horse_name'] = h_link.get_text().strip()
                    m = re.search(r'horse/(\w+)', h_link.get('href', ''))
                    if m: horse['horse_id'] = m.group(1)
                
                sa = row.select_one('td.Barei, span.Barei')
                if sa:
                    sa_text = sa.get_text().strip()
                    if len(sa_text) >= 2:
                        horse['sex'] = sa_text[0]
                        digits = ''.join(filter(str.isdigit, sa_text))
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
                    hw_text = hw.get_text().strip()
                    m = re.search(r'(\d+)', hw_text)
                    if m: horse['horse_weight'] = int(m.group(1))
                    m_diff = re.search(r'\(([+-]?\d+)\)', hw_text)
                    if m_diff: horse['horse_weight_diff'] = int(m_diff.group(1))

                if 'finish_position' in horse and 'horse_no' in horse:
                    results.append(horse)
            except:
                continue
        
        return results if results else None
        
    except Exception as e:
        logger.error(f"Error scraping {race_id}: {e}")
        return None


def collect_kanazawa_data():
    """Collect Kanazawa data, prioritizing confirmed dates."""
    
    known_dates = [
        '20241217', '20241215', '20241210', '20241208', '20241203', '20241202', '20241201'
    ]
    
    logger.info("PHASE 1: Collecting Known Dates (Dec 2024)")
    all_data = []
    
    for date_str in known_dates:
        logger.info(f"Checking Known Date: {date_str}...")
        
        day_data = []
        for race_num in range(1, 13):
            race_id = f"{date_str[:4]}46{date_str[4:]}{race_num:02d}"
            data = scrape_race(race_id)
            if data:
                day_data.extend(data)
                print(".", end="", flush=True)
            else:
                if race_num == 1: pass
                else: break
        print()
        
        if day_data:
            logger.info(f"  SUCCESS! Found {len(day_data)} records for {date_str}")
            all_data.extend(day_data)
            output_file = Path(f'data/kanazawa_2024_dec_partial.csv')
            pd.DataFrame(all_data).to_csv(output_file, index=False, encoding='utf-8-sig')
        else:
            logger.warning(f"  Failed to find data for known date {date_str}")

    logger.info("PHASE 2: Systematic Reverse Collection (2024 -> 2020)")
    # Should implement full loop here for production use, but for now getting Dec 2024 is priority
    
    return pd.DataFrame(all_data)


if __name__ == '__main__':
    logger.info("Kanazawa Data Collection - TARGETED MODE v2 (FIXED SELECTORS)")
    df = collect_kanazawa_data()
