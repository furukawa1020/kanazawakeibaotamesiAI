"""
Direct approach to collect Kanazawa race data.
Updated with correct Kanazawa track code (46) and active season logic.
OPTIMIZED: Collects in REVERSE order (2024 -> 2020) to get data faster.
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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Add delay to be polite
        time.sleep(1.0)
        
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check if race exists by looking for RaceName
        race_name_elem = soup.select_one('.RaceName')
        if not race_name_elem:
            return None
            
        race_name = race_name_elem.get_text().strip()
        if not race_name:
            return None
        
        race_info = {'race_id': race_id, 'race_name': race_name}
        
        # Extract date from race_id (YYYY + 46 + MMDD + RR)
        # ID: 202446040101 -> 2024-04-01
        try:
            date_part = race_id[6:10] # MMDD
            year_part = race_id[:4]
            race_info['date'] = f"{year_part}-{date_part[:2]}-{date_part[2:]}"
        except:
            pass
        
        # Surface and distance
        data1 = soup.select_one('div.RaceData01')
        if data1:
            text = data1.get_text()
            race_info['surface'] = 'ダ' if 'ダ' in text else '芝'
            m = re.search(r'(\d{3,4})m', text)
            if m:
                race_info['distance'] = int(m.group(1))
        
        # Track condition
        data2 = soup.select_one('div.RaceData02')
        if data2:
            text = data2.get_text()
            if '良' in text:
                race_info['track_condition'] = '良'
            elif '稍' in text:
                race_info['track_condition'] = '稍重'
            elif '重' in text and '稍' not in text:
                race_info['track_condition'] = '重'
            elif '不' in text:
                race_info['track_condition'] = '不良'
        else:
            # Try finding track condition in span
            spans = soup.select('span')
            for s in spans:
                if '天候' in s.get_text():
                    text = s.get_text()
                    if '良' in text: race_info['track_condition'] = '良'
                    elif '稍' in text: race_info['track_condition'] = '稍重'
                    elif '重' in text: race_info['track_condition'] = '重'
                    elif '不' in text: race_info['track_condition'] = '不良'
        
        race_info['class'] = 'C' # Default
        if 'A' in race_name: race_info['class'] = 'A'
        elif 'B' in race_name: race_info['class'] = 'B'
        
        # Parse results
        results = []
        rows = soup.select('tr')
        
        for row in rows:
            try:
                # Look for finish position
                finish = row.select_one('span.Num, td.Num')
                if not finish:
                    continue
                    
                finish_text = finish.get_text().strip()
                if not finish_text.isdigit():
                    continue
                
                horse = race_info.copy()
                horse['finish_position'] = int(finish_text)
                
                # Gate
                gate = row.select_one('span.Waku, td.Waku')
                if gate and gate.get_text().strip().isdigit():
                    horse['gate'] = int(gate.get_text().strip())
                
                # Horse number
                num = row.select_one('td.Umaban, span.Umaban')
                if num and num.get_text().strip().isdigit():
                    horse['horse_no'] = int(num.get_text().strip())
                
                # Horse ID & Name
                h_link = row.select_one('a[href*="/horse/"]')
                if h_link:
                    horse['horse_name'] = h_link.get_text().strip()
                    m = re.search(r'horse/(\w+)', h_link.get('href', ''))
                    if m:
                        horse['horse_id'] = m.group(1)
                
                # Sex/Age
                sa = row.select_one('td.Barei, span.Barei')
                if sa:
                    sa_text = sa.get_text().strip()
                    if len(sa_text) >= 2:
                        horse['sex'] = sa_text[0]
                        # Extract all digits
                        digits = ''.join(filter(str.isdigit, sa_text))
                        if digits:
                            horse['age'] = int(digits)
                
                # Weight carried
                w = row.select_one('td.Kinryo, span.Kinryo')
                if w:
                    m = re.search(r'(\d+\.?\d*)', w.get_text())
                    if m:
                        horse['weight_carried'] = float(m.group(1))
                
                # Jockey
                j = row.select_one('a[href*="/jockey/"]')
                if j:
                    horse['jockey_name'] = j.get_text().strip()
                    m = re.search(r'jockey/(\w+)', j.get('href', ''))
                    if m:
                        horse['jockey_id'] = m.group(1)
                
                # Trainer
                t = row.select_one('a[href*="/trainer/"]')
                if t:
                    horse['trainer_name'] = t.get_text().strip()
                    m = re.search(r'trainer/(\w+)', t.get('href', ''))
                    if m:
                        horse['trainer_id'] = m.group(1)
                
                # H Weight
                hw = row.select_one('td.Weight')
                if hw:
                    hw_text = hw.get_text().strip()
                    m = re.search(r'(\d+)', hw_text)
                    if m:
                        horse['horse_weight'] = int(m.group(1))
                    
                    m_diff = re.search(r'\(([+-]?\d+)\)', hw_text)
                    if m_diff:
                        horse['horse_weight_diff'] = int(m_diff.group(1))

                if all(k in horse for k in ['finish_position', 'horse_no']):
                    results.append(horse)
            except Exception as e:
                continue
        
        return results if results else None
        
    except Exception as e:
        logger.error(f"Error scraping {race_id}: {e}")
        return None


def collect_kanazawa_data(start_year=2020, end_year=2024):
    """Collect Kanazawa data systematically."""
    logger.info(f"Starting collection: {end_year} -> {start_year} (Reverse Order)")
    
    all_data = []
    
    # Kanazawa active season
    active_months = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    # Active weekdays: Sun(6), Mon(0), Tue(1) + Try Sun-Tue range
    active_days = [0, 1, 2, 6] # Sun, Mon, Tue, Wed just in case
    
    # REVERSE ORDER LOOP
    for year in range(end_year, start_year - 1, -1):
        logger.info(f"\n{'='*60}")
        logger.info(f"YEAR: {year}")
        logger.info(f"{'='*60}")
        
        year_data = []
        races_count = 0
        
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        
        # Iterate dates backwards too, to get most recent first
        current_date = end_date
        
        while current_date >= start_date:
            if current_date.month in active_months and current_date.weekday() in active_days:
                
                date_str = current_date.strftime('%m%d')
                check_id = f"{year}46{date_str}01"
                
                logger.info(f"Checking {current_date.strftime('%Y-%m-%d')}...")
                data = scrape_race(check_id)
                
                if data:
                    logger.info(f"  FOUND RACE DAY! Scraping races 1-12...")
                    all_data.extend(data)
                    year_data.extend(data)
                    races_count += 1
                    
                    for race_num in range(2, 13):
                        race_id = f"{year}46{date_str}{race_num:02d}"
                        r_data = scrape_race(race_id)
                        if r_data:
                            all_data.extend(r_data)
                            year_data.extend(r_data)
                            races_count += 1
                            print(".", end="", flush=True)
                        else:
                            break 
                    print()
                    
                    # Save incremental progress (every race day)
                    if races_count % 5 == 0:
                         output_file = Path(f'data/kanazawa_{year}_partial.csv')
                         pd.DataFrame(year_data).to_csv(output_file, index=False, encoding='utf-8-sig')
                         
                else:
                    pass
            
            else:
                pass
                
            current_date -= timedelta(days=1)
            
        # Save yearly file
        if year_data:
            output_file = Path(f'data/kanazawa_{year}.csv')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df_year = pd.DataFrame(year_data)
            df_year.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Saved {year}: {len(year_data)} records")
        else:
            logger.warning(f"No data found for {year}")
    
    # Save combined file
    if all_data:
        final_df = pd.DataFrame(all_data)
        output_file = Path(f'data/kanazawa_{start_year}_{end_year}.csv')
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        return final_df
    else:
        return pd.DataFrame()


if __name__ == '__main__':
    logger.info("Kanazawa Data Collection - REVERSE MODE")
    logger.info("Starting from Dec 2024 backwards...")
    
    # Collect 2020-2024
    df = collect_kanazawa_data(2020, 2024)
