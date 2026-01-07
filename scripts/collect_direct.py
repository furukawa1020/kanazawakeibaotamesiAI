"""
Direct approach to collect Kanazawa race data.
Uses systematic race ID generation and direct scraping.
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
        logging.FileHandler('data_collection.log', encoding='utf-8'),
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
        
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check if race exists
        if not soup.select_one('.RaceName'):
            return None
        
        race_info = {'race_id': race_id}
        
        # Extract date from race_id (YYYYMMDD...)
        if len(race_id) >= 8:
            race_info['date'] = f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}"
        
        # Surface and distance
        data1 = soup.select_one('div.RaceData01, span.RaceData01')
        if data1:
            text = data1.get_text()
            race_info['surface'] = 'ダ' if 'ダ' in text else '芝'
            m = re.search(r'(\d{3,4})m', text)
            if m:
                race_info['distance'] = int(m.group(1))
        
        # Track condition
        data2 = soup.select_one('div.RaceData02, span.RaceData02')
        if data2:
            text = data2.get_text()
            if '良' in text:
                race_info['track_condition'] = '良'
            elif '稍' in text:
                race_info['track_condition'] = '稍重'
            elif '重' in text and '稍' not in text:
                race_info['track_condition'] = '重'
            else:
                race_info['track_condition'] = '不良'
        
        race_info['class'] = 'C'
        
        # Parse results
        results = []
        rows = soup.select('tr')
        
        for row in rows:
            try:
                # Look for finish position
                finish = row.select_one('span.Num, td.Num span')
                if not finish:
                    continue
                    
                finish_text = finish.get_text().strip()
                if not finish_text.isdigit():
                    continue
                
                horse = race_info.copy()
                horse['finish_position'] = int(finish_text)
                
                # Gate
                gate = row.select_one('span.Waku, td.Waku span')
                if gate and gate.get_text().strip().isdigit():
                    horse['gate'] = int(gate.get_text().strip())
                
                # Horse number
                num = row.select_one('td.Umaban, span.Umaban')
                if num and num.get_text().strip().isdigit():
                    horse['horse_no'] = int(num.get_text().strip())
                
                # Horse ID
                h_link = row.select_one('a[href*="/horse/"]')
                if h_link:
                    m = re.search(r'horse/(\w+)', h_link.get('href', ''))
                    if m:
                        horse['horse_id'] = m.group(1)
                
                # Sex/Age
                sa = row.select_one('td.Barei, span.Barei')
                if sa:
                    sa_text = sa.get_text().strip()
                    if len(sa_text) >= 2:
                        horse['sex'] = sa_text[0]
                        if sa_text[1].isdigit():
                            horse['age'] = int(sa_text[1])
                
                # Weight carried
                w = row.select_one('td.Kinryo, span.Kinryo')
                if w:
                    m = re.search(r'(\d+\.?\d*)', w.get_text())
                    if m:
                        horse['weight_carried'] = float(m.group(1))
                
                # Jockey
                j = row.select_one('a[href*="/jockey/"]')
                if j:
                    m = re.search(r'jockey/(\w+)', j.get('href', ''))
                    if m:
                        horse['jockey_id'] = m.group(1)
                
                # Trainer
                t = row.select_one('a[href*="/trainer/"]')
                if t:
                    m = re.search(r'trainer/(\w+)', t.get('href', ''))
                    if m:
                        horse['trainer_id'] = m.group(1)
                
                if all(k in horse for k in ['finish_position', 'horse_no']):
                    results.append(horse)
            except:
                continue
        
        return results if results else None
        
    except:
        return None


def collect_kanazawa_data(start_year=2020, end_year=2024):
    """Collect Kanazawa data systematically."""
    logger.info(f"Starting collection: {start_year}-{end_year}")
    
    all_data = []
    races_count = 0
    
    # Kanazawa race days are typically Sundays
    for year in range(start_year, end_year + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"YEAR: {year}")
        logger.info(f"{'='*60}")
        
        year_data = []
        
        # Go through all days
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        current_date = start_date
        
        while current_date <= end_date:
            # Check Sundays and Saturdays (main race days)
            if current_date.weekday() in [5, 6]:  # Sat=5, Sun=6
                date_str = current_date.strftime('%Y%m%d')
                
                # Try races 1-12 for this day
                for race_num in range(1, 13):
                    race_id = f"{date_str}29{race_num:02d}"  # 29 = Kanazawa
                    
                    data = scrape_race(race_id)
                    if data:
                        all_data.extend(data)
                        year_data.extend(data)
                        races_count += 1
                        logger.info(f"  OK {race_id}: {len(data)} horses")
                    
                    time.sleep(0.5)  # Be polite
            
            current_date += timedelta(days=1)
        
        # Save yearly file
        if year_data:
            output_file = Path(f'data/kanazawa_{year}.csv')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df_year = pd.DataFrame(year_data)
            df_year.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"Saved {year}: {len(year_data)} records")
    
    # Save combined file
    if all_data:
        final_df = pd.DataFrame(all_data)
        output_file = Path(f'data/kanazawa_{start_year}_{end_year}.csv')
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPLETE!")
        logger.info(f"Total races: {races_count}")
        logger.info(f"Total records: {len(final_df)}")
        logger.info(f"File: {output_file}")
        logger.info(f"{'='*60}")
        
        return final_df
    else:
        logger.error("No data collected")
        return pd.DataFrame()


if __name__ == '__main__':
    logger.info("Kanazawa Data Collection - Direct Method")
    logger.info("This will take 2-3 hours...\n")
    
    df = collect_kanazawa_data(2020, 2024)
    
    if not df.empty:
        print(f"\nSuccess! Collected {len(df)} records from {df['race_id'].nunique()} races")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"\nSample:")
        print(df.head(20))
    else:
        print("\nNo data collected. Check logs.")
