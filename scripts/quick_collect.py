"""
Quick script to collect minimal Kanazawa data for demonstration.
This collects just a few races to get the system working.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_collect():
    """Quickly collect some Kanazawa race data."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    # Try to get recent Kanazawa races from the schedule page
    # Kanazawa track code is 29
    base_url = 'https://nar.netkeiba.com'
    
    all_races = []
    
    # Try a few known race IDs from 2024 (format: YYYYMMDD + track code + race number)
    # Example race IDs for Kanazawa (29)
    sample_race_ids = [
        '202401072901',  # 2024/01/07 Kanazawa R1
        '202401072902',
        '202401072903',
        '202401142901',  # 2024/01/14 Kanazawa R1
        '202401142902',
        '202401212901',  # 2024/01/21 Kanazawa R1
    ]
    
    for race_id in sample_race_ids:
        url = f'{base_url}/race/result.html?race_id={race_id}'
        
        try:
            logger.info(f"Trying race {race_id}...")
            response = session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Check if race exists
                race_name = soup.select_one('.RaceName')
                if not race_name:
                    logger.warning(f"Race {race_id} not found")
                    continue
                
                logger.info(f"✓ Found: {race_name.get_text().strip()}")
                
                # Parse race info
                race_info = {'race_id': race_id}
                
                # Extract basic info
                race_data = soup.select_one('div.RaceData01')
                if race_data:
                    text = race_data.get_text()
                    if 'ダ' in text:
                        race_info['surface'] = 'ダ'
                    match = re.search(r'(\d{3,4})m', text)
                    if match:
                        race_info['distance'] = int(match.group(1))
                    
                    # Date
                    date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
                    if date_match:
                        y, m, d = date_match.groups()
                        race_info['date'] = f"{y}-{int(m):02d}-{int(d):02d}"
                
                # Track condition
                race_data2 = soup.select_one('div.RaceData02')
                if race_data2:
                    text = race_data2.get_text()
                    if '良' in text:
                        race_info['track_condition'] = '良'
                    elif '稍' in text:
                        race_info['track_condition'] = '稍重'
                
                race_info['class'] = 'C'  # Default
                
                # Parse results
                results = []
                rows = soup.select('table.RaceTable01 tr.HorseList')
                
                for row in rows:
                    try:
                        horse = race_info.copy()
                        
                        # Finish
                        finish = row.select_one('td.Num span')
                        if finish:
                            horse['finish_position'] = int(finish.get_text().strip())
                        
                        # Gate and number
                        gate = row.select_one('td.Waku span')
                        if gate:
                            horse['gate'] = int(gate.get_text().strip())
                        
                        num = row.select_one('td.Num.Umaban')
                        if num:
                            horse['horse_no'] = int(num.get_text().strip())
                        
                        # Sex/Age
                        sex_age = row.select_one('td.Barei.Txt_C')
                        if sex_age:
                            sa = sex_age.get_text().strip()
                            if len(sa) >= 2:
                                horse['sex'] = sa[0]
                                horse['age'] = int(sa[1]) if sa[1].isdigit() else 4
                        
                        # Weight
                        weight = row.select_one('td.Txt_C.Kinryo')
                        if weight:
                            w = re.search(r'(\d+\.?\d*)', weight.get_text())
                            if w:
                                horse['weight_carried'] = float(w.group(1))
                        
                        # Jockey
                        jockey = row.select_one('td.Jockey a')
                        if jockey:
                            href = jockey.get('href', '')
                            jid = re.search(r'jockey/(\w+)/', href)
                            if jid:
                                horse['jockey_id'] = jid.group(1)
                        
                        # Trainer  
                        trainer = row.select_one('td.Trainer a')
                        if trainer:
                            href = trainer.get('href', '')
                            tid = re.search(r'trainer/(\w+)/', href)
                            if tid:
                                horse['trainer_id'] = tid.group(1)
                        
                        results.append(horse)
                    except:
                        continue
                
                if results:
                    all_races.extend(results)
                    logger.info(f"  → {len(results)} horses")
            
            time.sleep(2)  # Be polite
            
        except Exception as e:
            logger.error(f"Error: {e}")
            continue
    
    if all_races:
        df = pd.DataFrame(all_races)
        df.to_csv('data/kanazawa_quick_sample.csv', index=False, encoding='utf-8-sig')
        logger.info(f"\n✓ Saved {len(df)} records to data/kanazawa_quick_sample.csv")
        return df
    else:
        logger.warning("No data collected")
        return None

if __name__ == '__main__':
    df = quick_collect()
    if df is not None:
        print(f"\nCollected {len(df)} records from {df['race_id'].nunique()} races")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample:\n{df.head()}")
