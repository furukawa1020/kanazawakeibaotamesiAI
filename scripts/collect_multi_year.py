"""
Collect multiple years of Kanazawa horse racing data.
Optimized for 3-5 years of historical data (2019-2024).
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import logging
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KanazawaMultiYearCollector:
    """Collect multi-year Kanazawa race data."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.base_url = 'https://nar.netkeiba.com'
        self.track_code = '29'  # Kanazawa
        self.collected_races = set()
    
    def collect_year_data(self, year, save_progress=True):
        """Collect all Kanazawa race data for a specific year."""
        logger.info(f"\n{'='*70}")
        logger.info(f"COLLECTING YEAR: {year}")
        logger.info(f"{'='*70}")
        
        all_data = []
        races_found = 0
        
        for month in range(1, 13):
            logger.info(f"\n--- {year}/{month:02d} ---")
            
            # Get race schedule for this month
            schedule_url = f'{self.base_url}/top/race_list.html?pid=race_list&date={year}{month:02d}01'
            
            try:
                race_ids = self._get_race_ids_from_schedule(schedule_url, year, month)
                logger.info(f"Found {len(race_ids)} potential race IDs for {year}/{month:02d}")
                
                for race_id in race_ids:
                    if race_id in self.collected_races:
                        continue
                    
                    race_data = self._scrape_race(race_id)
                    if race_data:
                        all_data.extend(race_data)
                        self.collected_races.add(race_id)
                        races_found += 1
                        logger.info(f"  ✓ Race {race_id}: {len(race_data)} horses")
                    
                    time.sleep(1.5)  # Be polite
                
            except Exception as e:
                logger.error(f"Error in {year}/{month:02d}: {e}")
            
            # Save progress every month
            if save_progress and all_data:
                self._save_progress(all_data, year, month)
            
            time.sleep(2)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"YEAR {year} COMPLETE: {races_found} races, {len(all_data)} horse records")
        logger.info(f"{'='*70}\n")
        
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()
    
    def _get_race_ids_from_schedule(self, url, year, month):
        """Extract race IDs from the schedule page."""
        race_ids = []
        
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                # Try alternative: direct race list for Kanazawa
                for day in range(1, 32):
                    for race_num in range(1, 13):  # Up to 12 races per day
                        # Race ID format: YYYYMMDD + track(29) + race_num(01-12)
                        race_id = f"{year}{month:02d}{day:02d}{self.track_code}{race_num:02d}"
                        race_ids.append(race_id)
                return race_ids[:100]  # Limit to first 100 IDs per month
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find Kanazawa races
            links = soup.select('a[href*="race/result.html?race_id="]')
            for link in links:
                href = link.get('href', '')
                # Check if it's Kanazawa (contains track code 29)
                if f'{self.track_code}' in href:
                    match = re.search(r'race_id=(\d+)', href)
                    if match:
                        race_ids.append(match.group(1))
            
            return list(set(race_ids))
            
        except Exception as e:
            logger.warning(f"Schedule page error: {e}. Using systematic IDs...")
            # Fallback: generate systematic race IDs
            for day in [7, 14, 21, 28]:  # Common race days (Sundays)
                for race_num in range(1, 13):
                    race_id = f"{year}{month:02d}{day:02d}{self.track_code}{race_num:02d}"
                    race_ids.append(race_id)
            return race_ids
    
    def _scrape_race(self, race_id):
        """Scrape a single race."""
        url = f'{self.base_url}/race/result.html?race_id={race_id}'
        
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if race exists and has results
            race_name_elem = soup.select_one('.RaceName')
            if not race_name_elem or not race_name_elem.get_text().strip():
                return None
            
            # Parse race info
            race_info = {'race_id': race_id}
            
            # Date from race_id (YYYYMMDD...)
            if len(race_id) >= 8:
                date_str = race_id[:8]
                try:
                    race_info['date'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                except:
                    pass
            
            # Surface and distance
            race_data1 = soup.select_one('div.RaceData01')
            if race_data1:
                text = race_data1.get_text()
                race_info['surface'] = 'ダ' if 'ダ' in text else '芝'
                dist_match = re.search(r'(\d{3,4})m', text)
                if dist_match:
                    race_info['distance'] = int(dist_match.group(1))
            
            # Track condition
            race_data2 = soup.select_one('div.RaceData02')
            if race_data2:
                text = race_data2.get_text()
                if '良' in text:
                    race_info['track_condition'] = '良'
                elif '稍' in text:
                    race_info['track_condition'] = '稍重'
                elif '重' in text and '稍' not in text:
                    race_info['track_condition'] = '重'
                elif '不' in text:
                    race_info['track_condition'] = '不良'
            
            # Class (default)
            race_info['class'] = 'C'
            
            # Parse results table
            results = []
            rows = soup.select('table tr')
            
            for row in rows:
                try:
                    # Check if it's a horse result row
                    finish_elem = row.select_one('td.Num span, td span.Num')
                    if not finish_elem:
                        continue
                    
                    horse = race_info.copy()
                    
                    # Finish position
                    finish_text = finish_elem.get_text().strip()
                    if not finish_text.isdigit():
                        continue
                    horse['finish_position'] = int(finish_text)
                    
                    # Gate
                    gate_elem = row.select_one('td.Waku span, span.Waku')
                    if gate_elem:
                        gate_text = gate_elem.get_text().strip()
                        if gate_text.isdigit():
                            horse['gate'] = int(gate_text)
                    
                    # Horse number
                    num_elem = row.select_one('td.Num.Umaban, td.Umaban')
                    if num_elem:
                        num_text = num_elem.get_text().strip()
                        if num_text.isdigit():
                            horse['horse_no'] = int(num_text)
                    
                    # Horse ID
                    horse_link = row.select_one('a[href*="/horse/"]')
                    if horse_link:
                        href = horse_link.get('href', '')
                        hid = re.search(r'horse/(\w+)', href)
                        if hid:
                            horse['horse_id'] = hid.group(1)
                    
                    # Sex and Age
                    sex_age_elem = row.select_one('td.Barei, td[class*="Barei"]')
                    if sex_age_elem:
                        sa_text = sex_age_elem.get_text().strip()
                        if len(sa_text) >= 2:
                            horse['sex'] = sa_text[0]
                            if sa_text[1].isdigit():
                                horse['age'] = int(sa_text[1])
                    
                    # Weight carried
                    weight_elem = row.select_one('td.Kinryo, td[class*="Kinryo"]')
                    if weight_elem:
                        w_match = re.search(r'(\d+\.?\d*)', weight_elem.get_text())
                        if w_match:
                            horse['weight_carried'] = float(w_match.group(1))
                    
                    # Jockey
                    jockey_elem = row.select_one('a[href*="/jockey/"]')
                    if jockey_elem:
                        href = jockey_elem.get('href', '')
                        jid = re.search(r'jockey/(\w+)', href)
                        if jid:
                            horse['jockey_id'] = jid.group(1)
                    
                    # Trainer
                    trainer_elem = row.select_one('a[href*="/trainer/"]')
                    if trainer_elem:
                        href = trainer_elem.get('href', '')
                        tid = re.search(r'trainer/(\w+)', href)
                        if tid:
                            horse['trainer_id'] = tid.group(1)
                    
                    # Only add if we have minimum required fields
                    if all(k in horse for k in ['finish_position', 'horse_no']):
                        results.append(horse)
                
                except Exception as e:
                    continue
            
            return results if results else None
            
        except Exception as e:
            return None
    
    def _save_progress(self, data, year, month):
        """Save progress to file."""
        if not data:
            return
        
        output_dir = Path('data/collection_progress')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(data)
        filename = output_dir / f'kanazawa_{year}_{month:02d}.csv'
        df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    def collect_multi_year(self, start_year=2019, end_year=2024):
        """Collect data for multiple years."""
        logger.info(f"\n{'#'*70}")
        logger.info(f"MULTI-YEAR DATA COLLECTION: {start_year}-{end_year}")
        logger.info(f"{'#'*70}\n")
        
        all_years_data = []
        
        for year in range(start_year, end_year + 1):
            year_df = self.collect_year_data(year)
            if not year_df.empty:
                all_years_data.append(year_df)
                
                # Save yearly file
                output_file = Path(f'data/kanazawa_{year}.csv')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                year_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                logger.info(f"✓ Saved {year} data: {len(year_df)} records → {output_file}")
            
            time.sleep(5)  # Wait between years
        
        # Combine all years
        if all_years_data:
            combined_df = pd.concat(all_years_data, ignore_index=True)
            
            # Save combined file
            final_output = Path('data/kanazawa_2019_2024_combined.csv')
            combined_df.to_csv(final_output, index=False, encoding='utf-8-sig')
            
            logger.info(f"\n{'#'*70}")
            logger.info(f"COLLECTION COMPLETE!")
            logger.info(f"{'#'*70}")
            logger.info(f"Total records: {len(combined_df)}")
            logger.info(f"Total races: {combined_df['race_id'].nunique()}")
            logger.info(f"Years: {combined_df['date'].min()[:4]} - {combined_df['date'].max()[:4]}")
            logger.info(f"Output: {final_output.absolute()}")
            logger.info(f"{'#'*70}\n")
            
            return combined_df
        else:
            logger.error("No data collected!")
            return pd.DataFrame()


if __name__ == '__main__':
    collector = KanazawaMultiYearCollector()
    
    # Collect 2020-2024 (5 years) - start with recent years for faster results
    logger.info("Collecting 5 years of Kanazawa data (2020-2024)...")
    logger.info("This will take some time. Please be patient...")
    
    df = collector.collect_multi_year(start_year=2020, end_year=2024)
    
    if not df.empty:
        print(f"\n✓ Successfully collected {len(df)} records!")
        print(f"\nData summary:")
        print(f"- Races: {df['race_id'].nunique()}")
        print(f"- Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"- Columns: {len(df.columns)}")
        print(f"\nSample:")
        print(df.head(10))
    else:
        print("\n✗ No data collected. Please check the logs for errors.")
