"""
Scrape real Kanazawa horse racing data from netkeiba.
This collects actual race results for the Kanazawa racecourse.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import logging
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KanazawaDataCollector:
    """Collect real Kanazawa horse racing data from netkeiba."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        # Kanazawa track code is 29 on NAR netkeiba
        self.track_code = '29'
        self.base_url = 'https://nar.netkeiba.com'
    
    def get_race_calendar(self, year=2023, month=1):
        """Get list of race days for Kanazawa in a specific month."""
        url = f'{self.base_url}/top/calendar.html?year={year}&month={month}'
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            race_days = []
            
            # Look for Kanazawa race days in calendar
            calendar_cells = soup.select('td.RaceCellBox')
            for cell in calendar_cells:
                if '金沢' in cell.get_text():
                    day_link = cell.select_one('a')
                    if day_link:
                        race_days.append({
                            'url': self.base_url + day_link.get('href'),
                            'text': cell.get_text().strip()
                        })
            
            logger.info(f"Found {len(race_days)} Kanazawa race days in {year}/{month}")
            return race_days
            
        except Exception as e:
            logger.error(f"Error getting calendar for {year}/{month}: {e}")
            return []
    
    def get_races_from_day(self, day_url):
        """Get all race IDs from a specific race day."""
        try:
            response = self.session.get(day_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            race_ids = []
            
            # Find race result links
            race_links = soup.select('a[href*="race/result.html?race_id="]')
            for link in race_links:
                href = link.get('href', '')
                match = re.search(r'race_id=(\d+)', href)
                if match:
                    race_ids.append(match.group(1))
            
            logger.info(f"Found {len(race_ids)} races for this day")
            return list(set(race_ids))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error getting races from day: {e}")
            return []
    
    def scrape_race_result(self, race_id):
        """Scrape detailed results for a specific race."""
        url = f'{self.base_url}/race/result.html?race_id={race_id}'
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse race information
            race_info = self._parse_race_header(soup, race_id)
            
            # Parse results table
            results = self._parse_results_table(soup, race_info)
            
            if results:
                logger.info(f"✓ Race {race_id}: {len(results)} horses")
                return pd.DataFrame(results)
            else:
                logger.warning(f"No results found for race {race_id}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error scraping race {race_id}: {e}")
            return pd.DataFrame()
    
    def _parse_race_header(self, soup, race_id):
        """Extract race metadata from page header."""
        info = {'race_id': race_id}
        
        # Race title and conditions
        title_elem = soup.select_one('.RaceName')
        if title_elem:
            info['race_name'] = title_elem.get_text().strip()
        
        # Race data (distance, surface, etc.)
        race_data = soup.select_one('div.RaceData01')
        if race_data:
            text = race_data.get_text()
            
            # Surface
            if 'ダート' in text or 'ダ' in text:
                info['surface'] = 'ダ'
            elif '芝' in text:
                info['surface'] = '芝'
            else:
                info['surface'] = 'ダ'  # Default for Kanazawa
            
            # Distance
            distance_match = re.search(r'(\d{3,4})m', text)
            if distance_match:
                info['distance'] = int(distance_match.group(1))
        
        # Weather and track condition
        race_data2 = soup.select_one('div.RaceData02')
        if race_data2:
            text = race_data2.get_text()
            
            # Track condition
            if '良' in text:
                info['track_condition'] = '良'
            elif '稍' in text or '稍重' in text:
                info['track_condition'] = '稍重'
            elif '重' in text and '稍' not in text:
                info['track_condition'] = '重'
            elif '不' in text or '不良' in text:
                info['track_condition'] = '不良'
        
        # Date
        date_elem = soup.select_one('.RaceData01')
        if date_elem:
            date_text = date_elem.get_text()
            date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', date_text)
            if date_match:
                year, month, day = date_match.groups()
                info['date'] = f"{year}-{int(month):02d}-{int(day):02d}"
        
        # Race class (try to extract from title)
        if 'race_name' in info:
            race_name = info['race_name']
            if 'A' in race_name:
                info['class'] = 'A'
            elif 'B' in race_name:
                info['class'] = 'B'
            elif 'C' in race_name:
                info['class'] = 'C'
            else:
                info['class'] = 'C'  # Default
        
        return info
    
    def _parse_results_table(self, soup, race_info):
        """Parse the results table to extract horse data."""
        results = []
        
        # Find result rows
        result_rows = soup.select('table.RaceTable01 tr.HorseList')
        
        for row in result_rows:
            horse_data = race_info.copy()
            
            try:
                # Finish position
                finish_elem = row.select_one('td.Num span')
                if finish_elem:
                    finish_text = finish_elem.get_text().strip()
                    try:
                        horse_data['finish_position'] = int(finish_text)
                    except:
                        continue
                
                # Frame (Waku)
                frame_elem = row.select_one('td.Waku span')
                if frame_elem:
                    horse_data['gate'] = int(frame_elem.get_text().strip())
                
                # Horse number (Umaban)
                num_elem = row.select_one('td.Num.Umaban')
                if num_elem:
                    horse_data['horse_no'] = int(num_elem.get_text().strip())
                
                # Horse name and ID
                horse_name_elem = row.select_one('td.Horse span.Horse_Name a')
                if horse_name_elem:
                    horse_data['horse_name'] = horse_name_elem.get_text().strip()
                    href = horse_name_elem.get('href', '')
                    horse_id_match = re.search(r'horse/(\w+)/', href)
                    if horse_id_match:
                        horse_data['horse_id'] = horse_id_match.group(1)
                
                # Sex and Age
                sex_age_elem = row.select_one('td.Barei.Txt_C')
                if sex_age_elem:
                    sex_age = sex_age_elem.get_text().strip()
                    if len(sex_age) >= 2:
                        horse_data['sex'] = sex_age[0]
                        try:
                            horse_data['age'] = int(sex_age[1])
                        except:
                            pass
                
                # Weight carried (Kinryo)
                weight_elem = row.select_one('td.Txt_C.Kinryo')
                if weight_elem:
                    weight_text = weight_elem.get_text().strip()
                    weight_match = re.search(r'(\d+\.?\d*)', weight_text)
                    if weight_match:
                        horse_data['weight_carried'] = float(weight_match.group(1))
                
                # Jockey
                jockey_elem = row.select_one('td.Jockey a')
                if jockey_elem:
                    horse_data['jockey_name'] = jockey_elem.get_text().strip()
                    href = jockey_elem.get('href', '')
                    jockey_id_match = re.search(r'jockey/(\w+)/', href)
                    if jockey_id_match:
                        horse_data['jockey_id'] = jockey_id_match.group(1)
                
                # Trainer
                trainer_elem = row.select_one('td.Trainer a')
                if trainer_elem:
                    horse_data['trainer_name'] = trainer_elem.get_text().strip()
                    href = trainer_elem.get('href', '')
                    trainer_id_match = re.search(r'trainer/(\w+)/', href)
                    if trainer_id_match:
                        horse_data['trainer_id'] = trainer_id_match.group(1)
                
                # Horse weight
                horse_weight_elem = row.select_one('td.Weight')
                if horse_weight_elem:
                    weight_text = horse_weight_elem.get_text().strip()
                    # Format: "450(+2)" or "450(-3)"
                    match = re.search(r'(\d+)\(([+-]?\d+)\)', weight_text)
                    if match:
                        horse_data['horse_weight'] = int(match.group(1))
                        horse_data['horse_weight_diff'] = int(match.group(2))
                
                results.append(horse_data)
                
            except Exception as e:
                logger.warning(f"Error parsing row: {e}")
                continue
        
        return results
    
    def collect_data(self, year=2023, start_month=1, end_month=12, max_days_per_month=None):
        """
        Collect Kanazawa race data for a specific period.
        
        Args:
            year: Year to collect data for
            start_month: Starting month
            end_month: Ending month
            max_days_per_month: Limit number of race days per month (for testing)
        """
        all_data = []
        
        for month in range(start_month, end_month + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Collecting data for {year}/{month:02d}")
            logger.info(f"{'='*60}")
            
            # Get race calendar for this month
            race_days = self.get_race_calendar(year, month)
            
            if max_days_per_month:
                race_days = race_days[:max_days_per_month]
            
            for idx, day_info in enumerate(race_days, 1):
                logger.info(f"\nDay {idx}/{len(race_days)}: {day_info['text']}")
                
                # Get races for this day
                race_ids = self.get_races_from_day(day_info['url'])
                
                for race_id in race_ids:
                    df = self.scrape_race_result(race_id)
                    if not df.empty:
                        all_data.append(df)
                    
                    # Be polite - wait between requests
                    time.sleep(2)
                
                # Wait between days
                time.sleep(3)
            
            # Wait between months
            time.sleep(5)
        
        # Combine all data
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            
            # Save to CSV
            output_path = Path(f'data/kanazawa_races_{year}.csv')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✓ COLLECTION COMPLETE")
            logger.info(f"{'='*60}")
            logger.info(f"Total records: {len(final_df)}")
            logger.info(f"Total races: {final_df['race_id'].nunique()}")
            logger.info(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
            logger.info(f"Output file: {output_path.absolute()}")
            logger.info(f"\nColumns: {', '.join(final_df.columns.tolist())}")
            
            return final_df
        else:
            logger.warning("No data collected!")
            return pd.DataFrame()


if __name__ == '__main__':
    collector = KanazawaDataCollector()
    
    # Collect 2023 data (limit to first 2 race days per month for testing)
    logger.info("Starting Kanazawa data collection...")
    logger.info("Collecting 2023 data with limit for testing")
    
    df = collector.collect_data(
        year=2023,
        start_month=1,
        end_month=3,  # First 3 months for testing
        max_days_per_month=2  # Limit to 2 race days per month
    )
    
    if not df.empty:
        print(f"\n✓ Sample data collected: {len(df)} records")
        print(f"\nFirst few rows:")
        print(df.head(10))
