"""
Scrape Kanazawa horse racing data from netkeiba.
WARNING: Please respect the site's robots.txt and terms of service.
This is a basic example for educational purposes.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetkeibaScraper:
    """Scrape horse racing data from netkeiba."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.base_url = 'https://nar.netkeiba.com'
   
    def scrape_race_list(self, track_code='29', year=2023, month=1):
        """
        Scrape race list for a specific month.
        
        Args:
            track_code: Track code (29 = Kanazawa)
            year: Year
            month: Month
        
        Returns:
            List of race IDs
        """
        url = f"{self.base_url}/race/list.html?jyo={track_code}&year={year}&mon={month}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            race_ids = []
            race_links = soup.select('a[href*="/race/result.html?race_id="]')
            
            for link in race_links:
                href = link.get('href')
                if 'race_id=' in href:
                    race_id = href.split('race_id=')[1].split('&')[0]
                    race_ids.append(race_id)
            
            logger.info(f"Found {len(race_ids)} races for {year}/{month}")
            return race_ids
            
        except Exception as e:
            logger.error(f"Error scraping race list: {e}")
            return []
    
    def scrape_race_result(self, race_id):
        """
        Scrape result for a specific race.
        
        Args:
            race_id: Race ID
        
        Returns:
            DataFrame with race results
        """
        url = f"{self.base_url}/race/result.html?race_id={race_id}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract race info
            race_info = self._parse_race_info(soup, race_id)
            
            # Extract horse info
            results = self._parse_results_table(soup, race_info)
            
            logger.info(f"Scraped race {race_id}: {len(results)} horses")
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error scraping race {race_id}: {e}")
            return pd.DataFrame()
    
    def _parse_race_info(self, soup, race_id):
        """Parse race information from page."""
        info = {'race_id': race_id}
        
        # Distance and surface
        race_data = soup.select_one('.RaceData01')
        if race_data:
            text = race_data.get_text()
            if 'ダ' in text:
                info['surface'] = 'ダ'
            elif '芝' in text:
                info['surface'] = '芝'
            
            # Extract distance
            import re
            distance_match = re.search(r'(\d{4})m', text)
            if distance_match:
                info['distance'] = int(distance_match.group(1))
        
        # Track condition
        track_cond = soup.select_one('.RaceData02')
        if track_cond:
            text = track_cond.get_text()
            if '良' in text:
                info['track_condition'] = '良'
            elif '稍' in text:
                info['track_condition'] = '稍重'
            elif '重' in text:
                info['track_condition'] = '重'
            elif '不良' in text:
                info['track_condition'] = '不良'
        
        # Date
        date_elem = soup.select_one('.RaceData01 span')
        if date_elem:
            date_text = date_elem.get_text()
            # Parse date
            info['date'] = self._parse_date(date_text)
        
        return info
    
    def _parse_date(self, date_text):
        """Parse date from text."""
        import re
        # Example: "2023年1月1日"
        match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', date_text)
        if match:
            year, month, day = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
        return None
    
    def _parse_results_table(self, soup, race_info):
        """Parse results table."""
        results = []
        
        result_table = soup.select('table.ResultTableWrap tr.HorseList')
        
        for row in result_table:
            horse_data = race_info.copy()
            
            # Finish position
            finish = row.select_one('.Num')
            if finish:
                horse_data['finish_position'] = int(finish.get_text().strip())
            
            # Horse number
            horse_num = row.select_one('.Horse_Num')
            if horse_num:
                horse_data['horse_no'] = int(horse_num.get_text().strip())
            
            # Gate
            gate = row.select_one('.Waku')
            if gate:
                horse_data['gate'] = int(gate.get_text().strip())
            
            # Horse name (can extract ID if needed)
            horse_name = row.select_one('.Horse_Name a')
            if horse_name:
                horse_data['horse_name'] = horse_name.get_text().strip()
                href = horse_name.get('href', '')
                if 'horse_id=' in href:
                    horse_data['horse_id'] = href.split('horse_id=')[1].split('&')[0]
            
            # Sex and Age
            sex_age = row.select_one('.SexAge')
            if sex_age:
                text = sex_age.get_text().strip()
                if len(text) >= 2:
                    horse_data['sex'] = text[0]
                    horse_data['age'] = int(text[1]) if text[1].isdigit() else 0
            
            # Weight
            weight = row.select_one('.Weight')
            if weight:
                weight_text = weight.get_text().strip()
                import re
                match = re.search(r'(\d+\.?\d*)', weight_text)
                if match:
                    horse_data['weight_carried'] = float(match.group(1))
            
            # Jockey
            jockey = row.select_one('.Jockey a')
            if jockey:
                horse_data['jockey_name'] = jockey.get_text().strip()
                href = jockey.get('href', '')
                if 'jockey_id=' in href:
                    horse_data['jockey_id'] = href.split('jockey_id=')[1].split('&')[0]
            
            # Trainer
            trainer = row.select_one('.Trainer a')
            if trainer:
                horse_data['trainer_name'] = trainer.get_text().strip()
                href = trainer.get('href', '')
                if 'trainer_id=' in href:
                    horse_data['trainer_id'] = href.split('trainer_id=')[1].split('&')[0]
            
            results.append(horse_data)
        
        return results


def main():
    """Main scraping function."""
    scraper = NetkeibaScraper()
    
    all_data = []
    
    # Scrape data for 2023 (example)
    for month in range(1, 13):
        logger.info(f"Scraping 2023/{month}...")
        
        race_ids = scraper.scrape_race_list(track_code='29', year=2023, month=month)
        
        for race_id in race_ids[:10]:  # Limit to first 10 races per month for demo
            df = scraper.scrape_race_result(race_id)
            if not df.empty:
                all_data.append(df)
            
            # Be polite - wait between requests
            time.sleep(1)
        
        # Wait between months
        time.sleep(2)
    
    # Combine all data
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Save to CSV
        output_path = Path('data/kanazawa_races_2023.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"Saved {len(final_df)} records to {output_path}")
        logger.info(f"Columns: {final_df.columns.tolist()}")
    else:
        logger.warning("No data collected")


if __name__ == '__main__':
    main()
