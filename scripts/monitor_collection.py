"""
Monitor data collection progress and report.
"""
import time
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

def monitor_progress():
    """Monitor and report collection progress."""
    print("="*70)
    print("KANAZAWA DATA COLLECTION MONITOR")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    data_dir = Path('data')
    log_file = Path('data_collection.log')
    
    iteration = 0
    last_size = 0
    
    while True:
        iteration += 1
        print(f"\n[Check #{iteration}] {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 70)
        
        # Check collected files
        csv_files = list(data_dir.glob('kanazawa_*.csv'))
        
        if csv_files:
            print(f"\nCollected files: {len(csv_files)}")
            total_records = 0
            
            for f in sorted(csv_files):
                try:
                    df = pd.read_csv(f)
                    records = len(df)
                    total_records += records
                    races = df['race_id'].nunique() if 'race_id' in df.columns else 0
                    print(f"  {f.name}: {records} records, {races} races")
                except:
                    print(f"  {f.name}: {os.path.getsize(f)} bytes")
            
            print(f"\nTotal records so far: {total_records}")
            
            if total_records > last_size:
                print(f"New data collected: +{total_records - last_size} records")
                last_size = total_records
        else:
            print("No data files yet...")
        
        # Check log file
        if log_file.exists():
            log_size = os.path.getsize(log_file)
            print(f"\nLog file size: {log_size:,} bytes")
            
            # Show last few lines
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        print("\nLast log entries:")
                        for line in lines[-5:]:
                            print(f"  {line.rstrip()}")
            except:
                pass
        
        print("\nWaiting 2 minutes before next check...")
        print("(Press Ctrl+C to stop monitoring)")
        
        try:
            time.sleep(120)  # Check every 2 minutes
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            break

if __name__ == '__main__':
    try:
        monitor_progress()
    except Exception as e:
        print(f"Error: {e}")
