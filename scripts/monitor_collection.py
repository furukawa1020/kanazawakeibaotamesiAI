"""
Monitor the data collection progress.
"""
import time
import os
from pathlib import Path
import pandas as pd

def monitor_progress():
    """Monitor and display collection progress."""
    data_dir = Path('data')
    progress_dir = data_dir / 'collection_progress'
    
    print("="*70)
    print("金沢競馬データ収集 - 進捗モニター")
    print("="*70)
    print("\n収集中のファイル:")
    
    while True:
        # Check for collected files
        if data_dir.exists():
            csv_files = list(data_dir.glob('*.csv'))
            if csv_files:
                print(f"\n✓ 完成ファイル ({len(csv_files)}):")
                for f in sorted(csv_files):
                    size = f.stat().st_size / 1024  # KB
                    print(f"  - {f.name}: {size:.1f} KB")
                    
                    # Show record count
                    try:
                        df = pd.read_csv(f)
                        print(f"      レコード数: {len(df)}, レース数: {df['race_id'].nunique()}")
                    except:
                        pass
        
        # Check progress files
        if progress_dir.exists():
            progress_files = list(progress_dir.glob('*.csv'))
            if progress_files:
                print(f"\n🔄 進行中 ({len(progress_files)} 月分):")
                for f in sorted(progress_files)[-5:]:  # Show last 5
                    print(f"  - {f.name}")
        
        print(f"\n最終更新: {time.strftime('%H:%M:%S')}")
        print("Ctrl+C で終了")
        
        time.sleep(30)  # Update every 30 seconds

if __name__ == '__main__':
    try:
        monitor_progress()
    except KeyboardInterrupt:
        print("\n\nモニター終了")
