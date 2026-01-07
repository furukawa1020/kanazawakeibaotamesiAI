"""
Download JRA Horse Racing Dataset from Kaggle.
This dataset contains real race data from 1986-2021.
"""
import os
from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_kaggle_dataset():
    """
    Download JRA horse racing dataset from Kaggle.
    
    Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Setup Kaggle API credentials: https://www.kaggle.com/docs/api
       - Create account on Kaggle
       - Go to Account settings -> API -> Create New API Token
       - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)
    """
    # Dataset information
    dataset_name = "nechoppy/jra"
    output_dir = Path("data/jra_raw")
    
    logger.info(f"Downloading dataset: {dataset_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download using kaggle API
        cmd = f"kaggle datasets download -d {dataset_name} -p {output_dir} --unzip"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✓ Dataset downloaded successfully!")
            logger.info(f"Files: {list(output_dir.glob('*'))}")
        else:
            logger.error(f"Error downloading dataset: {result.stderr}")
            logger.info("\nTo use Kaggle API:")
            logger.info("1. pip install kaggle")
            logger.info("2. Create Kaggle account and get API token")
            logger.info("3. Place kaggle.json in ~/.kaggle/ directory")
            logger.info("\nAlternatively, download manually from:")
            logger.info(f"https://www.kaggle.com/datasets/{dataset_name}/download")
            
    except Exception as e:
        logger.error(f"Exception: {e}")
        logger.info("\nManual download instructions:")
        logger.info(f"1. Visit: https://www.kaggle.com/datasets/{dataset_name}")
        logger.info("2. Click 'Download' button")
        logger.info(f"3. Extract to: {output_dir.absolute()}")


if __name__ == '__main__':
    download_kaggle_dataset()
