# 🏇 Kanazawa 3T - 競馬予想AI

**今の技術で、地方競馬で、個人でどこまでできるのか！**

人類の夢――競馬の未来を予測すること。  
かつてはプロの馬券師や大手企業だけが持ちえた高度な予測技術を、最新のAI・機械学習技術で個人が実現する。

**Kanazawa 3T**は、金沢競馬の過去データから学習し、三連単の買い目を**期待値ベース**で自動生成する、完全オープンソースの競馬予測システムです。

##  このプロジェクトが目指すもの

- **個人でも最先端のAI予測を**: LightGBM Ranker + GPU加速モンテカルロサンプリング
- **透明性と再現性**: 全工程をコード化し、誰でも検証・改良可能
- **地方競馬の可能性？？**: 金沢競馬で、個人がどこまで勝てるのか挑戦

## Features

- **Learning-to-Rank Model**: LightGBM ranker trained on historical race data
- **Probability Estimation**: Plackett-Luce model with GPU-accelerated Monte Carlo sampling
- **EV-Based Betting**: Automatic generation of up to 30 trifecta bets with optimal stake allocation
- **Backtesting Framework**: Time-series validation with comprehensive performance metrics
- **Leak Prevention**: Strict temporal validation to prevent data leakage

## Installation

### Using Docker (Recommended)

```bash
cd docker
docker-compose up -d
docker-compose exec keibaai bash
```

### Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train Model

```bash
python src/cli.py train \
  --data-path data/races.csv \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --output models/ranker_v1
```

### 2. Generate Predictions

```bash
python src/cli.py predict \
  --model models/ranker_v1 \
  --race-file data/upcoming_race.csv \
  --odds-file data/trifecta_odds.csv \
  --output predictions/race_123.json
```

### 3. Run Backtest

```bash
python src/cli.py backtest \
  --model models/ranker_v1 \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output results/backtest_2024.json
```

## Data Format

### Race Data (races.csv)

Required columns:
- `race_id`: Unique race identifier
- `date`: Race date (YYYY-MM-DD)
- `distance`: Distance in meters
- `surface`: Track surface (芝/ダ)
- `track_condition`: Track condition (良/稍重/重/不良)
- `class`: Race class (A/B/C)
- `horse_no`: Horse number in race
- `gate`: Gate number
- `sex`: Horse sex
- `age`: Horse age
- `weight_carried`: Weight carried (kg)
- `jockey_id`: Jockey identifier
- `trainer_id`: Trainer identifier
- `finish_position`: Final position (1-N)

### Odds Data (trifecta_odds.csv)

Required columns:
- `race_id`: Race identifier
- `first`: First place horse number
- `second`: Second place horse number
- `third`: Third place horse number
- `odds`: Trifecta odds (payout per 100 yen)

## Configuration

Edit `configs/default.yaml` to customize:

- Model hyperparameters
- Feature engineering settings
- Betting strategy parameters (EV threshold, max bets, budget)
- Evaluation metrics

## Project Structure

```
keibaai/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model training and calibration
│   ├── inference/      # Scoring and probability estimation
│   ├── betting/        # Bet generation and allocation
│   ├── evaluation/     # Metrics and backtesting
│   └── cli.py          # Command-line interface
├── configs/            # Configuration files
├── tests/              # Unit tests
├── docker/             # Docker configuration
└── notebooks/          # Jupyter notebooks for exploration
```

## Performance Metrics

The system tracks:

- **Prediction Quality**: Brier score, calibration curves, NDCG
- **Betting Performance**: ROI, expected value, maximum drawdown
- **Coverage**: Percentage of races with generated bets
- **Baseline Comparison**: Performance vs. popularity/jockey/last-race strategies

## License

MIT License

## Acknowledgments

Built for Kanazawa horse racing analysis and prediction.
