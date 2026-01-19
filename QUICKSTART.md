# Kanazawa 3T - Quick Start Guide

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. サンプルデータの生成

```bash
python scripts/generate_sample_data.py --n-races 200 --output-dir data
```

## 使い方

### 学習

```bash
python src/cli.py train \
  --data-path data/sample_races.csv \
  --start-date 2023-01-01 \
  --end-date 2023-06-30 \
  --model-name ranker_v1
```

### 予測

```bash
python src/cli.py predict \
  --model-name ranker_v1 \
  --race-file data/test_race.csv \
  --odds-file data/test_odds.csv \
  --output predictions/race_001.json
```

### バックテスト

```bash
python src/cli.py backtest \
  --model-name ranker_v1 \
  --data-path data/sample_races.csv \
  --odds-path data/sample_odds.csv \
  --start-date 2023-07-01 \
  --end-date 2023-12-31 \
  --compare-baselines
```

## GPU設定

RTX 5060を最大限活用するため、以下の設定を確認してください：

1. CUDA Toolkitがインストールされていること
2. PyTorchがCUDA対応でインストールされていること

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 設定のカスタマイズ

`configs/default.yaml`を編集して、以下をカスタマイズできます：

- モデルのハイパーパラメータ
- 特徴量エンジニアリングの設定
- 買い目生成のパラメータ（EV閾値、最大点数、予算など）
- GPU設定

## トラブルシューティング

### GPU が認識されない

```bash
# CUDA バージョン確認
nvidia-smi

# PyTorch 再インストール
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### メモリ不足

`configs/default.yaml`で以下を調整：
- `inference.probability.n_samples`: サンプル数を減らす（50000 → 10000）
- `betting.top_k_horses`: 対象馬を減らす（7 → 5）
