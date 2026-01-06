# Kanazawa 3T - サンプルデータ生成スクリプト

このディレクトリには、システムをテストするためのサンプルデータ生成スクリプトが含まれています。

## 使い方

```bash
python scripts/generate_sample_data.py --output data/sample_races.csv --n-races 100
```

実際のデータを使用する場合は、以下の形式でCSVファイルを準備してください。

## データ形式

### races.csv (レースデータ)

必須カラム:
- race_id: レースID
- date: 開催日 (YYYY-MM-DD)
- distance: 距離 (m)
- surface: 馬場 (芝/ダ)
- track_condition: 馬場状態 (良/稍重/重/不良)
- class: クラス (A/B/C)
- horse_no: 馬番
- gate: 枠番
- sex: 性別
- age: 年齢
- weight_carried: 斤量 (kg)
- jockey_id: 騎手ID
- trainer_id: 調教師ID
- finish_position: 着順

### trifecta_odds.csv (三連単オッズ)

必須カラム:
- race_id: レースID
- first: 1着馬番
- second: 2着馬番
- third: 3着馬番
- odds: オッズ (100円あたりの払戻金)
