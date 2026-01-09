"""
Check LightGBM GPU Platforms.
"""
import lightgbm as lgb
from sklearn.datasets import load_iris
import time

def check_lgbm_gpu():
    print("Checking LightGBM GPU connectivity...")
    data = load_iris()
    X, y = data.data, data.target
    train_data = lgb.Dataset(X, label=y)
    
    # Try Platform 0
    print("\n--- Testing Platform 0 ---")
    try:
        params = {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'verbose': 1
        }
        model = lgb.train(params, train_data, num_boost_round=10)
        print("Platform 0: SUCCESS")
    except Exception as e:
        print(f"Platform 0 Failed: {e}")

    # Try Platform 1
    print("\n--- Testing Platform 1 ---")
    try:
        params = {
            'device': 'gpu',
            'gpu_platform_id': 1,
            'gpu_device_id': 0,
            'verbose': 1
        }
        model = lgb.train(params, train_data, num_boost_round=10)
        print("Platform 1: SUCCESS")
    except Exception as e:
        print(f"Platform 1 Failed: {e}")

if __name__ == '__main__':
    check_lgbm_gpu()
