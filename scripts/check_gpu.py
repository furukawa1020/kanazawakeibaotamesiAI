import torch
import lightgbm as lgb
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

print("\nChecking LightGBM GPU support...")
try:
    # Quick dummy train to check if 'gpu' param is accepted without error
    # Note: Installing LightGBM with GPU support on Windows can be tricky. 
    # Often the standard pip install is CPU-only. We will check if it runs.
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data.data, data.target
    train_data = lgb.Dataset(X, label=y)
    params = {'device': 'gpu'}
    model = lgb.train(params, train_data, num_boost_round=1)
    print("LightGBM GPU parameter accepted!")
except Exception as e:
    print(f"LightGBM GPU Check Warning: {e}")
    print("Will fall back to CPU for LightGBM if necessary, but PyTorch components will use GPU.")
