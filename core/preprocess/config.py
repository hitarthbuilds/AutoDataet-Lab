# sensible defaults, tweak from your UI or config.toml
DEFAULT_CONFIG = {
    "rare_threshold": 0.01,          # categories < 1% -> 'OTHER'
    "missing_indicator": True,
    "imputer_numeric_strategy": "median",
    "imputer_categorical_strategy": "constant",
    "scaler": "standard",            # "minmax" | "robust" | "standard"
    "target_encode": False,          # target-encoding toggle
    "leakage_corr_threshold": 0.95,  # perfect or near-perfect correlation
    "leakage_mi_threshold": 0.6,     # mutual info threshold (heuristic)
    "max_unique_for_onehot": 20,     # if categorical unique < this -> one-hot
}
