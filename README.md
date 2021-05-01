# Analyzing Handwritten Text to Understand Financial Sentiments/Behaviour

## Usage
1. Install requirements

```bash
pip install -r requirements.txt
```

2. Preprocess the data, currently we only use text and make labels from them for various tasks

```bash
python preprocess_data.py
```

3. Get baselines, we use random and an only neutral baseline.

```bash
python get_baseline.py
```

4. Train or Pre-train your models. The configurations are present as `.yaml` files.

```bash
python train.py -config_dir configs/risk_profiling/bert-base
```

5. Use the notebooks provided for basic EDA, Unsupervised Analysis (Clustering).