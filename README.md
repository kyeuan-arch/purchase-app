# Online Shopping Purchase Intention Simulator

A Streamlit app that predicts whether an online shopping session will end in a
purchase, using a class-weighted Random Forest compared against Logistic
Regression. Trained on the UCI Online Shoppers Purchasing Intention Dataset.

Live app: purchase-app-cst9l.streamlit.app

## Features

- Simulate a single session and get a purchase probability
- Batch simulation on unseen test sessions
- Model performance at any decision threshold
- Random Forest vs Logistic Regression comparison
- Results with and without `PageValues`, a feature that leaks the outcome
- Feature importance

## Installation

Requires Python 3.9 or later.

```
git clone https://github.com/kyeuan-arch/purchase-app.git
cd purchase-app
python -m pip install -r requirements.txt
```

## Usage

1. Build the model file (only needed once, or after changing `train_model.py`):

```
python train_model.py
```

2. Start the app:

```
python -m streamlit run app.py
```

The app opens in your browser at http://localhost:8501. Press Ctrl+C in the
terminal to stop it.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `train_model.py` | Trains the models and saves `model.joblib` |
| `model.joblib` | Saved models and test-set results used by the app |
| `online_shoppers_intention.csv` | Dataset |
| `requirements.txt` | Python packages |

## Dataset

Sakar, C. O., & Kastro, Y. (2018). Online Shoppers Purchasing Intention
Dataset. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/468

## Authors

Aredondua, Jenny L.
Bautista, Trisha Kate B.
Pagayon, Marion Lou A.
