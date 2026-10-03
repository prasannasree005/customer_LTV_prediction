# Customer Lifetime Value (LTV) Prediction

Predicts each customer's spend over the next 180 days from purchase behaviour (Frequency, Recency, AOV + more), then segments customers for targeted marketing.

## Run
    pip install -r requirements.txt
    jupyter notebook LTV_Prediction.ipynb      # Run All
    # or: python run_pipeline.py

## Structure
| Path | Content |
|---|---|
| LTV_Prediction.ipynb | Full notebook (preprocessing, features, models, validation, segmentation) |
| run_pipeline.py | Same code as a plain script |
| data/ | transactions.csv, customers.csv (synthetic; replace with your own, same columns) |
| models/ltv_model.joblib | Trained model + feature list |
| visualizations/ | 8 PNG charts |
| outputs/ltv_predictions.csv | FINAL per-customer LTV predictions + segment + action |
| outputs/ltv_summary.xlsx | Excel: segment summary, model metrics, predictions |

## Using your own data
Put `transactions.csv` (order_id, customer_id, order_date, category, quantity, unit_price) and `customers.csv` (customer_id, signup_date, channel, region) into `data/`, adjust `TRAIN_CUTOFF` / `SCORE_CUTOFF` in the first cell, and re-run.
