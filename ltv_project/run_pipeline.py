import matplotlib
matplotlib.use('Agg')

import warnings, json, joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.base import clone

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed -> using sklearn HistGradientBoosting as the boosted model. Run: pip install xgboost")

try:
    from IPython.display import display
except ImportError:
    display = print

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
SEED = 42

# ---- folders ----
DATA_DIR, MODEL_DIR, VIZ_DIR, OUT_DIR = Path("data"), Path("models"), Path("visualizations"), Path("outputs")
for d in (DATA_DIR, MODEL_DIR, VIZ_DIR, OUT_DIR):
    d.mkdir(exist_ok=True)

# ---- project settings ----
TRAIN_CUTOFF = pd.Timestamp("2025-06-30")   # features use data up to here; target = spend after it
SCORE_CUTOFF = pd.Timestamp("2025-12-31")   # final predictions: features as of the latest data
HORIZON_DAYS = 180                          # LTV window we predict

def generate_data(n_customers=3000, seed=SEED):
    """Simulates customers who buy at random intervals and may churn after any order (BG/NBD-style)."""
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
    span = (end - start).days
    categories = ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports", "Books"]
    channels = ["Organic", "Paid Search", "Social", "Email", "Referral"]
    regions = ["North", "South", "East", "West"]
    customers, rows, oid = [], [], 0

    for cid in range(1, n_customers + 1):
        signup = int(rng.uniform(0, span * 0.95))
        lam = rng.gamma(1.3, 0.011)                 # purchase rate per day while active
        p_churn = rng.beta(1.5, 5)                  # chance of churning after each order
        channel = rng.choice(channels, p=[.30, .25, .20, .15, .10])
        region = rng.choice(regions)
        aov_mu = rng.normal(np.log(55), 0.5) + (0.15 if channel == "Referral" else 0)
        fav = rng.dirichlet(np.ones(len(categories)) * 0.6)
        customers.append((cid, start + pd.Timedelta(days=signup), channel, region))

        t = signup
        while t <= span:
            oid += 1
            value = rng.lognormal(aov_mu, 0.35)
            n_items = 1 + rng.poisson(1.2)
            shares = rng.dirichlet(np.ones(n_items))
            for s in shares:
                qty = int(rng.integers(1, 4))
                rows.append((oid, cid, start + pd.Timedelta(days=int(t)),
                             rng.choice(categories, p=fav), qty, max(0.5, round(value * s / qty, 2))))
            if rng.random() < p_churn:
                break
            t += max(1.0, rng.exponential(1 / lam))

    tx = pd.DataFrame(rows, columns=["order_id", "customer_id", "order_date", "category", "quantity", "unit_price"])
    cu = pd.DataFrame(customers, columns=["customer_id", "signup_date", "channel", "region"])
    return tx, cu

if (DATA_DIR / "transactions.csv").exists() and (DATA_DIR / "customers.csv").exists():
    transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["order_date"])
    customers = pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["signup_date"])
    print("Loaded existing data from ./data")
else:
    transactions, customers = generate_data()
    transactions.to_csv(DATA_DIR / "transactions.csv", index=False)
    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    print("Generated synthetic data and saved to ./data")

print(f"transactions: {transactions.shape}  |  customers: {customers.shape}")
display(transactions.head())

n_raw = len(transactions)
tx = transactions.drop_duplicates().dropna(subset=["customer_id", "order_date", "quantity", "unit_price"])
tx = tx[(tx["quantity"] > 0) & (tx["unit_price"] > 0)].copy()     # drop returns / bad rows
tx["revenue"] = tx["quantity"] * tx["unit_price"]

# merge transactions with the customer table on customer_id
tx = tx.merge(customers, on="customer_id", how="left", validate="many_to_one")

print(f"rows kept: {len(tx):,} of {n_raw:,}")
print("missing customer attributes:", int(tx["channel"].isna().sum()))
print(f"date range: {tx['order_date'].min().date()} -> {tx['order_date'].max().date()}")
print(f"customers with purchases: {tx['customer_id'].nunique():,}  |  orders: {tx['order_id'].nunique():,}  |  revenue: {tx['revenue'].sum():,.0f}")

# EDA plots
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
monthly = tx.set_index("order_date")["revenue"].resample("MS").sum()
ax[0].plot(monthly.index, monthly.values, marker="o", color="#2a6fdb")
ax[0].set_title("Monthly revenue"); ax[0].set_ylabel("Revenue")
order_vals = tx.groupby("order_id")["revenue"].sum()
sns.histplot(order_vals.clip(upper=order_vals.quantile(.99)), bins=40, ax=ax[1], color="#2a6fdb")
ax[1].set_title("Order value distribution (clipped at 99th pct)"); ax[1].set_xlabel("Order value")
plt.tight_layout(); plt.savefig(VIZ_DIR / "01_eda_revenue_and_orders.png", dpi=150); plt.show()

def build_features(tx, cutoff):
    """RFM + behavioural features using only transactions on/before `cutoff`."""
    cutoff = pd.Timestamp(cutoff)
    t = tx[tx["order_date"] <= cutoff]

    orders = (t.groupby(["customer_id", "order_id"])
                .agg(order_date=("order_date", "max"), order_value=("revenue", "sum"), items=("quantity", "sum"))
                .reset_index())
    g = orders.groupby("customer_id")
    f = pd.DataFrame({
        "frequency":     g["order_id"].nunique(),                               # F
        "monetary":      g["order_value"].sum(),                                # total spend
        "aov":           g["order_value"].mean(),                               # avg order value
        "aov_std":       g["order_value"].std().fillna(0),
        "max_order":     g["order_value"].max(),
        "avg_items":     g["items"].mean(),
        "first_order":   g["order_date"].min(),
        "last_order":    g["order_date"].max(),
    })
    f["recency_days"] = (cutoff - f["last_order"]).dt.days                      # R
    f["tenure_days"]  = (cutoff - f["first_order"]).dt.days
    f["avg_gap_days"] = np.where(f["frequency"] > 1,
                                 (f["last_order"] - f["first_order"]).dt.days / (f["frequency"] - 1).clip(lower=1),
                                 f["tenure_days"])
    f["recency_ratio"] = f["recency_days"] / (f["avg_gap_days"] + 1)            # overdue vs own rhythm
    f["orders_per_month"] = f["frequency"] / (f["tenure_days"] / 30 + 1)

    for d in (30, 90):
        recent = orders[orders["order_date"] > cutoff - pd.Timedelta(days=d)].groupby("customer_id")
        f[f"orders_last_{d}d"] = recent["order_id"].nunique()
        f[f"spend_last_{d}d"] = recent["order_value"].sum()
    f[[c for c in f.columns if "last_" in c and c.endswith("d")]] = f[[c for c in f.columns if "last_" in c and c.endswith("d")]].fillna(0)

    f["n_categories"] = t.groupby("customer_id")["category"].nunique()

    attrs = customers.set_index("customer_id")[["channel", "region"]]
    f = f.join(attrs).drop(columns=["first_order", "last_order"])
    f = pd.get_dummies(f, columns=["channel", "region"], drop_first=False, dtype=int)
    return f

def build_target(tx, cutoff, horizon):
    """Actual spend in the `horizon` days after cutoff."""
    cutoff = pd.Timestamp(cutoff)
    fut = tx[(tx["order_date"] > cutoff) & (tx["order_date"] <= cutoff + pd.Timedelta(days=horizon))]
    return fut.groupby("customer_id")["revenue"].sum().rename("future_ltv")

X_all = build_features(tx, TRAIN_CUTOFF)
y_all = build_target(tx, TRAIN_CUTOFF, HORIZON_DAYS).reindex(X_all.index).fillna(0.0)

print(f"Training table: {X_all.shape[0]:,} customers x {X_all.shape[1]} features")
print(f"Customers who bought in the next {HORIZON_DAYS} days: {(y_all > 0).mean():.1%}")
display(X_all.describe().T[["mean", "std", "min", "max"]].round(2).head(14))

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(X_all["recency_days"], bins=40, ax=ax[0], color="#2a6fdb"); ax[0].set_title("Recency (days)")
sns.histplot(X_all["frequency"].clip(upper=25), bins=25, ax=ax[1], color="#2a6fdb"); ax[1].set_title("Frequency (orders, clipped at 25)")
sns.histplot(X_all["aov"].clip(upper=X_all["aov"].quantile(.99)), bins=40, ax=ax[2], color="#2a6fdb"); ax[2].set_title("Average order value")
plt.tight_layout(); plt.savefig(VIZ_DIR / "02_rfm_distributions.png", dpi=150); plt.show()

corr = X_all[["frequency", "recency_days", "aov", "monetary", "tenure_days", "spend_last_90d"]].assign(future_ltv=y_all).corr()
plt.figure(figsize=(7, 5.5)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Correlation: RFM features vs future LTV"); plt.tight_layout()
plt.savefig(VIZ_DIR / "03_correlation_heatmap.png", dpi=150); plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED)

if HAS_XGB:
    boosted_name = "XGBoost"
    boosted = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8,
                           colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                           objective="reg:squarederror", random_state=SEED, n_jobs=-1)
else:
    boosted_name = "GradientBoosting (XGBoost fallback)"
    boosted = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_depth=4,
                                            min_samples_leaf=30, l2_regularization=2.0, random_state=SEED)

models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.6,
                                           random_state=SEED, n_jobs=-1),
    boosted_name: boosted,
}

def evaluate(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred)}

results, preds = {}, {}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for name, m in models.items():
    cv_rmse = -cross_val_score(m, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error").mean()
    m.fit(X_train, y_train)
    preds[name] = np.clip(m.predict(X_test), 0, None)
    results[name] = {**evaluate(y_test, preds[name]), "CV_RMSE(train)": cv_rmse}

# business heuristic: historical run-rate
runrate = (X_test["monetary"] / (X_test["tenure_days"] + 30) * HORIZON_DAYS)
preds["Run-rate heuristic"] = runrate.values
results["Run-rate heuristic"] = {**evaluate(y_test, runrate.values), "CV_RMSE(train)": np.nan}

metrics = pd.DataFrame(results).T.sort_values("RMSE").round(3)
display(metrics)

model_names = [n for n in models if n != "Mean baseline"]
best_name = metrics.loc[model_names, "RMSE"].idxmin()
best_model = models[best_name]
print(f"Best model on hold-out RMSE: {best_name}")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))

# model comparison
m_plot = metrics[["MAE", "RMSE"]].reset_index().melt(id_vars="index", var_name="metric", value_name="value")
sns.barplot(data=m_plot, x="index", y="value", hue="metric", ax=ax[0], palette=["#2a6fdb", "#f28e2b"])
ax[0].set_title("Model comparison (lower is better)"); ax[0].set_xlabel(""); ax[0].tick_params(axis="x", rotation=25)

# actual vs predicted
lim = float(np.quantile(y_test, 0.99))
ax[1].scatter(y_test, preds[best_name], s=10, alpha=.4, color="#2a6fdb")
ax[1].plot([0, lim], [0, lim], "r--", lw=1); ax[1].set_xlim(0, lim); ax[1].set_ylim(0, lim)
ax[1].set_title(f"Actual vs predicted ({best_name})"); ax[1].set_xlabel("Actual 180-day spend"); ax[1].set_ylabel("Predicted")

# residuals
resid = y_test.values - preds[best_name]
sns.histplot(np.clip(resid, -lim, lim), bins=50, ax=ax[2], color="#2a6fdb")
ax[2].axvline(0, color="r", ls="--"); ax[2].set_title("Residuals (actual - predicted)")
plt.tight_layout(); plt.savefig(VIZ_DIR / "04_model_validation.png", dpi=150); plt.show()

# how well does the model RANK customers? (what matters for targeting)
dec = pd.DataFrame({"actual": y_test.values, "pred": preds[best_name]})
dec["decile"] = pd.qcut(dec["pred"].rank(method="first"), 10, labels=range(1, 11))
lift = dec.groupby("decile")["actual"].mean()
plt.figure(figsize=(8, 4.5)); sns.barplot(x=lift.index, y=lift.values, color="#2a6fdb")
plt.title("Average ACTUAL spend by predicted-LTV decile (10 = highest)"); plt.xlabel("Predicted LTV decile"); plt.ylabel("Avg actual spend")
plt.tight_layout(); plt.savefig(VIZ_DIR / "05_decile_lift.png", dpi=150); plt.show()
top10_share = dec.nlargest(int(len(dec) * .1), "pred")["actual"].sum() / dec["actual"].sum()
print(f"Top 10% of customers by predicted LTV capture {top10_share:.1%} of actual future revenue")

# Feature importance (permutation on the hold-out set: works for any model)
pi = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=SEED,
                            scoring="neg_root_mean_squared_error", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False).head(12)
plt.figure(figsize=(8, 5)); sns.barplot(x=imp.values, y=imp.index, color="#2a6fdb")
plt.title(f"Top features driving predicted LTV ({best_name})"); plt.xlabel("Increase in RMSE when feature is shuffled")
plt.tight_layout(); plt.savefig(VIZ_DIR / "06_feature_importance.png", dpi=150); plt.show()
display(imp.round(3).to_frame("importance"))

final_model = clone(best_model).fit(X_all, y_all)

X_score = build_features(tx, SCORE_CUTOFF).reindex(columns=X_all.columns, fill_value=0)   # same columns/order as training
pred_ltv = np.clip(final_model.predict(X_score), 0, None)

out = pd.DataFrame({
    "customer_id": X_score.index,
    "frequency": X_score["frequency"].values,
    "recency_days": X_score["recency_days"].values,
    "aov": X_score["aov"].round(2).values,
    "historical_revenue": X_score["monetary"].round(2).values,
    "predicted_ltv_180d": pred_ltv.round(2),
})
out["total_projected_value"] = (out["historical_revenue"] + out["predicted_ltv_180d"]).round(2)

# ---- Segmentation on predicted LTV ----
out["ltv_percentile"] = out["predicted_ltv_180d"].rank(method="first", pct=True)
out["ltv_segment"] = pd.cut(out["ltv_percentile"], [0, .40, .70, .90, 1.0],
                            labels=["Bronze", "Silver", "Gold", "Platinum"], include_lowest=True)
actions = {
    "Platinum": "VIP program, early access, dedicated offers; protect and retain",
    "Gold":     "Cross-sell/upsell bundles, loyalty points; grow toward Platinum",
    "Silver":   "Targeted promotions and email nurturing; encourage repeat purchase",
    "Bronze":   "Low-cost automated campaigns, win-back only if margin allows",
}
out["recommended_action"] = out["ltv_segment"].map(actions)
out = out.sort_values("predicted_ltv_180d", ascending=False).reset_index(drop=True)

out.drop(columns="ltv_percentile").to_csv(OUT_DIR / "ltv_predictions.csv", index=False)
print(f"Saved {len(out):,} predictions -> outputs/ltv_predictions.csv")
display(out.head(10))

seg = (out.groupby("ltv_segment", observed=True)
          .agg(customers=("customer_id", "count"),
               avg_predicted_ltv=("predicted_ltv_180d", "mean"),
               total_predicted_ltv=("predicted_ltv_180d", "sum"),
               avg_frequency=("frequency", "mean"),
               avg_recency_days=("recency_days", "mean"),
               avg_aov=("aov", "mean"))
          .reindex(["Platinum", "Gold", "Silver", "Bronze"]))
seg["share_of_predicted_revenue"] = seg["total_predicted_ltv"] / seg["total_predicted_ltv"].sum()
display(seg.round(2))

palette = {"Platinum": "#6c4fd6", "Gold": "#f2b01e", "Silver": "#9aa5b1", "Bronze": "#b87333"}
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
sns.histplot(out["predicted_ltv_180d"].clip(upper=out["predicted_ltv_180d"].quantile(.99)), bins=50, ax=ax[0], color="#2a6fdb")
ax[0].set_title("Predicted 180-day LTV distribution")
ax[1].bar(seg.index, seg["customers"], color=[palette[s] for s in seg.index]); ax[1].set_title("Customers per segment")
ax[2].bar(seg.index, seg["share_of_predicted_revenue"] * 100, color=[palette[s] for s in seg.index])
ax[2].set_title("Share of predicted revenue (%)")
plt.tight_layout(); plt.savefig(VIZ_DIR / "07_ltv_segments.png", dpi=150); plt.show()

plt.figure(figsize=(8, 5.5))
samp = out.sample(min(1500, len(out)), random_state=SEED)
sns.scatterplot(data=samp, x="recency_days", y="frequency", hue="ltv_segment", size="predicted_ltv_180d",
                palette=palette, alpha=.6, sizes=(10, 150), hue_order=["Platinum", "Gold", "Silver", "Bronze"])
plt.ylim(0, samp["frequency"].quantile(.99)); plt.title("Segments in Recency vs Frequency space")
plt.tight_layout(); plt.savefig(VIZ_DIR / "08_segments_rfm_scatter.png", dpi=150); plt.show()

joblib.dump({"model": final_model, "model_name": best_name, "features": list(X_all.columns),
             "train_cutoff": str(TRAIN_CUTOFF.date()), "horizon_days": HORIZON_DAYS},
            MODEL_DIR / "ltv_model.joblib")

metrics.to_csv(OUT_DIR / "model_metrics.csv")
with open(OUT_DIR / "metrics.json", "w") as fh:
    json.dump({"best_model": best_name,
               "holdout": metrics.loc[best_name, ["MAE", "RMSE", "R2"]].to_dict(),
               "top10pct_revenue_capture": float(top10_share)}, fh, indent=2)

with pd.ExcelWriter(OUT_DIR / "ltv_summary.xlsx") as xw:          # Excel deliverable for marketing
    seg.round(2).to_excel(xw, sheet_name="Segment_Summary")
    metrics.to_excel(xw, sheet_name="Model_Metrics")
    out.drop(columns="ltv_percentile").to_excel(xw, sheet_name="Predictions", index=False)

print("Saved: models/ltv_model.joblib, outputs/ltv_predictions.csv, outputs/ltv_summary.xlsx, outputs/metrics.json")