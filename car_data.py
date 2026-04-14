
"""
============================================================
  Car Price Prediction - Used Cars

  File    : car data.csv
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(filepath="car data.csv"):
    df = pd.read_csv(filepath)

    # Strip whitespace from column names and string values
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    print("=" * 55)
    print("DATASET OVERVIEW")
    print("=" * 55)
    print(f"Shape         : {df.shape}")
    print(f"Columns       : {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nStatistics:\n{df.describe()}")
    return df


# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
def eda(df):
    print("\n" + "=" * 55)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Car Price - Exploratory Data Analysis", fontsize=16, fontweight="bold")

    # 1. Distribution of Selling Price
    axes[0, 0].hist(df["Selling_Price"], bins=30, color="#4C72B0", edgecolor="white")
    axes[0, 0].set_title("Distribution of Selling Price")
    axes[0, 0].set_xlabel("Selling Price (Lakhs)")
    axes[0, 0].set_ylabel("Count")

    # 2. Selling Price vs Present Price
    axes[0, 1].scatter(df["Present_Price"], df["Selling_Price"],
                       alpha=0.6, color="#DD8452", edgecolors="white", linewidths=0.4)
    axes[0, 1].set_title("Selling Price vs Present Price")
    axes[0, 1].set_xlabel("Present Price (Lakhs)")
    axes[0, 1].set_ylabel("Selling Price (Lakhs)")

    # 3. Fuel Type count
    fuel_counts = df["Fuel_Type"].value_counts()
    axes[0, 2].bar(fuel_counts.index, fuel_counts.values,
                   color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0, 2].set_title("Cars by Fuel Type")
    axes[0, 2].set_xlabel("Fuel Type")
    axes[0, 2].set_ylabel("Count")

    # 4. Selling Price by Fuel Type
    fuel_groups = [grp["Selling_Price"].values for _, grp in df.groupby("Fuel_Type")]
    fuel_labels  = df["Fuel_Type"].unique()
    axes[1, 0].boxplot(fuel_groups, labels=fuel_labels)
    axes[1, 0].set_title("Selling Price by Fuel Type")
    axes[1, 0].set_xlabel("Fuel Type")
    axes[1, 0].set_ylabel("Selling Price")

    # 5. Selling Price by Transmission
    trans_groups = [grp["Selling_Price"].values for _, grp in df.groupby("Transmission")]
    trans_labels  = df["Transmission"].unique()
    axes[1, 1].boxplot(trans_groups, labels=trans_labels)
    axes[1, 1].set_title("Selling Price by Transmission")
    axes[1, 1].set_xlabel("Transmission")
    axes[1, 1].set_ylabel("Selling Price")

    # 6. Kms Driven vs Selling Price
    axes[1, 2].scatter(df["Driven_kms"], df["Selling_Price"],
                       alpha=0.5, color="#55A868", edgecolors="white", linewidths=0.4)
    axes[1, 2].set_title("Kms Driven vs Selling Price")
    axes[1, 2].set_xlabel("Driven_kms")
    axes[1, 2].set_ylabel("Selling Price")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150)
    plt.show()
    print("EDA plot saved -> eda_plots.png")


# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    df = df.copy()

    # Derive Car_Age from Year
    df["Car_Age"] = 2024 - df["Year"]
    df.drop(columns=["Car_Name", "Year"], inplace=True)

    # Label encode categorical columns
    le = LabelEncoder()
    for col in ["Fuel_Type", "Selling_type", "Transmission"]:
        df[col] = le.fit_transform(df[col].astype(str))

    print("\nEncoding Reference:")
    print("  Fuel_Type    -> Diesel=0, CNG=1, Petrol=2")
    print("  Selling_type  -> Dealer=0, Individual=1")
    print("  Transmission -> Automatic=0, Manual=1")
    print(f"\nProcessed shape: {df.shape}")
    print(df.head())
    return df


# ─────────────────────────────────────────────
# 4. CORRELATION HEATMAP
# ─────────────────────────────────────────────
def plot_correlation(df):
    plt.figure(figsize=(10, 7))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                mask=mask, linewidths=0.5, square=True)
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
    print("Correlation heatmap saved -> correlation_heatmap.png")


# ─────────────────────────────────────────────
# 5. SPLIT & SCALE
# ─────────────────────────────────────────────
def split_and_scale(df, target="Selling_Price"):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"\nTrain samples : {X_train_sc.shape[0]}")
    print(f"Test  samples : {X_test_sc.shape[0]}")
    return X_train_sc, X_test_sc, y_train, y_test, list(X.columns), scaler


# ─────────────────────────────────────────────
# 6. TRAIN & EVALUATE
# ─────────────────────────────────────────────
def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

    print(f"\n{'─' * 45}")
    print(f"  Model  : {name}")
    print(f"{'─' * 45}")
    print(f"  MAE    : {mae:.4f}")
    print(f"  RMSE   : {rmse:.4f}")
    print(f"  R²     : {r2:.4f}")
    print(f"  CV R²  : {cv:.4f}  (5-fold)")

    return {"Model": name, "MAE": mae, "RMSE": rmse,
            "R2": r2, "CV_R2": cv, "y_pred": y_pred}


# ─────────────────────────────────────────────
# 7. PLOT COMPARISON
# ─────────────────────────────────────────────
def plot_results(results_list, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Comparison", fontsize=15, fontweight="bold")

    models  = [r["Model"] for r in results_list]
    r2_vals = [r["R2"]    for r in results_list]
    colors  = ["#4C72B0", "#DD8452", "#55A868"]

    bars = axes[0].bar(models, r2_vals, color=colors, edgecolor="white", width=0.5)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("R² Score")
    axes[0].set_title("R² Score Comparison")
    for bar, val in zip(bars, r2_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01, f"{val:.3f}",
                     ha="center", va="bottom", fontweight="bold")

    best = max(results_list, key=lambda r: r["R2"])
    axes[1].scatter(y_test, best["y_pred"], alpha=0.6,
                    color="#4C72B0", edgecolors="white", linewidths=0.4)
    lo = min(float(y_test.min()), float(best["y_pred"].min())) - 1
    hi = max(float(y_test.max()), float(best["y_pred"].max())) + 1
    axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect Prediction")
    axes[1].set_xlabel("Actual Selling Price")
    axes[1].set_ylabel("Predicted Selling Price")
    axes[1].set_title(f"Actual vs Predicted\n({best['Model']})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()
    print("Model comparison plot saved -> model_comparison.png")


# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def plot_feature_importance(rf_model, feature_names):
    importances  = rf_model.feature_importances_
    indices      = np.argsort(importances)[::-1]
    sorted_feats = [feature_names[i] for i in indices]
    sorted_imp   = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_feats, sorted_imp, color="#4C72B0", edgecolor="white")
    plt.title("Feature Importance - Random Forest", fontsize=14, fontweight="bold")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("Feature importance plot saved -> feature_importance.png")


# ─────────────────────────────────────────────
# 9. PREDICT NEW CAR
# ─────────────────────────────────────────────
def predict_new(model, scaler, feature_names):
    """
    Encoding guide:
        Fuel_Type    : Diesel=0, CNG=1, Petrol=2
        Seller_Type  : Dealer=0, Individual=1
        Transmission : Automatic=0, Manual=1
    Edit values below for your car.
    """
    new_car = pd.DataFrame([{
        "Present_Price": 5.59,
        "Kms_Driven"   : 27000,
        "Fuel_Type"    : 2,       # Petrol
        "Seller_Type"  : 0,       # Dealer
        "Transmission" : 1,       # Manual
        "Owner"        : 0,
        "Car_Age"      : 6        # 2024 - 2018
    }])
    new_car=new_car.rename(columns={
        "Kms_Driven":"Driven_kms",
        "Seller_Type":"Selling_type"
    })

    new_car    = new_car[feature_names]
    new_car_sc = scaler.transform(new_car)
    price      = model.predict(new_car_sc)[0]

    print("\n" + "=" * 55)
    print("  PREDICTION ON NEW CAR")
    print("=" * 55)
    for k, v in new_car.to_dict(orient="records")[0].items():
        print(f"  {k:<18}: {v}")
    print(f"\n  >>> Predicted Selling Price : Rs. {price:.2f} Lakhs")
    print("=" * 55)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # Step 1 – Load (columns auto-stripped here)
    df = load_data("car data.csv")

    # Step 2 – EDA
    eda(df)

    # Step 3 – Preprocess
    df_proc = preprocess(df)

    # Step 4 – Correlation heatmap
    plot_correlation(df_proc)

    # Step 5 – Split & scale
    X_train, X_test, y_train, y_test, feature_names, scaler = split_and_scale(df_proc)

    # Step 6 – Train models
    print("\n" + "=" * 55)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 55)

    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                   max_depth=4, random_state=42)

    results = []
    results.append(train_and_evaluate("Linear Regression", lr, X_train, X_test, y_train, y_test))
    results.append(train_and_evaluate("Random Forest",     rf, X_train, X_test, y_train, y_test))
    results.append(train_and_evaluate("Gradient Boosting", gb, X_train, X_test, y_train, y_test))

    # Step 7 – Compare models visually
    plot_results(results, y_test)

    # Step 8 – Feature importance
    plot_feature_importance(rf, feature_names)

    # Step 9 – Summary table
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "y_pred"} for r in results])
    print("\n" + "=" * 55)
    print("FINAL SUMMARY TABLE")
    print("=" * 55)
    print(summary.to_string(index=False))

    best_name = summary.loc[summary["R2"].idxmax(), "Model"]
    print(f"\nBest Model: {best_name}")

    # Step 10 – Predict on a new sample
    model_map = {"Linear Regression": lr, "Random Forest": rf, "Gradient Boosting": gb}
    predict_new(model_map[best_name], scaler, feature_names)
