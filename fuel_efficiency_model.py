"""
fuel_efficiency_model.py

Train and evaluate a Linear Regression model to predict car fuel efficiency (MPG)
using the Auto MPG dataset.

Usage:
    python fuel_efficiency_model.py --train
    python fuel_efficiency_model.py --plot
    python fuel_efficiency_model.py --save-model model.pkl
"""

import argparse
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Paths
DATA_PATH = Path(__file__).resolve().parent / "auto-mpg.csv"
IMAGE_DIR = Path(__file__).resolve().parent / "images"
IMAGE_DIR.mkdir(exist_ok=True)

# Columns
TARGET = "mpg"
NUMERIC_FEATURES = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model year",
]
CATEGORICAL_FEATURES = ["origin"]  # treat origin as categorical for One‑Hot Encoding


def load_data() -> pd.DataFrame:
    """Load and clean the Auto MPG dataset."""
    df = pd.read_csv(DATA_PATH)

    # Clean the horsepower column (replace '?' with NaN and convert to numeric)
    df["horsepower"].replace("?", np.nan, inplace=True)
    df["horsepower"] = pd.to_numeric(df["horsepower"])

    # Drop rows with any missing values
    df.dropna(inplace=True)

    return df


def build_pipeline() -> Pipeline:
    """Build a preprocessing + Linear Regression pipeline."""
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])
    return model


def train(args: argparse.Namespace) -> None:
    """Train the model, evaluate it, and save artifacts/plots."""
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {rmse:.3f} MPG")

    # Save model if requested
    if args.save_model:
        joblib.dump(model, args.save_model)
        print(f"Model saved to {args.save_model}")

    # ----- Visualizations -----
    # 1. Actual vs Predicted
    plt.figure()
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual MPG")
    plt.ylabel("Predicted MPG")
    plt.title("Actual vs Predicted MPG")
    plt.savefig(IMAGE_DIR / "actual_vs_predicted.png", bbox_inches="tight")
    plt.close()

    # 2. Residuals distribution
    residuals = y_test - y_pred
    plt.figure()
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residual (MPG)")
    plt.title("Distribution of Residuals")
    plt.savefig(IMAGE_DIR / "residuals_distribution.png", bbox_inches="tight")
    plt.close()

    print("Training complete. Plots saved to the images/ directory.")


def plot_corr() -> None:
    """Generate and save a correlation heatmap for exploratory analysis."""
    df = load_data()
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix (Numeric Features vs MPG)")
    plt.savefig(IMAGE_DIR / "correlation_matrix.png", bbox_inches="tight")
    plt.close()

    print("Correlation heatmap saved to images/correlation_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Fuel Efficiency Prediction")
    parser.add_argument(
        "--train", action="store_true", help="Train the Linear Regression model"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate correlation plot only"
    )
    parser.add_argument(
        "--save-model", type=str, default=None, help="Path to save trained model (joblib)"
    )
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.plot:
        plot_corr()
    else:
        parser.print_help()
