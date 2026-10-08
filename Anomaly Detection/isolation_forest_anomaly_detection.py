"""
Isolation Forest - Anomaly Detection

This is an unsupervised learning example.
There is NO target variable y.

The algorithm detects unusual customers based on
their spending behavior and transaction frequency.

Workflow:
1. Generate synthetic customer data
2. Add some abnormal customers
3. Train Isolation Forest
4. Predict normal and anomalous points
5. Visualize the anomalies
6. Save the model
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent


def create_customer_data() -> np.ndarray:
    """Create normal customers and a few abnormal customers."""

    rng = np.random.default_rng(RANDOM_STATE)

    # Normal customers
    normal_customers = rng.multivariate_normal(
        mean=[50, 50],
        cov=[[100, 0], [0, 100]],
        size=450,
    )

    # Abnormal customers
    anomalies = np.array(
        [
            [10, 150],
            [150, 20],
            [160, 160],
            [5, 5],
            [140, 10],
            [10, 140],
            [170, 170],
            [180, 5],
            [5, 180],
            [160, 5],
        ]
    )

    data = np.vstack(
        [
            normal_customers,
            anomalies,
        ]
    )

    return np.clip(data, 0, 200)


def main() -> None:
    # -------------------------------------------------
    # 1. Create customer data
    # -------------------------------------------------
    data = create_customer_data()

    print("\n=== Isolation Forest: Anomaly Detection ===")
    print(f"Number of customers: {len(data)}")

    # Feature 1: Spending Score
    # Feature 2: Transaction Frequency

    # -------------------------------------------------
    # 2. Scale the features
    # -------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # -------------------------------------------------
    # 3. Train Isolation Forest
    # -------------------------------------------------
    model = IsolationForest(
        contamination=0.02,
        random_state=RANDOM_STATE,
        n_estimators=100,
    )

    predictions = model.fit_predict(X_scaled)

    # Isolation Forest returns:
    #  1  -> Normal
    # -1  -> Anomaly

    anomalies = predictions == -1
    normal = predictions == 1

    print(f"Normal customers: {normal.sum()}")
    print(f"Anomalous customers: {anomalies.sum()}")

    # -------------------------------------------------
    # 4. Visualize anomalies
    # -------------------------------------------------
    plt.figure(figsize=(10, 7))

    plt.scatter(
        data[normal, 0],
        data[normal, 1],
        s=50,
        alpha=0.7,
        label="Normal",
    )

    plt.scatter(
        data[anomalies, 0],
        data[anomalies, 1],
        s=120,
        marker="X",
        label="Anomaly",
    )

    plt.xlabel("Spending Score")
    plt.ylabel("Transaction Frequency")
    plt.title("Isolation Forest - Anomaly Detection")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        BASE_DIR / "isolation_forest_anomalies.png",
        dpi=150,
    )

    plt.show()
    plt.close()

    # -------------------------------------------------
    # 5. Print detected anomalies
    # -------------------------------------------------
    print("\nDetected anomalies:")

    for index in np.where(anomalies)[0]:
        print(
            f"Customer {index}: "
            f"spending={data[index, 0]:.2f}, "
            f"frequency={data[index, 1]:.2f}"
        )

    # -------------------------------------------------
    # 6. Save model and scaler
    # -------------------------------------------------
    joblib.dump(
        {
            "scaler": scaler,
            "model": model,
        },
        BASE_DIR / "isolation_forest_anomaly_detection.pkl",
    )

    print(
        f"\nModel saved to: "
        f"{BASE_DIR / 'isolation_forest_anomaly_detection.pkl'}"
    )


if __name__ == "__main__":
    main()