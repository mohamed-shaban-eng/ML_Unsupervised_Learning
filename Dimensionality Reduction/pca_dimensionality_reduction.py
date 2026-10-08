"""
PCA - Dimensionality Reduction

This is an unsupervised learning example.
There is NO target variable y.

PCA reduces the number of features while preserving
as much information (variance) as possible.

Workflow:
1. Generate synthetic customer data with multiple features
2. Standardize the features
3. Apply PCA
4. Inspect explained variance
5. Reduce the data to 2 dimensions
6. Visualize the reduced data
7. Save the PCA model and scaler
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent


def create_customer_data() -> np.ndarray:
    """Create synthetic customer data with multiple features."""
    rng = np.random.default_rng(RANDOM_STATE)

    n_customers = 400

    age = rng.normal(35, 10, n_customers)
    income = rng.normal(50, 15, n_customers)
    spending = rng.normal(50, 20, n_customers)
    online_purchases = rng.normal(30, 10, n_customers)
    store_purchases = rng.normal(40, 12, n_customers)
    website_visits = rng.normal(50, 15, n_customers)

    data = np.column_stack(
        [
            age,
            income,
            spending,
            online_purchases,
            store_purchases,
            website_visits,
        ]
    )

    return np.clip(data, 0, None)


def main() -> None:
    # -------------------------------------------------
    # 1. Create customer data
    # -------------------------------------------------
    data = create_customer_data()

    print("\n=== PCA Dimensionality Reduction ===")
    print(f"Original shape: {data.shape}")

    # -------------------------------------------------
    # 2. Standardize the features
    # -------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # -------------------------------------------------
    # 3. Apply PCA
    # -------------------------------------------------
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    print(f"Reduced shape: {X_pca.shape}")

    # -------------------------------------------------
    # 4. Explained variance
    # -------------------------------------------------
    explained_variance = pca.explained_variance_ratio_

    print("\nExplained variance ratio:")
    for component, variance in enumerate(explained_variance, start=1):
        print(
            f"Principal Component {component}: "
            f"{variance:.2%}"
        )

    print(
        f"\nTotal variance preserved: "
        f"{explained_variance.sum():.2%}"
    )

    # -------------------------------------------------
    # 5. Visualize the reduced data
    # -------------------------------------------------
    plt.figure(figsize=(10, 7))

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        s=50,
        alpha=0.7,
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA - Customer Data in 2D")
    plt.tight_layout()

    plt.savefig(
        BASE_DIR / "pca_customer_data.png",
        dpi=150,
    )

    plt.show()
    plt.close()

    # -------------------------------------------------
    # 6. Save the PCA model and scaler
    # -------------------------------------------------
    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca,
        },
        BASE_DIR / "pca_customer_reduction.pkl",
    )

    print(
        f"\nModel saved to: "
        f"{BASE_DIR / 'pca_customer_reduction.pkl'}"
    )


if __name__ == "__main__":
    main()