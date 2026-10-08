"""
K-Means Clustering - Customer Segmentation

This is an unsupervised learning example. There is NO target variable y.
The algorithm groups customers based on annual income and spending score.

Workflow:
1. Generate synthetic customer data
2. Scale the features
3. Use the Elbow Method to inspect a suitable number of clusters
4. Train K-Means
5. Assign each customer to a cluster
6. Visualize the customer segments
7. Save the clustering model
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent


def create_customer_data() -> np.ndarray:
    """Create four customer groups with different behavior patterns."""
    rng = np.random.default_rng(RANDOM_STATE)

    group_1 = rng.multivariate_normal(
        mean=[25, 25],
        cov=[[20, 0], [0, 20]],
        size=100,
    )
    group_2 = rng.multivariate_normal(
        mean=[25, 75],
        cov=[[20, 0], [0, 20]],
        size=100,
    )
    group_3 = rng.multivariate_normal(
        mean=[75, 25],
        cov=[[20, 0], [0, 20]],
        size=100,
    )
    group_4 = rng.multivariate_normal(
        mean=[75, 75],
        cov=[[20, 0], [0, 20]],
        size=100,
    )

    data = np.vstack([group_1, group_2, group_3, group_4])
    return np.clip(data, 0, 100)


def main() -> None:
    # -------------------------------------------------
    # 1. Create customer data
    # -------------------------------------------------
    data = create_customer_data()

    # Feature 1: Annual Income Score (normalized illustrative value)
    # Feature 2: Spending Score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # -------------------------------------------------
    # 2. Elbow Method
    # -------------------------------------------------
    k_values = range(1, 9)
    inertia = []

    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(9, 5))
    plt.plot(list(k_values), inertia, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("K-Means - Elbow Method")
    plt.xticks(list(k_values))
    plt.tight_layout()
    plt.savefig(BASE_DIR / "elbow_method.png", dpi=150)
    plt.show()
    plt.close()

    # -------------------------------------------------
    # 3. Train K-Means with K=4
    # -------------------------------------------------
    n_clusters = 4
    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    labels = model.fit_predict(X_scaled)

    # -------------------------------------------------
    # 4. Print cluster information
    # -------------------------------------------------
    print("\n=== K-Means Clustering: Customer Segmentation ===")
    print(f"Number of customers: {len(data)}")
    print(f"Number of clusters: {n_clusters}")

    for cluster_id in range(n_clusters):
        cluster_data = data[labels == cluster_id]
        print(
            f"Cluster {cluster_id}: "
            f"customers={len(cluster_data)}, "
            f"avg_income={cluster_data[:, 0].mean():.2f}, "
            f"avg_spending={cluster_data[:, 1].mean():.2f}"
        )

    # -------------------------------------------------
    # 5. Visualize clusters
    # -------------------------------------------------
    plt.figure(figsize=(10, 7))
    plt.scatter(
        data[:, 0],
        data[:, 1],
        c=labels,
        s=50,
        alpha=0.75,
    )

    # Convert cluster centers back to original scale.
    centers = scaler.inverse_transform(model.cluster_centers_)
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=220,
        edgecolors="black",
        linewidths=1.5,
        label="Cluster centers",
    )

    plt.xlabel("Annual Income Score")
    plt.ylabel("Spending Score")
    plt.title("K-Means Customer Segmentation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "customer_segments.png", dpi=150)
    plt.show()
    plt.close()

    # -------------------------------------------------
    # 6. Save the model and scaler together
    # -------------------------------------------------
    joblib.dump(
        {
            "scaler": scaler,
            "model": model,
        },
        BASE_DIR / "kmeans_customer_segmentation.pkl",
    )

    print(f"\nModel saved to: {BASE_DIR / 'kmeans_customer_segmentation.pkl'}")


if __name__ == "__main__":
    main()
