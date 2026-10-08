# ML Unsupervised Learning

A collection of practical **Unsupervised Machine Learning** projects implemented in Python using Scikit-learn.

This repository demonstrates three fundamental unsupervised learning techniques:

- **K-Means Clustering**
- **Dimensionality Reduction using PCA**
- **Anomaly Detection using Isolation Forest**

---

## 📁 Project Structure

```text
ML_Unsupervised_Learning/
│
├── README.md
│
├── Anomaly Detection/
│   └── isolation_forest_anomaly_detection.py
│
├── Dimensionality Reduction/
│   └── pca_dimensionality_reduction.py
│
└── KMeans_Clustering/
    └── kmeans_clustering.py
```

---

## 🔹 1. K-Means Clustering

### `KMeans_Clustering/kmeans_clustering.py`

This project demonstrates **K-Means Clustering** using a customer segmentation example.

The algorithm groups customers based on:

- Annual Income Score
- Spending Score

There is no target variable `y` because K-Means is an **unsupervised learning algorithm**.

### Workflow

```text
Generate Customer Data
        ↓
Feature Scaling
        ↓
Elbow Method
        ↓
Select Number of Clusters
        ↓
Train K-Means
        ↓
Assign Cluster Labels
        ↓
Visualize Customer Segments
```

### Main Concepts

- K-Means Clustering
- Customer Segmentation
- Feature Scaling
- Elbow Method
- Inertia
- Cluster Centers

---

## 🔹 2. Dimensionality Reduction

### `Dimensionality Reduction/pca_dimensionality_reduction.py`

This project demonstrates **Principal Component Analysis (PCA)** for dimensionality reduction.

PCA reduces the number of features while preserving as much of the data's variance as possible.

The project transforms multiple customer features into a smaller number of **Principal Components**.

### Workflow

```text
Generate Customer Data
        ↓
Feature Scaling
        ↓
Apply PCA
        ↓
Calculate Explained Variance
        ↓
Reduce Features
        ↓
Visualize Reduced Data
```

### Main Concepts

- Principal Component Analysis (PCA)
- Dimensionality Reduction
- Feature Scaling
- Principal Components
- Explained Variance

---

## 🔹 3. Anomaly Detection

### `Anomaly Detection/isolation_forest_anomaly_detection.py`

This project demonstrates **Anomaly Detection** using the **Isolation Forest** algorithm.

Isolation Forest identifies observations that are significantly different from the majority of the dataset.

For example, unusually high or low customer activity can be detected as anomalies.

### Workflow

```text
Generate Normal Data
        ↓
Add Abnormal Data
        ↓
Feature Scaling
        ↓
Train Isolation Forest
        ↓
Detect Anomalies
        ↓
Visualize Results
```

Isolation Forest produces:

```text
 1  → Normal
-1  → Anomaly
```

### Main Concepts

- Anomaly Detection
- Isolation Forest
- Outliers
- Contamination
- Unsupervised Learning

---

# 🧠 Unsupervised Learning

**Unsupervised Learning** is a type of machine learning where the model learns patterns and structures from data without a target variable.

This repository demonstrates three different applications:

| Technique | Algorithm | Purpose |
|---|---|---|
| Clustering | K-Means | Group similar observations |
| Dimensionality Reduction | PCA | Reduce the number of features |
| Anomaly Detection | Isolation Forest | Detect unusual observations |

### Overview

```text
                    Unsupervised Learning
                            │
           ┌────────────────┼────────────────┐
           │                │                │
      Clustering     Dimensionality     Anomaly Detection
                       Reduction
           │                │                │
        K-Means            PCA       Isolation Forest
           │                │                │
      Find Groups      Reduce Data    Find Outliers
```

---

# 🛠️ Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ML_Unsupervised_Learning.git
```

Navigate to the project directory:

```bash
cd ML_Unsupervised_Learning
```

Install the required dependencies:

```bash
pip install numpy matplotlib scikit-learn joblib
```

---

# ▶️ Running the Projects

## K-Means Clustering

```bash
cd "KMeans_Clustering"
python kmeans_clustering.py
```

## PCA

```bash
cd "../Dimensionality Reduction"
python pca_dimensionality_reduction.py
```

## Isolation Forest

```bash
cd "../Anomaly Detection"
python isolation_forest_anomaly_detection.py
```

---

# 🎯 Learning Objectives

Through these projects, I practiced:

- Understanding Unsupervised Machine Learning
- Generating synthetic datasets using NumPy
- Feature scaling using `StandardScaler`
- Customer segmentation using K-Means
- Selecting the number of clusters using the Elbow Method
- Dimensionality reduction using PCA
- Understanding explained variance
- Detecting anomalies using Isolation Forest
- Visualizing machine learning results using Matplotlib
- Saving trained models using Joblib

---

# 📌 Key Takeaway

Different unsupervised learning techniques solve different types of problems:

```text
Need to find groups?
        ↓
     K-Means


Need to reduce the number of features?
        ↓
       PCA


Need to detect unusual observations?
        ↓
 Isolation Forest
```

Together, these projects provide a practical introduction to several important techniques in **Unsupervised Machine Learning**.