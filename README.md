
# Fraud Clustering – Robust Scaling + PCA (whiten) + K‑Means

This repository/notebook demonstrates an unsupervised **clustering** workflow on the anonymized
Kaggle dataset (**~20k rows, 111 numeric features + optional label**) using an assignment‑ready
pipeline:

**Clean → Winsorize (1%) → Robust scale → PCA (95% variance, `whiten=True`) → K‑Means → Evaluate → Visualize (2‑D/3‑D/“4‑D”) → Talking points**

Dataset: https://www.kaggle.com/datasets/volodymyrgavrysh/fraud-detection-bank-dataset-20k-records-binary

---

## What’s included

- **Fraud_KMeans_Clustering_Scaled.ipynb** – main notebook with the updated scaling pipeline
  (winsorize + `RobustScaler` + optional `PowerTransformer` + `PCA(whiten=True)` + K‑Means).
- Auto reporting of:
  - Best **k** (Elbow + **Silhouette** + **Calinski–Harabasz** + **Davies–Bouldin**)
  - **Convergence** (`n_iter_`) – “when the algorithm stops”
  - **Separation ratio** (mean inter‑centroid distance / mean intra‑cluster distance)
  - Cluster **sizes** and **signed top features** at centroids
  - **2‑D / 3‑D / “4‑D”** visualizations from the same PCA used for clustering
- (Optional) **IsolationForest** pre‑filter to remove extreme outliers, then re‑run clustering

> “4‑D” visualization = PC1–PC2 on axes, **PC3 as point size**, **PC4 as transparency (alpha)**.

---

## Quick start

1) **Download** the Kaggle CSV and place it next to the notebook as `fraud_detection_bank_dataset.csv`  
2) **Install** requirements (Python 3.9+ recommended):  
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3) **Launch** Jupyter and run the notebook top‑to‑bottom:  
   ```bash
   jupyter notebook
   ```

If you prefer VS Code, just open the folder and run the notebook with the Python extension.

---

## Methodology (high level)

1. **Clean**
   - Drop duplicates; impute missing values (median)
   - Detect label column if present (optional; clustering does not use it)

2. **Tame outliers**
   - **Winsorize 1% tails per feature** to reduce the impact of extreme points that can form a fake “tiny far‑away cluster”

3. **Scale**
   - **`RobustScaler`** (median/IQR) instead of z‑score to resist remaining outliers
   - *(Optional)* **`PowerTransformer` (Yeo‑Johnson)** if heavy skew persists

4. **Dimensionality reduction**
   - **`PCA(n_components=0.95, whiten=True)`** → keep as many PCs as needed for 95% variance, decorrelate and set unit variance
   - Reuse this same PCA for **all visualizations** so plots match the clustering space

5. **K‑Means & model selection**
   - Evaluate **k = 2…12** with Elbow, Silhouette (higher better), **CH** (higher better) and **DB** (lower better)
   - Pick the smallest **k** near the elbow that also has strong Silhouette, better CH, lower DB, **balanced clusters** (avoid <5% micro‑clusters), and stable inertia across seeds

6. **Interpretation & reporting**
   - **Convergence**: report `n_iter_` and any `max_iter` hits
   - **Separation ratio**: inter‑centroid / intra‑cluster
   - **Profiles**: for each cluster, list **signed** top directions (e.g., `col_42 (+1.20σ)`, `col_7 (−0.85σ)`)
   - **2‑D/3‑D/“4‑D”**: describe what becomes clearer as more PCs are considered

---

## Exactly what to say (talking points template)

- **Best k**: Selected **k = {best_k}** (silhouette = {S:.3f}; CH = {CH:.1f}; DB = {DB:.3f}); no clusters <5%.  
- **Convergence**: K‑Means stopped after **{n_iter}** iterations (tol = 1e‑4, max_iter = 300).  
- **Geometry**: Separation ratio **{sep_ratio:.2f}** (higher indicates better separation).  
- **Profiles**: Cluster 0 top = {top0}, Cluster 1 top = {top1}, … (signed standardized directions).  
- **Views**: 2‑D shows __; 3‑D reveals __; “4‑D” highlights __ via size/alpha.

*(Replace braces with your run’s numbers from the notebook.)*

---

## 50‑word summaries (paste into your report)

**Applicability (50 words)**  
K‑Means clustering groups anonymized transactions by behavioral similarity, enabling pattern discovery without field semantics. Using standardized, PCA‑reduced features, clusters expose compact structures and hidden axes that inform threshold setting, rule proposals, feature engineering, analyst triage, and ongoing monitoring for evolving fraud‑risk behaviors within the term project’s end‑to‑end data pipeline.

**Results (50 words)**  
K‑Means selected k based on silhouette and elbow. The algorithm converged within the iteration limit. Clusters exhibit measurable separation; signed centroid loadings identify key directions. Two‑dimensional views partially overlap, while three‑dimensional and four‑dimensional encodings reveal additional structure that informs actionable grouping decisions, model features, and analyst review priorities.

---

## Reproducibility

- Fixed `random_state = 42` for deterministic reruns; stability can be shown by comparing inertia across several seeds (0, 1, 2, 42, 1337).  
- Use the same PCA for clustering and plotting to avoid mismatched views.  
- Record package versions with:  
  ```python
  import sklearn, numpy, pandas, matplotlib
  print(sklearn.__version__, numpy.__version__, pandas.__version__, matplotlib.__version__)
  ```

---

## Optional extensions

- **IsolationForest** to remove the most extreme outliers before scaling/PCA  
- **DBSCAN** to discover dense sub‑clouds if k‑means keeps isolating a tiny outlier cluster  
- Add a small command‑line batch scorer that loads the fitted pipeline and assigns cluster labels to a CSV
