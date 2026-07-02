# 🎯 Customer Segmentation Engine: Precision Marketing via K-Means Clustering

### Transforming raw transactional data into actionable, revenue-driving customer personas.

## Introduction 🌟

As a data scientist tackling digital marketing efficiency, I built an unsupervised machine learning pipeline to segment a customer base into **6 statistically distinct personas** using K-Means clustering. Rather than relying on generic demographics, this model uncovers behavioral and spending patterns — enabling marketing teams to shift from broad, wasteful campaigns to **hyper-targeted, ROI-driven strategies**.

The pipeline handles everything from raw data ingestion to a production-ready, serialized model, following a rigorous EDA → Optimization → Deployment workflow.

> **Business Impact**: Instead of a "one-size-fits-all" campaign, marketing teams can now allocate budget based on persona value — prioritizing high-LTV segments like *Elite Young Spenders* while designing win-back campaigns for *Lapsed Bargain Hunters*.

## Key Features & Pipeline 🛠️

**1. Data Preparation & Feature Engineering**
- Cleaned raw marketing data (missing value imputation, type correction, duplicate checks)
- Engineered derived features (e.g., total spending aggregation, age binning into generational cohorts)

**2. Exploratory Data Analysis (EDA)**
- Analyzed distribution and skew of core numerical features (Age, Income, Total Spend) via histograms and density plots
- Detected outliers in Income and Spending across categorical segments (Education, Marital Status) using boxplots
- Ran correlation analysis to evaluate multicollinearity and identify key linear relationships prior to modeling
- Conducted cohort and behavioral analysis (e.g., campaign conversion rates by demographic)

**3. Clustering Optimization**
- Applied `StandardScaler` to normalize features (mean=0, variance=1), preventing high-magnitude features (like Income) from dominating distance calculations
- Used the **Elbow Method** (WCSS across K=2 to 9) to mathematically justify the optimal cluster count
- Validated cluster separation visually using **PCA** to project scaled data into a 2D component space

**4. Persona Profiling**
- Aggregated feature means by cluster to translate raw centroids into human-readable customer personas
- Assigned strategic, marketing-ready labels based on spending behavior, channel preference, and demographic traits

## The 6 Customer Segments 📊

| Cluster ID | Persona Name | Core Traits |
|:---|:---|:---|
| **0** | 🛍️ Window Shoppers | High web visit frequency, but lowest overall spending |
| **1** | 🌐 Omnichannel Spenders | Older demographic; high engagement across multiple sales channels |
| **2** | 📉 Lapsed Bargain Hunters | Low spending combined with high account inactivity |
| **3** | 🏛️ Affluent Traditionalists | Wealthy, senior customers who shop strictly brick-and-mortar |
| **4** | ⭐ Active Core Customers | Recent purchasers with balanced, consistent spending habits |
| **5** | 💎 Elite Young Spenders | Youngest segment; highest income and highest total spend |

## Visualizations Utilized 📈

- **Distribution Plots**: Density and histogram analysis of Age, Income, and Spending to detect skew
- **Boxplots**: Income and Spending variance across Education and Marital Status
- **Correlation Heatmap**: Multicollinearity check prior to feature selection
- **Elbow Curve**: WCSS vs. K plot to justify cluster count selection
- **PCA Scatter Plot**: 2D visualization confirming clean separation between the 6 clusters

## Tech Stack 💻

| Category | Tools |
|:---|:---|
| **Language** | Python |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn (K-Means, PCA, StandardScaler) |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib / Pickle |


## Model Deployment / Persistence 🚀

The final trained **K-Means model** and fitted **StandardScaler** are serialized and saved to the `/models` directory, ensuring identical preprocessing is applied to any new customer data at inference time:

```python
import joblib

joblib.dump(kmeans_model, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
```

This setup allows new, unseen customers to be scored and assigned a persona in real time — making the model **deployment-ready** for integration into a CRM or marketing automation platform.

## Conclusion

This project demonstrates end-to-end proficiency in **unsupervised learning, feature engineering, and dimensionality reduction** — translating raw behavioral data into a segmentation framework that directly informs marketing spend and campaign strategy.

**Key Takeaways**:
- Data-driven cluster count selection (Elbow Method) over arbitrary guessing
- PCA-validated cluster separation confirms model reliability, not just statistical convenience
- Personas are directly actionable — each maps to a distinct marketing strategy

**Technologies**: Python • Scikit-Learn • K-Means • PCA • Pandas • Seaborn • Matplotlib

---
