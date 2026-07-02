# 🎯 Customer Segmentation Engine: Precision Marketing via K-Means Clustering

### Transforming raw transactional data into actionable, revenue-driving customer personas.

## Introduction 🌟

As a data scientist tackling digital marketing efficiency, I built an unsupervised machine learning pipeline to segment a customer base into **6 statistically distinct personas** using K-Means clustering. Rather than relying on generic demographics, this model uncovers behavioral and spending patterns — enabling marketing teams to shift from broad, wasteful campaigns to **hyper-targeted, ROI-driven strategies**.

<img width="527" height="407" alt="5" src="https://github.com/user-attachments/assets/9060f856-e30b-48f1-9ea2-fc7e7b500d64" />

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
<img width="511" height="407" alt="Age Distribution" src="https://github.com/user-attachments/assets/3f035f78-4df5-4411-86f8-dc827839e664" />

- **Boxplots**: Income and Spending variance across Education and Marital Status
<img width="523" height="407" alt="2" src="https://github.com/user-attachments/assets/3e7a0448-2095-4564-87ac-1088a7675c58" />

- **Correlation Heatmap**: Multicollinearity check prior to feature selection
<img width="556" height="417" alt="3" src="https://github.com/user-attachments/assets/3610344c-8781-4381-8ae3-9caa19d161ca" />

- **Elbow Curve**: WCSS vs. K plot to justify cluster count selection
<img width="533" height="405" alt="4" src="https://github.com/user-attachments/assets/adea8801-5c2f-4ef9-b075-b029271f7005" />

- **PCA Scatter Plot**: 2D visualization confirming clean separation between the 6 clusters
<img width="527" height="407" alt="5" src="https://github.com/user-attachments/assets/e47b0847-cbfb-459b-8e3f-e77992b3fd99" />

## Tech Stack 💻

| Category | Tools |
|:---|:---|
| **Language** | Python |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn (K-Means, PCA, StandardScaler) |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib / Pickle |


## Conclusion

This project demonstrates end-to-end proficiency in **unsupervised learning, feature engineering, and dimensionality reduction** — translating raw behavioral data into a segmentation framework that directly informs marketing spend and campaign strategy.

**Key Takeaways**:
- Data-driven cluster count selection (Elbow Method) over arbitrary guessing
- PCA-validated cluster separation confirms model reliability, not just statistical convenience
- Personas are directly actionable — each maps to a distinct marketing strategy

**Technologies**: Python • Scikit-Learn • K-Means • PCA • Pandas • Seaborn • Matplotlib

---
