# Data Dictionary for Customer Segmentation Dataset 📊

This data dictionary describes the columns in the `customer_segmentation.csv` file used in the Customer Segmentation via K-Means Clustering project. The dataset contains demographic, behavioral, and campaign-response data for **2,240 customers**, used to engineer features for unsupervised clustering and persona identification.

## Dataset Overview
- **Rows**: 2,240
- **Columns**: 29 (raw)
- **Target Variable**: None (unsupervised) — clusters are derived, not predicted
- **Key Preprocessing Steps**:
  - Missing values: 24 rows with NaN `Income`; imputed or dropped prior to scaling.
  - Feature Engineering: Derived `Total_Spending` (sum of all `Mnt*` columns), `Age` (from `Year_Birth`), and `Customer_Tenure` (from `Dt_Customer`).
  - Outlier handling: Reviewed extreme values in `Income` (max ~$666K) and `Year_Birth` (min 1893) prior to clustering, as K-Means is distance-based and outlier-sensitive.
  - Encoding: `Education` and `Marital_Status` reviewed for consolidation (e.g., grouping rare categories like 'Absurd', 'YOLO' under 'Other').
  - Scaling: All numerical features passed through `StandardScaler` before K-Means (required — unlike tree-based models, K-Means is not scale-invariant).

## Column Descriptions

| Column Name | Data Type | Description | Possible Values/Range | Notes |
|---|:---:|---|---|---|
| ID | int64 | Unique customer identifier. | 1 to ~11,191 | Not used as a feature; dropped prior to clustering. |
| Year_Birth | int64 | Customer's birth year. | 1893 to 1996 | Used to derive `Age`; extreme low values (e.g., 1893) are likely data entry errors and reviewed as outliers. |
| Education | object | Customer's education level. | 'Graduation', 'PhD', 'Master', 'Basic', '2n Cycle' | Categorical; contributes to persona segmentation (e.g., "Affluent Traditionalists"). |
| Marital_Status | object | Customer's relationship status. | 'Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO' | Categorical; 'Absurd' and 'YOLO' are data anomalies, consolidated into 'Other' during cleaning. |
| Income | float64 | Customer's yearly household income. | 1,730 to 666,666 | Numerical; 24 missing values; contains extreme outlier(s) reviewed before scaling. |
| Kidhome | int64 | Number of small children in household. | 0 to 2 | Numerical; used in household-composition feature engineering. |
| Teenhome | int64 | Number of teenagers in household. | 0 to 2 | Numerical; used in household-composition feature engineering. |
| Dt_Customer | object | Date customer enrolled with the company. | 01-01-2013 to 31-12-2013 | Used to derive `Customer_Tenure`; parsed to datetime during cleaning. |
| Recency | int64 | Days since customer's last purchase. | 0 to 99 | Numerical; core behavioral/RFM-style feature. |
| MntWines | int64 | Amount spent on wine in the last 2 years. | 0 to 1,493 | Numerical; contributes to `Total_Spending`. |
| MntFruits | int64 | Amount spent on fruit in the last 2 years. | 0 to 199 | Numerical; contributes to `Total_Spending`. |
| MntMeatProducts | int64 | Amount spent on meat in the last 2 years. | 0 to 1,725 | Numerical; contributes to `Total_Spending`. |
| MntFishProducts | int64 | Amount spent on fish in the last 2 years. | 0 to 259 | Numerical; contributes to `Total_Spending`. |
| MntSweetProducts | int64 | Amount spent on sweets in the last 2 years. | 0 to 263 | Numerical; contributes to `Total_Spending`. |
| MntGoldProds | int64 | Amount spent on gold/premium products in the last 2 years. | 0 to 362 | Numerical; contributes to `Total_Spending`. |
| NumDealsPurchases | int64 | Number of purchases made with a discount. | 0 to 15 | Numerical; signals price sensitivity (key for "Bargain Hunter" persona). |
| NumWebPurchases | int64 | Number of purchases made via the website. | 0 to 27 | Numerical; channel-preference feature. |
| NumCatalogPurchases | int64 | Number of purchases made via catalog. | 0 to 28 | Numerical; channel-preference feature. |
| NumStorePurchases | int64 | Number of purchases made in-store. | 0 to 13 | Numerical; channel-preference feature; key for "Traditionalist" persona. |
| NumWebVisitsMonth | int64 | Number of website visits in the last month. | 0 to 20 | Numerical; high visits + low spend flags "Window Shopper" persona. |
| AcceptedCmp1 | int64 | Whether customer accepted the offer in campaign 1. | 0, 1 | Binary; campaign responsiveness feature. |
| AcceptedCmp2 | int64 | Whether customer accepted the offer in campaign 2. | 0, 1 | Binary; campaign responsiveness feature. |
| AcceptedCmp3 | int64 | Whether customer accepted the offer in campaign 3. | 0, 1 | Binary; campaign responsiveness feature. |
| AcceptedCmp4 | int64 | Whether customer accepted the offer in campaign 4. | 0, 1 | Binary; campaign responsiveness feature. |
| AcceptedCmp5 | int64 | Whether customer accepted the offer in campaign 5. | 0, 1 | Binary; campaign responsiveness feature. |
| Complain | int64 | Whether customer complained in the last 2 years. | 0, 1 | Binary; low-frequency flag, minimal clustering influence. |
| Z_CostContact | int64 | Constant cost-per-contact value. | 3 (constant) | No variance across dataset; dropped prior to modeling (zero predictive/clustering value). |
| Z_Revenue | int64 | Constant revenue-per-response value. | 11 (constant) | No variance across dataset; dropped prior to modeling (zero predictive/clustering value). |
| Response | int64 | Whether customer accepted the offer in the last campaign. | 0, 1 | Binary; held out of clustering features to avoid leakage into unsupervised persona definitions. |

## Additional Notes
- **Constant Columns**: `Z_CostContact` and `Z_Revenue` have zero variance across all 2,240 rows and were excluded from the clustering feature set entirely.
- **Outlier Review**: `Income` and `Year_Birth` contain a small number of extreme, likely erroneous values (e.g., 666,666 income; birth year 1893) that were assessed before scaling, since K-Means is sensitive to outliers unlike tree-based models.
- **Feature Engineering Summary**: Final clustering feature set consolidates the six `Mnt*` spending columns into `Total_Spending`, and derives `Age` and `Customer_Tenure` — reducing dimensionality while preserving behavioral signal for PCA and K-Means.
- **Scaling Requirement**: Unlike the tree-based TikTok classifier, this K-Means pipeline required `StandardScaler` normalization, since Euclidean distance (the basis of K-Means) is highly sensitive to feature magnitude differences (e.g., `Income` in the thousands vs. `Kidhome` in single digits).
- **Data Source**: `pd.read_csv("customer_segmentation.csv")`; customer relationship management (CRM) and campaign-response data used for behavioral clustering.

For details, see the project README or Jupyter Notebook.
