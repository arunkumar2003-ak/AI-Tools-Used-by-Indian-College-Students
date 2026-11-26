# AI/ML Capstone Project - Part 1
## AI Tools Used by Indian College Students - Data Preprocessing and Exploratory Data Analysis

### Student Information

- **Student Name:** Arun Kumar T
- **Course:** B.Tech Artificial Intelligence and Machine Learning
- **Project Title:** AI Tools Used by Indian College Students

---

# Project Overview

The objective of this project is to analyze the usage of Artificial Intelligence (AI) tools among Indian college students. Part 1 focuses on data preprocessing and exploratory data analysis (EDA). The dataset was cleaned, analyzed, visualized, and prepared for machine learning tasks in later stages.

---

# Dataset Information

The dataset contains information about Indian college students including:

- Student Name
- College Name
- Stream
- Year of Study
- AI Tools Used
- Daily Usage Hours
- Use Cases
- Trust in AI Tools
- Impact on Grades
- Professors' Permission
- Preferred AI Tool
- Awareness Level
- Willingness to Pay
- State
- Device Used
- Internet Access

Dataset Size:

- Rows: **3614**
- Columns: **16**

---

# Data Cleaning

## Missing Value Analysis

Missing values were identified using:

```python
df.isnull().sum()
```

Columns with less than 20% missing values were imputed using the **median**.

### Why Median?

The median is more robust to outliers and skewed distributions than the mean. Since survey data may contain extreme values, the median provides a more reliable estimate for missing-value imputation.

---

## Duplicate Removal

Duplicate records were detected using:

```python
df.duplicated().sum()
```

Duplicates were removed using:

```python
df.drop_duplicates()
```

After duplicate removal, the null percentage of each column remained unchanged, indicating that duplicate removal improved data quality without affecting the missing-value distribution.

---

## Data Type Conversion

Numeric columns already had appropriate data types.

Several repetitive string columns were converted from **object** to **category** to improve memory efficiency.

Although the memory reduction was minimal for this dataset, using categorical data types is considered a best practice for larger datasets.

---

# Descriptive Statistics

Descriptive statistics were generated using:

```python
df.describe()
```

The numeric variables were summarized using:

- Count
- Mean
- Standard Deviation
- Minimum
- Quartiles
- Maximum

---

# Skewness Analysis

Skewness was calculated for every numeric column.

The highest absolute skewness was observed in:

- **Awareness_Level**
- Skewness = **-0.15283**

This indicates a slightly negatively skewed distribution. Since the skewness is close to zero, the distribution is approximately symmetric.

The median was selected for missing-value imputation because it is less sensitive to skewness and extreme values than the mean.

---

# Outlier Detection

The Interquartile Range (IQR) method was used to detect outliers.

The following statistics were calculated:

- Q1
- Q3
- IQR
- Lower Bound
- Upper Bound

Outliers were documented but not removed.

These observations may represent genuine student behavior rather than data errors. Outliers will be retained initially and reviewed again during model development in Part 2.

---

# Data Visualization

The following visualizations were created and saved in the **plots/** folder.

1. Line Plot
2. Bar Chart
3. Histogram
4. Scatter Plot
5. Box Plot
6. Correlation Heatmap

---

## Histogram Interpretation

The histogram of **Awareness_Level** shows a nearly symmetric distribution with a slight negative skew.

---

## Scatter Plot Interpretation

The scatter plot between **Daily Usage Hours** and **Trust in AI Tools** was created to examine the relationship between usage behavior and trust.

---

## Box Plot Interpretation

The box plot compares daily AI usage across different academic streams.

Differences in median and spread indicate variation in AI usage among different student groups.

---

# Correlation Analysis

Pearson correlation was computed for all numeric variables.

A correlation heatmap was generated and saved in the **plots** folder.

Correlation measures association but does not imply causation. External factors such as awareness level, academic stream, internet access, and professors' policies may influence the observed relationships.

---

# Spearman Correlation

Spearman rank correlation was computed to compare monotonic relationships with Pearson's linear relationships.

The differences between Pearson and Spearman correlations were analyzed to identify possible non-linear monotonic relationships.

Spearman correlation will be considered alongside Pearson correlation during feature selection in Part 2 whenever monotonic relationships are stronger than linear relationships.

---

# Grouped Aggregation

The dataset was grouped by **Stream** and summarized for **Daily_Usage_Hours**.

### Highest Mean

- Pharmacy
- Mean Daily Usage = **2.744 hours**

### Highest Standard Deviation

- Management
- Standard Deviation = **1.328**

### Mean Ratio

Highest Mean / Lowest Mean = **1.22**

The relatively small ratio suggests that **Stream** alone provides only moderate predictive information and should be combined with other features.

---

# Clean Dataset

After preprocessing, the cleaned dataset was saved as:

```
cleaned_data.csv
```

using

```python
df.to_csv("cleaned_data.csv", index=False)
```

This cleaned dataset will be used for feature engineering and machine learning in Part 2.

---

# Project Structure

```
AIML CAPSTONE PROJECT/
│
├── Part1.ipynb
├── original_dataset.csv
├── cleaned_data.csv
├── README.md
│
└── plots/
    ├── line_plot_daily_usage.png
    ├── bar_chart_stream_usage.png
    ├── histogram_awareness_level.png
    ├── scatter_usage_trust.png
    ├── boxplot_stream_usage.png
    └── correlation_heatmap.png
```

---

# Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Conclusion

In Part 1, the dataset was successfully cleaned and prepared for machine learning. Missing values were handled using median imputation, duplicates were removed, categorical variables were optimized, descriptive statistics and skewness were analyzed, outliers were documented, visualizations were created, and correlation analyses were performed. The resulting cleaned dataset is ready for feature engineering, model training, and evaluation in Part 2.