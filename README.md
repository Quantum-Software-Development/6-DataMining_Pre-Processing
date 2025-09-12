
<br>

**\[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]**


<br><br>

# 6- [Data Mining]() / Data Cleaning, Preparation and Detection of Anomalies (Outlier Detection)


<!-- ======================================= Start DEFAULT HEADER ===========================================  -->

<br><br>


[**Institution:**]() Pontifical Catholic University of São Paulo (PUC-SP)  
[**School:**]() Faculty of Interdisciplinary Studies  
[**Program:**]() Humanistic AI and Data Science
[**Semester:**]() 2nd Semester 2025  
Professor:  [***Professor Doctor in Mathematics Daniel Rodrigues da Silva***](https://www.linkedin.com/in/daniel-rodrigues-048654a5/)

<br><br>

#### <p align="center"> [![Sponsor Quantum Software Development](https://img.shields.io/badge/Sponsor-Quantum%20Software%20Development-brightgreen?logo=GitHub)](https://github.com/sponsors/Quantum-Software-Development)


<br><br>

<!--Confidentiality statement -->

#

<br><br><br>

> [!IMPORTANT]
> 
> ⚠️ Heads Up
>
> * Projects and deliverables may be made [publicly available]() whenever possible.
> * The course emphasizes [**practical, hands-on experience**]() with real datasets to simulate professional consulting scenarios in the fields of **Data Analysis and Data Mining** for partner organizations and institutions affiliated with the university.
> * All activities comply with the [**academic and ethical guidelines of PUC-SP**]().
> * Any content not authorized for public disclosure will remain [**confidential**]() and securely stored in [private repositories]().  
>


<br><br>

#

<!--END-->




<br><br><br><br>



<!-- PUC HEADER GIF
<p align="center">
  <img src="https://github.com/user-attachments/assets/0d6324da-9468-455e-b8d1-2cce8bb63b06" />
-->


<!-- video presentation -->


##### 🎶 Prelude Suite no.1 (J. S. Bach) - [Sound Design Remix]()

https://github.com/user-attachments/assets/4ccd316b-74a1-4bae-9bc7-1c705be80498

####  📺 For better resolution, watch the video on [YouTube.](https://youtu.be/_ytC6S4oDbM)


<br><br>


> [!TIP]
> 
>  This repository is a review of the Statistics course from the undergraduate program Humanities, AI and Data Science at PUC-SP.
>
>  Access Data Mining [Main Repository](https://github.com/Quantum-Software-Development/1-Main_DataMining_Repository)
> 
>  If you’d like to explore the full materials from the 1st year (not only the review), you can visit the complete repository [here](https://github.com/FabianaCampanari/PracticalStats-PUCSP-2024).
>
>

<!-- =======================================END DEFAULT HEADER ===========================================  -->

<br><br>


Explore datasets from the [**University of California Irvine (UCI) Machine Learning Repository :**](https://archive.ics.uci.edu/ml/index.php) such as the **Balloon**, **Bank Marketing**, and **Mammogram** datasets to practice these concepts of data pre-processing and mining.

<br><br>

## Table of Contents

- [Introduction](#introduction)
- [Common Problems in Raw Data](#common-problems-in-raw-data)
  - [Incompleteness](#incompleteness)
  - [Inconsistency](#inconsistency)
  - [Noise](#noise)
- [Garbage In, Garbage Out (GIGO)](#garbage-in-garbage-out-gigo)
- [Types of Data](#types-of-data)
  - Structured, Semi-Structured, Unstructured
- [Data Attributes and Their Types](#data-attributes-and-their-types)
  - ☞ [Datasets from University of California - Irvine (UCI)](https://archive.ics.uci.edu/datasets)
  - ☞ [Balloon Dataset](https://archive.ics.uci.edu/dataset/13/balloons)
  -  ☞ [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
  -  ☞ [Mammographic Mass Dataset](https://archive.ics.uci.edu/dataset/161/mammographic+mass)
- [Steps of Data Pre-Processing](#steps-of-data-pre-processing)
  - Cleaning
  - Integration
  - Reduction
  - Transformation
  - Discretization
- [Data Cleaning Techniques](#data-cleaning-techniques)
  - Handling Missing Values
  - Noise Reduction Techniques
  - Handling Inconsistencies
- [Data Integration Issues](#data-integration-issues)
- [Data Reduction Techniques](#data-reduction-techniques)
- [Data Standardization & Normalization](#data-standardization--normalization)
- [Discretization](#discretization)
- [Python Code Examples](#python-code-examples)
- [ASCII Diagrams](#ascii-diagrams)


<br><br>


## [Introduction]()

Real-world data are almost always incomplete, inconsistent, and noisy. These problems must be addressed via pre-processing to ensure clean, reliable data, a prerequisite for successful data mining.

The [**pre-processing**]() step manipulates raw data into a form that enables better and more accurate knowledge extraction.


<br>


## [Common Problems in Raw Data]()

<bvr>

### [Incompleteness]()

Missing attribute values, records, or features.

Example: "?" in the credit card field or missing rows.

<br>

### [Inconsistency]()

Contradictory or conflicting entries within the data, e.g., units mixing kg with lbs.

<br>

### [Noise]()

Random variations or errors that obscure real data trends.


<br>


## [Garbage In, Garbage Out (GIGO)]()

Poor quality input data produce poor quality outputs and insights. Cleaning data beforehand is critical.

<br><br>

## Types of Data)]()

<br>

| [Type)]()       | [Description)]()                       | [Examples)]()            |
|-----------------|---------------------------------------|--------------------------|
| Structured      | Fixed fields, clear schema             | CSV, SQL tables          |
| Semi-Structured | Partial structure with markers         | XML, JSON, Emails        |
| Unstructured    | No strict structure or schema          | Text, images, video files|



<br><br>


## [Data Attributes and Their Types)]()

<br>

| [Attribute Type)]() |[Description)]()                   | [Example)]()              |
|----------------|---------------------------------------|-----------------------------|
| Binary         | Two possible values                    | Yes/No, 0/1                  |
| Nominal        | Categorical, no order                  | Marital Status               |
| Ordinal        | Ordered categories                    | Education Level              |
| Ratio          | Numeric with meaningful zero          | Age, Salary                  |



<br><br>


## ☞ [Datasets from University of California - Irvine (UCI)](https://archive.ics.uci.edu/datasets)

### ☞ [Balloon Dataset - CSV](https://github.com/Quantum-Software-Development/6-DataMining_Pre-Processing/blob/270915823e93c7288c6d8c9f7c0f83143570f530/dataset_balao/balao.csv)
### ☞ [Balloon Dataset - Link](https://archive.ics.uci.edu/dataset/13/balloons)

- 20 samples, 5 attributes: color, size, action, age, inflated (True/False).  
- Simple dataset to illustrate basic concepts.

<br>

### ☞ [Bank Marketing Dataset]](https://archive.ics.uci.edu/dataset/222/bank+marketing)

- 4521 samples, 17 attributes related to direct marketing campaigns.
- Predict whether client will subscribe a term deposit (`application`).

<br>

[Example attributes]():

<br>

| [Attribute]()         | [Type]()     | [Description ]()             |
|--------------------|-------------|---------------------------------|
| age                | Numeric     | Client's age                   |
| job                | Categorical | Job type                      |
| marital            | Categorical | Marital Status                |
| education          | Categorical | Education Level               |
| credit             | Binary      | Has credit line (yes/no)      |
| balance            | Numeric     | Account balance               |
| ...                | ...         | ...                          |


<br><br>


###  ☞ [Mammographic Mass Dataset](https://archive.ics.uci.edu/dataset/161/mammographic+mass)

<br>

- 960 samples, 6 attributes related to breast masses.  
- Used for predicting severity (benign/malign).

<br><br>

## Steps of Data Pre-Processing

1. [**Cleaning:**]() Handling missing, noisy, and inconsistent data.  
2. [**Integration:**]() Combine data from multiple sources.  
3. [**Reduction:**]() Reduce dimensionality or data volume.  
4. [**Transformation:**]() Normalize and format data.  
5. [**Discretization:**]() Convert continuous attributes into categorical intervals.


<br><br>


## [Data Cleaning Techniques]()


<br>

### [Handling Missing Values]()

- [**Remove rows**]() with missing data (not recommended if much data lost).  
- [**Manual imputation**]() with domain knowledge.  
- [**Global constant imputation**]() (e.g. zero, -1) — caution advised.  
- [**Hot-deck imputation:**]() Use value from a similar record.  
- [**Last observation carried forward:**]() Use previous valid value.  
- [**Mean/mode imputation:**]() Replace missing with mean (numeric) or mode (categorical).  
- [**Predictive models:**]() Use other attributes to infer missing values.

<br>

### [Noise Reduction Techniques]()

- [**Binning:**]() Group values into intervals (*equal width* or *equal frequency* bins). Replace each value by bin mean or bin boundaries.  
- [**Clustering:**]() Group similar data points; replace with cluster centroid or medoid.  
- [**Approximation:**]() Fit data to smoothing functions like polynomials.

<br>

### [Handling Inconsistent Data]()

- Detect out-of-domain or conflicting values.  
- Correct with manual review or automated scripts verifying domain constraints.  
- Use visualization, statistical summaries, and input from domain experts.


<br><br>


## [Data Integration Issues]()

- [**Redundancy:**]() Duplicate or derivable data attributes or records.  
- [**Duplicity:**]() Exact copies of objects or attributes.  
- [**Conflicts:**]() Different units or representations of the same attribute.  
- Resolve by normalization and unifying units or standards.

<br><br>

## [Data Reduction Techniques]()

- [**Attribute selection:**]() Remove irrelevant or redundant attributes.  
- [**Attribute compression:**]() Use methods like Principal Component Analysis (PCA).  
- [**Data volume reduction:**]() Use sampling, clustering, or parametric models.  
- [**Discretization:**]() Convert continuous data to intervals.

<br><br>


## [Data Standardization & Normalization]()

[**Normalization**]() rescales data for algorithm compatibility:

<br>

### [Max-Min Normalization]()

Maps value \(a\) to \(a'\) in new range \([new_{min}, new_{max}]\):


<br>

$$
\Huge
a' = \frac{a - min_a}{max_a - min_a} \times (new_{max} - new_{min}) + new_{min}
$$


<br>

```latex
a' = \frac{a - min_a}{max_a - min_a} \times (new_{max} - new_{min}) + new_{min}
```

<br><br>


### [Z-Score Normalization]()

Centers attribute around zero and scales by standard deviation:

<br>

$$
\Huge
a' = \frac{a - \bar{a}}{\sigma_a}
$$

<br>

```latex
a' = \frac{a - \bar{a}}{\sigma_a}
```


<br><br>


## [Discretization]()

<br>

- Converts numeric attributes into categorical bins.  
- Methods include equal-width, equal-frequency, histogram-based, and entropy-based discretization.

<br><br>


> [!TIP]
> 
> ☞ [Get Discretization Coce](https://github.com/Quantum-Software-Development/6-DataMining_Pre-Processing/blob/c08d18552890d94b2b64858a4776ebf2d2da9736/Discretization_Code/Discretization_Method.ipynb) + [Cancer Dataset](https://github.com/Quantum-Software-Development/6-DataMining_Pre-Processing/blob/ed46c81806fc425a51b46e8b4859ce34e0b3ebd6/Discretization_Code/cancer.csv)
>
> 
>

<br><br>


## [Python Code Examples]()


<br>


```python
import pandas as pd
from sklearn.preprocessing import minmax_scale, scale

# Load Bank Marketing dataset from UCI (use your local copy or URL)

# Example URL requires downloading and preprocessing: demonstration uses local CSV

data_path = "bank-additional-full.csv"
df = pd.read_csv(data_path, sep=';')

# Drop unnamed columns (typical from CSV exports)

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Handling missing values example:

# Replace '?' with NaN for easier manipulation

df.replace('?', pd.NA, inplace=True)

# Remove rows with any missing values (not recommended if too many deleted)

df_clean = df.dropna()

# Or imputing missing values with mode for categorical attribute, e.g. 'job'

df['job'].fillna(df['job'].mode(), inplace=True)

# Max-Min normalization for numeric columns

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
df[num_cols] = df[num_cols].apply(minmax_scale)

# Z-score normalization example

df[num_cols] = df[num_cols].apply(scale)

# Discretization example - age into 5 bins

df['age_binned'] = pd.cut(df['age'], bins=5, labels=False)

# Drop columns example: if any 'Unnamed' columns exist

df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
```


<br><br>


## [ASCII Diagrams]()

### [Data Pre-Processing Flow]()

<br>

```
+-------------------+
| Define Problem    |
+---------+---------+
|
v
+-------------------+
| Select Data       |
+---------+---------+
|
v
+-------------------+     +--------------------------+
| Choose Algorithm  +---->+ Apply Preprocessing Steps |
+---------+---------+     +--------------------------+
|
v
+-------------------+
| Apply Data Mining |
+-------------------+
```

<br><br>

### [Binning (Equal Width)]()

<br>

```python
Values: 2, 3, 7, 8, 11, 12, 16, 18, 20
Bins: 3 (width = 6)

Bin 1: 2 - 7  -> 2, 3, 7
Bin 2: 8 - 13 -> 8, 11, 12
Bin 3: 14 - 20->16, 18, 20
```

<br><br>

### [Clustering Prototypes: Centroid vs Medoid]()

<br>

```python
(a) Centroid                     (b) Medoid

*                             *
    
*o*                           ooo
oooo                          o o o

Centroid = mean point (artificial)
Medoid  = actual central object
```


<br><br>

## [Summary]()

Data pre-processing ensures the quality of mining outcomes by addressing missing, noisy, and inconsistent data; integrating multiple data sources; reducing data size and dimensionality; normalizing; and discretizing attributes for algorithm compatibility.

[UCI]() datasets like [**Balloon**](), [**Bank Marketing**](), and [**Mammographic Mass**]() provide excellent real-world scenarios to practice these techniques using Python.



<br><br>


# [I]()- Data Mining Pre-Processing - [Code  with Breast Cancer Dataset]()

This code notebook guides you through essential data pre-processing steps applied to datasets similar to those from the UCI Machine Learning Repository, such as Bank Marketing or Breast Cancer datasets. The steps cover cleaning, handling missing values, normalizing, discretizing, and preparing data for mining.

<br>

### 1. [Import Required Libraries]()

<br>

```python
# Import libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale, scale
```

<br>

### 2.  [Load Dataset]()

<br>

```python
# Load dataset from local or online source

# Example: Breast Cancer Wisconsin dataset (adjust path or URL as needed)

url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/cancer.csv'  \# Example URL
dados = pd.read_csv(url)

# Show first rows to inspect data

dados.head()

```

<br>

## 3.  [Inspect Columns and Data Types]()

<br>

```python
# List columns and data types

print(dados.dtypes)

# Check for 'Unnamed' columns (common from CSV exports)

unnamed_cols = [col for col in dados.columns if "Unnamed" in col]
print("Unnamed columns:", unnamed_cols)

```

<br>


## 4.  [Drop Unnamed Columns]()

<br>

```python
# Drop unnamed columns if any

dados.drop(columns=unnamed_cols, inplace=True)
print("Columns after dropping unnamed ones:", dados.columns)
```

<br>

### 5.  [Handling Missing Values]()

<br>

```python
# Check for missing values per column

print(dados.isnull().sum())

# Example: Impute missing values using mean for numerical columns

num_cols = dados.select_dtypes(include=['float64', 'int64']).columns.tolist()

for col in num_cols:
mean_value = dados[col].mean()
dados[col].fillna(mean_value, inplace=True)

# For categorical columns, impute mode if needed

cat_cols = dados.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
mode_value = dados[col].mode()
dados[col].fillna(mode_value, inplace=True)

# Verify no missing values left

print(dados.isnull().sum())
```


<br>

### 6.  [Remove Rows with Missing Data (Alternative approach)]()

<br>

```python
# Alternatively, remove rows with any missing value

dados_clean = dados.dropna()
print(f"Data shape after dropping missing values: {dados_clean.shape}")
```

<br>

### 7.  [Normalization: Max-Min Scaling]()

<br>

```python
# Select numeric columns for normalization, excluding identifiers or labels

cols = list(dados.columns)

# Example: remove columns if they are IDs or target variables

if 'id' in cols: cols.remove('id')
if 'diagnosis' in cols: cols.remove('diagnosis')

# Apply Min-Max normalization to selected columns

dados_minmax = dados.copy()
dados_minmax[cols] = dados_minmax[cols].apply(minmax_scale)

dados_minmax.head()
```


<br>

### 8.  [Normalization: Z-Score Standardization]()

<br>

```python
# Apply Z-score standardization: mean=0, std=1

dados_zscore = dados.copy()
dados_zscore[cols] = dados_zscore[cols].apply(scale)

dados_zscore.head()
```

<br>

### 9.  [Discretization of Continuous Attributes]()

<br>

```python
# Example: Discretizing 'radius_mean' into 10 equal-width bins with labels 0-9

if 'radius_mean' in dados.columns:
dados['radius_mean_binned'] = pd.cut(dados['radius_mean'], bins=10, labels=range(10))
print(dados[['radius_mean', 'radius_mean_binned']].head())
else:
print('Column radius_mean not found in dataset')

```


<br>

### 10.  [Handling Categorical Variables]()

<br>

```python
# Example: Encoding categorical columns (if needed)

categorical_cols = dados.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns:", categorical_cols)

# Convert categorical columns to category dtype

for col in categorical_cols:
dados[col] = dados[col].astype('category')

# Display categorical columns with their unique values

for col in categorical_cols:
print(f"{col} unique values: {dados[col].cat.categories}")
```

<br>

### 11.  [Summary: Data Ready for Mining]()

<br>

```python
# Final look at data shape and first few rows after pre-processing

print("Data shape:", dados.shape)
dados.head()
```


<br><br><br><br>




# [II]()- Data Mining Pre-Processing - [Code  with  Bank Marketing Dataset]()

<br>

You can copy this script into Google Colab, run cell-by-cell, and get a complete data pre-processing example using a real UCI dataset.


<br>


### - [**Cell 1:**]() Imports all necessary libraries.

```python
# Cell 1 - Import libraries
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from sklearn.preprocessing import minmax_scale, scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
```

<br>

### - [**Cell 2:**]() Downloads and extracts the Bank Marketing dataset from UCI and loads it into a DataFrame.


<br>

```python
# Cell 2 - Download and load UCI Bank Marketing dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall('bank_data')

df = pd.read_csv('bank_data/bank-full.csv', sep=';')
print("Loaded dataset shape:", df.shape)
df.head()
```


<br>

### - [**Cell 3:**]() Detects and removes any extraneous unnamed columns.racts the Bank Marketing dataset from UCI and loads it into a DataFrame.


<br>


```python
# Cell 3 - Check for unnamed or unwanted columns and remove them
unnamed_cols = [col for col in df.columns if "Unnamed" in col]
print("Unnamed columns:", unnamed_cols)
if unnamed_cols:
    df.drop(columns=unnamed_cols, inplace=True)
print("Columns after drop:", df.columns.tolist())
```


<br>

### - [**Cell 4:**]() Identifies `'unknown'` values as missing and replaces with `NaN`.


<br>

```python
# Cell 4 - Explore missing values and replace 'unknown' with NaN
print("Count 'unknown' per column before replacement:")
print((df == 'unknown').sum())

df.replace('unknown', np.nan, inplace=True)  # Mark 'unknown' as NaN

print("Count missing values after replacement:")
print(df.isnull().sum())
```


<br>

### - [**Cell 5:**]() Imputes missing categorical values with the mode of each column


<br>

```python
# Cell 5 - Imputation of missing values (fill NaN for categorical columns with mode)
cat_cols = df.select_dtypes(include='object').columns.tolist()
print("Categorical columns:", cat_cols)

for col in cat_cols:
    mode_val = df[col].mode()[^0]
    df[col].fillna(mode_val, inplace=True)

print("Missing values after imputation:")
print(df.isnull().sum())
```


<br>

### - [**Cell 6:**]() Prints unique values of categorical columns to check for inconsistencies.

<br>

```python
# Cell 6 - Check unique values for categorical attributes to spot inconsistencies
for col in cat_cols:
    print(f"Unique values in '{col}':")
    print(df[col].unique())
    print('------')
```


<br>

### - [**Cell 7:**]() Demonstrates binning — converting continuous `age` into categorical bins.

<br>

```python
# Cell 7 - Binning the 'age' variable into 5 equal-width intervals
df['age_binned'] = pd.cut(df['age'], bins=5, labels=range(5))
print(df[['age', 'age_binned']].head())
````

<br>

### - [**Cell 8:**]() Applies Min-Max normalization to numeric columns.

<br>

```python
# Cell 8 - Normalize numeric attributes using Max-Min scaling
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Numeric columns:", num_cols)

df_minmax = df.copy()
df_minmax[num_cols] = df_minmax[num_cols].apply(minmax_scale)
print(df_minmax[num_cols].head())
```

<br>

### - [**Cell 9:**]() Applies standard Z-score normalization to numeric columns.

<br>

```python
# Cell 9 - Normalize numeric attributes using Z-score standardization
df_zscore = df.copy()
df_zscore[num_cols] = df_zscore[num_cols].apply(scale)
print(df_zscore[num_cols].head())
```





<br>

```python
# Cell 10 - Encode categorical variables before PCA
df_encoded = pd.get_dummies(df.drop(columns=['age_binned']))  # drop binned for now
print("Shape after encoding:", df_encoded.shape)
```




<br>

```python
# Cell 11 - Apply PCA for dimensionality reduction (2 components) and visualize
X = df_encoded.select_dtypes(include=[np.number])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance ratio of PCA components:", pca.explained_variance_ratio_)

plt.figure(figsize=(8,5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
plt.title('PCA projection onto 2 components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```












<!-- ========================== [Bibliographr ====================  -->

<br><br>


## [Bibliography]()

[1](). **Castro, L. N. & Ferrari, D. G.** (2016). *Introduction to Data Mining: Basic Concepts, Algorithms, and Applications*. Saraiva.

[2](). **Ferreira, A. C. P. L. et al.** (2024). *Artificial Intelligence – A Machine Learning Approach*. 2nd Ed. LTC.

[3](). **Larson & Farber** (2015). *Applied Statistics*. Pearson.

<br><br>

      
<!-- ======================================= Bibliography Portugues ===========================================  -->

<!--

## [Bibliography]()


[1](). **Castro, L. N. & Ferrari, D. G.** (2016). *Introdução à mineração de dados: conceitos básicos, algoritmos e aplicações*. Saraiva.

[2](). **Ferreira, A. C. P. L. et al.** (2024). *Inteligência Artificial - Uma Abordagem de Aprendizado de Máquina*. 2nd Ed. LTC.

[3](). **Larson & Farber** (2015). *Estatística Aplicada*. Pearson.


<br><br>
-->

<!-- ======================================= Start Footer ===========================================  -->


<br><br>


## 💌 [Let the data flow... Ping Me !](mailto:fabicampanari@proton.me)

<br><br>



#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)


<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />




<br><br><br>

<p align="center">  ────────────── 🔭⋆ ──────────────


<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>

<!--
<p align="center">  ────────────── ✦ ──────────────
-->



<!-- Programmers and artists are the only professionals whose hobby is their profession."

" I love people who are committed to transforming the world "

" I'm big fan of those who are making waves in the world! "

##### <p align="center">( Rafael Lain ) </p>   -->

#

###### <p align="center"> Copyright 2025 Quantum Software Development. Code released under the [MIT License license.](https://github.com/Quantum-Software-Development/Math/blob/3bf8270ca09d3848f2bf22f9ac89368e52a2fb66/LICENSE)










