
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

- **Binning:** Group values into intervals (*equal width* or *equal frequency* bins). Replace each value by bin mean or bin boundaries.  
- **Clustering:** Group similar data points; replace with cluster centroid or medoid.  
- **Approximation:** Fit data to smoothing functions like polynomials.

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

https://github.com/Quantum-Software-Development/6-DataMining_Pre-Processing/blob/7abc5ba87cef15c188d47c6a1eb66752bfbf9980/Discretization_Code/Discretization_Method.ipynb

- Converts numeric attributes into categorical bins.  
- Methods include equal-width, equal-frequency, histogram-based, and entropy-based discretization.

> [!TIP]
> 
> ☞ [Get Discretization Coce](https://github.com/Quantum-Software-Development/6-DataMining_Pre-Processing/blob/c08d18552890d94b2b64858a4776ebf2d2da9736/Discretization_Code/Discretization_Method.ipynb) + [Dataset](https://github.com/Quantum-Software-Development/6-DataMining_Pre-Processing/blob/ed46c81806fc425a51b46e8b4859ce34e0b3ebd6/Discretization_Code/cancer.csv)
>

<br><br>


##[Python Code Examples]()


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























<br><br><br><br>
<br><br><br><br>
<br><br><br><br>
<br><br><br><br>
<br><br><br><br>


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










