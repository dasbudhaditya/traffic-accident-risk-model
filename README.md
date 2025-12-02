# 🚗 Prediction of Casualty Severity in Road Traffic Accidents

This project builds and compares several machine learning models to predict **casualty severity** in UK road traffic accidents (slight, serious, fatal), supporting an insurance use-case for **Aviva PLC**.   

---

## 🎯 Business Context

Aviva wants to design **risk-sensitive car insurance pricing** based on predicted casualty severity. By predicting whether a casualty is likely to be slight, serious, or fatal, this model can support:

- **Risk-based pricing** of insurance policies  
- **Fraud detection**  
- **Customer segmentation** and targeted interventions  
- Prioritisation of **road safety measures**   

---

## 📊 Data

Prepared inputs from previous group work:   

- Cleaned and merged accident / casualty / vehicle tables  
- Imbalanced target variable:

  - Class 3 (slight): 83.3%  
  - Class 2 (serious): 15.6%  
  - Class 1 (fatal): 1.1%  

- Exported as:
  - `Xtrain.xlsx`, `Xtest.xlsx` (63 features)  
  - `ytrain.xlsx`, `ytest.xlsx` (casualty_severity)

---

## 🧹 Handling Class Imbalance

- Checked class distribution and confirmed **severe imbalance**.   
- Applied **SMOTE oversampling** on the training set to balance classes:

  - Each class oversampled to **9,282 observations**.  

This helps prevent the model from ignoring minority classes (fatal and serious casualties).

---

## 🧠 Models

Four models are built and compared:

1. **Baseline Decision Tree** (shallow depth = 3)   
2. **Random Forest (Model 1)** – tuned via **RandomizedSearchCV**  
3. **Logistic Regression (Model 2)** – tuned via **RandomizedSearchCV**  
4. **Decision Tree (Model 3)** – tuned via **RandomizedSearchCV**  
5. **Random Forest (Model 4)** – tuned via **GridSearchCV**   

Key evaluation metrics (macro-averaged):

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

---

## 📈 Key Results

From the comparison table:   

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Random Forest (Rand) | ~0.7947  | ~0.5174   | ~0.4249| ~0.4502  |
| Decision Tree (tuned)| ~0.7140  | ~0.3902   | ~0.4139| ~0.3979  |
| Baseline DT          | ~0.7627  | ~0.2812   | ~0.3518| ~0.2988  |
| Logistic Regression  | ~0.3953  | ~0.3471   | ~0.4022| ~0.2672  |

- **Random Forest (RandomizedSearchCV)** is selected as the **best model**.  
- GridSearchCV RF gave similar but slightly worse overall metrics.   

---

## 📊 Visualisations

- Confusion matrices for:
  - Baseline DT
  - Tuned RF
  - Tuned Logistic Regression
  - Tuned Decision Tree  
- Bar plots comparing metrics across models.   

---

## 🛠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-Learn (RandomForestClassifier, DecisionTreeClassifier, LogisticRegression, GridSearchCV, RandomizedSearchCV)  
- Imbalanced-Learn (SMOTE)  
- Matplotlib, Seaborn  

---

## 📁 Suggested Folder Structure

```bash
traffic-accident-risk-model/
├── README.md
├── notebooks/
│   ├── 01_load_and_inspect_data.ipynb
│   ├── 02_smote_and_baseline.ipynb
│   ├── 03_model_tuning_random_forest.ipynb
│   ├── 04_model_tuning_logistic_dt.ipynb
│   └── 05_evaluation_and_plots.ipynb
├── data/
│   ├── Xtrain.xlsx
│   ├── Xtest.xlsx
│   ├── ytrain.xlsx
│   └── ytest.xlsx
└── outputs/
    ├── confusion_matrix_rf.png
    ├── model_comparison_barplot.png
    └── other_figures/
```

---

## 📬 Contact

Questions about the project or dataset?  
📧 **budhaditydasfall@gmail.com**  
🔗 [LinkedIn](https://www.linkedin.com/in/dasbudhaditya/)
