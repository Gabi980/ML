# Machine Learning Project

Tema propusa: **Platforma de predictii de business folosind Machine Learning**.

Proiectul are doua module:

- regresie: predictia pretului unei case din King County;
- clasificare: predictia churn-ului unui client telecom.

## Cerinte acoperite

- incarcarea si intelegerea dataseturilor prin sumar statistic si vizualizari;
- pregatirea dataseturilor prin transformari studiate la laborator;
- antrenarea si compararea a 5 modele ML pentru fiecare dataset;
- folosirea a 2 metode ensemble pentru fiecare dataset;
- salvarea celui mai bun model;
- GUI minimal pentru predictii.

## Dataseturi

Descarca manual CSV-urile din Kaggle si pune-le in `data/raw`:

- House Sales in King County, USA: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
- Telco Customer Churn: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Nume recomandate:

- `data/raw/kc_house_data.csv`
- `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Instalare

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Rulare training

```powershell
python -m src.train
```

Optional, pentru evaluare mai apropiata de laborator:

```powershell
python -m src.train --cv 10
```

Rezultatele apar in:

- `reports/tables`: tabele CSV cu sumar statistic, valori lipsa, comparatii modele;
- `reports/figures`: grafice EDA;
- `reports/project_summary.md`: sinteza proiectului;
- `outputs/models`: cele mai bune modele salvate.

## Rulare GUI

Dupa training:

```powershell
streamlit run app.py
```

Aplicatia are doua taburi:

- House Price Regression;
- Telco Churn Classification.

## Modele folosite

Regresie:

- Linear Regression;
- Ridge Regression;
- Lasso Regression;
- KNN Regressor;
- Decision Tree Regressor;
- Random Forest Regressor;
- AdaBoost Regressor.

Clasificare:

- Logistic Regression;
- KNN Classifier;
- Gaussian Naive Bayes;
- Decision Tree Classifier;
- Support Vector Machine;
- Random Forest Classifier;
- AdaBoost Classifier.
