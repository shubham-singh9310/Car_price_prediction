# Car_price_prediction
# 🚗 Car Price Prediction using Machine Learning

A Machine Learning project that predicts used car selling prices based on features like year of manufacture, fuel type, transmission, and more — trained using both Linear Regression and Random Forest Regressor models.

## 📌 Project Overview

This project was built as part of my Machine Learning learning journey. The goal is to predict the **selling price of a used car** given various attributes, using supervised regression techniques.

The model was trained on a real-world dataset and evaluated using standard regression metrics. The **Random Forest Regressor** achieved an impressive **R² score of ~0.96**, demonstrating strong predictive performance.

## 📊 Dataset

**File:** "car data.csv"

The dataset contains the following features:

| Feature | Description |

 `Car_Name` -  Name of the car (dropped during preprocessing) |
 `Year` - Year of manufacture |
 `Selling_Price` - Target variable — price at which the car is sold (in Lakhs) |
 `Present_Price` - Current ex-showroom price of the car (in Lakhs) |
 `Kms_Driven` - Total kilometers driven |
 `Fuel_Type` - Fuel type — Petrol / Diesel / CNG |
 `Seller_Type` - Dealer or Individual |
 `Transmission` - Manual or Automatic |
 `Owner` - Number of previous owners |

---

## 🧠 ML Workflow

1. Data Loading & Exploration
- Loaded the dataset using **Pandas**
- Inspected the first few rows with `df.head()`
- Checked for missing values — **no nulls found**

2. Data Preprocessing
- Dropped the `Car_Name` column (non-numeric, high cardinality)
- Applied **One-Hot Encoding** using `pd.get_dummies()` on categorical features:
  - `Fuel_Type` → `Fuel_Type_Diesel`, `Fuel_Type_Petrol`
  - `Seller_Type` → `Seller_Type_Individual`
  - `Transmission` → `Transmission_Manual.`

3. Feature & Target Split
```python
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
```

4. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

5. Model Training
Two models were trained and compared:

| Model | Description |
|---|---|
| **Linear Regression** | Baseline model |
| **Random Forest Regressor** | Ensemble model (best performer) |

6. Evaluation Metrics (Random Forest)

| Metric | Score |
|---|---|
| MAE (Mean Absolute Error) | **0.6325** |
| MSE (Mean Squared Error) | **0.9217** |
| R² Score | **0.9600** |

7. Visualization
A scatter plot was generated comparing **Actual vs Predicted Prices**, along with a perfect prediction reference line.

🛠️ Tech Stack

- **Language:** Python 3.13
- **Libraries:**
  - `pandas` — Data manipulation
  - `numpy` — Numerical operations
  - `matplotlib` & `seaborn` — Data visualization
  - `scikit-learn` — ML models & evaluation

🚀 Getting Started

Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the Notebook
```bash
jupyter notebook car_price_prediction.ipynb
```

Make sure `car data.csv` is in the same directory as the notebook.


📁 Project Structure

```
car-price-prediction/
│
├── car_price_prediction.ipynb   # Main Jupyter Notebook
├── car data.csv                 # Dataset
└── README.md                    # Project documentation
```

---

📈 Results

The **Random Forest Regressor** significantly outperformed Linear Regression with an R² score of **~0.96**, meaning the model explains 96% of the variance in car selling prices.

The Actual vs Predicted scatter plot confirms predictions closely align with real values, particularly for lower-priced vehicles.


🙋‍♂️ Author

**Shubham Singh**
- 🔗 [LinkedIn](https://www.linkedin.com/in/shubham-singh38)
- 💻 Python Developer | AI & ML Engineer

---

📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

> ⭐ If you found this project helpful, consider giving it a star!
