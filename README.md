# ⚖️ Fair Salary Assessment & Anomaly Detection System

A Machine Learning-powered web application designed to assist HR departments in predicting employee salary categories and identifying potential compensation anomalies.

## 📌 Overview
This project leverages historical census data to predict whether an employee's salary exceeds **$50,000/year**. Built with **Streamlit** and **Scikit-Learn**, it provides a user-friendly interface for both individual assessments and bulk analysis, offering transparency through feature importance visualizations.

## 🚀 Key Features
* **Real-time Prediction:** Input employee details via an interactive sidebar to get instant salary classification (`<=50K` or `>50K`).
* **Batch Processing:** Upload a CSV file to generate predictions for hundreds of employees simultaneously.
* **Model Explainability:** Visualizes "Feature Importance" to show exactly which factors (e.g., Education, Capital Gain) influenced the model's decision.
* **Data Preprocessing:** Automatically handles missing values, removes outliers, and encodes categorical data to match the training pipeline.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (Gradient Boosting Classifier)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure
* `app.py` - The main Streamlit application file.
* `best_model.pkl` - The trained machine learning pipeline (StandardScaler + GradientBoosting).
* `adult.csv` - The dataset used for training (Adult Census Income dataset).
* `requirements.txt` - List of dependencies.

## ⚙️ Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/salary-prediction.git](https://github.com/yourusername/salary-prediction.git)
    cd salary-prediction
    ```

2.  **Install Dependencies**
    ```bash
    pip install streamlit pandas scikit-learn matplotlib seaborn joblib
    ```

3.  **Train the Model (First Run Only)**
    If you don't have `best_model.pkl` yet, run the training script included in `app.py` or your separate training notebook to generate it.
    *(Note: Ensure `adult.csv` is in the root directory)*.

4.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

## 📊 How It Works
1.  **Data Input:** The user provides demographic info (Age, Workclass, Education, etc.).
2.  **Preprocessing:** The app scales numerical values and encodes categorical text (e.g., "Private Sector" → `3`) using pre-defined mappings.
3.  **Inference:** The data is passed through a **Gradient Boosting Classifier**.
4.  **Output:** The app displays the predicted class, confidence probability, and a graph of the top influencing features.

## 📈 Model Performance
The model uses a **Pipeline** approach with `StandardScaler` and `GradientBoostingClassifier`, optimized for the Adult Census dataset. It filters outliers (e.g., removing ages > 75) to ensure robust predictions.

## 👨‍💻 Author
**Arijit Bhaya**
* [LinkedIn](https://www.linkedin.com/in/arijitbhaya)
* [GitHub](https://github.com/Arijit123789)
