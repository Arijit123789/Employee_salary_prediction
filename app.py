# --- Step 1: Install necessary libraries (if you haven't already) ---
# Run this cell if you get ModuleNotFoundError for any of the libraries.
# You might need to restart your kernel after installation.
# !pip install streamlit pandas scikit-learn matplotlib seaborn joblib

# --- Step 2: Create the app.py file from within Jupyter using standard Python file I/O ---
# This cell will write the content of your Streamlit app into a file named 'app.py'
# in the same directory as your Jupyter Notebook.
# MAKE SURE TO RUN THIS CELL TO OVERWRITE THE OLD APP.PY FILE.

# Changed outer quotes to triple single quotes (''') to avoid conflict with inner triple double quotes (""")
app_content = '''
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline # Import Pipeline to check model type

# The problematic line 'st.set_option('deprecation.showPyplotGlobalUse', False)' has been removed.

# --- Data Preprocessing Functions (Replicated from Notebook for Consistency) ---
# These functions ensure that the input data for the Streamlit app
# is preprocessed in the same way as the training data.

def preprocess_data(df):
    """Applies the same preprocessing steps as the notebook."""
    # Handle '?' values
    df['workclass'].replace({'?': 'Others'}, inplace=True)
    df['occupation'].replace({'?': 'Others'}, inplace=True)
    df['native-country'].replace({'?': 'Others'}, inplace=True) # Ensure this is handled if present

    # Remove specific workclass rows (as done in notebook)
    df = df[df['workclass'] != 'Without-pay']
    df = df[df['workclass'] != 'Never-worked']

    # Outlier removal for 'age' and 'educational-num' (as done in notebook)
    df = df[(df['age'] <= 75) & (df['age'] >= 17)]
    df = df[(df['educational-num'] <= 16) & (df['educational-num'] >= 5)]

    # Drop 'education' column (redundant with 'educational-num')
    if 'education' in df.columns:
        df = df.drop(columns=['education'])

    # Label Encoding for categorical features
    categorical_cols = [
        'workclass', 'marital-status', 'occupation',
        'relationship', 'race', 'gender', 'native-country'
    ]
    for col in categorical_cols:
        if col in df.columns:
            # Create a unique encoder for each column to avoid issues with unseen labels
            # In a real-world scenario, you'd save/load these encoders.
            # For this demo, we'll fit them on a dummy full dataset for consistency.
            # A more robust solution would involve saving the fitted encoders during training.
            le = LabelEncoder()
            # To ensure consistent encoding, fit on all possible values if known,
            # or handle unknown values gracefully. For this example, we'll assume
            # the app inputs will be within the trained categories.
            # A safer approach for deployment is to use OneHotEncoder or pre-fit LabelEncoders.
            # For simplicity in this demo, we'll just fit_transform directly.
            # This is a simplification and might need adjustment for production.
            df[col] = le.fit_transform(df[col])
    return df

# --- Streamlit App Layout ---
st.set_page_config(page_title="Fair Salary Assessment & Anomaly Detection", page_icon="⚖️", layout="wide")

st.title("⚖️ Fair Salary Assessment & Anomaly Detection System")
st.markdown("""
This application leverages machine learning to predict employee salary categories
(<=50K or >50K) and provides insights to help HR departments assess fairness
and identify potential compensation anomalies.
""")

# Load the trained model
# Note: The model was trained on scaled data, so inputs must be scaled before prediction.
# The original notebook used a Pipeline, so the scaler is part of the loaded model.
try:
    model = joblib.load("best_model.pkl")
    # Check if the loaded model is a Pipeline
    if not isinstance(model, Pipeline):
        st.warning("Warning: The loaded model is not a scikit-learn Pipeline. "
                   "Ensure 'best_model.pkl' contains a Pipeline with a 'scaler' and 'model' step "
                   "for proper preprocessing and feature importance extraction.")
except FileNotFoundError:
    st.error("Error: Model file 'best_model.pkl' not found. Please ensure the model is trained and saved.")
    st.stop()

# --- Single Employee Prediction ---
st.header("👤 Individual Employee Assessment")
st.markdown("Enter an employee's details to predict their salary category and understand contributing factors.")

# Sidebar inputs (these must match your training feature columns and their encoded values)
# For the Streamlit app, we need to map user-friendly labels back to the numerical encodings
# used by the LabelEncoder during training. This is a crucial step for consistency.

# Define the mappings based on the original notebook's data and typical label encoding order
# (assuming alphabetical order for LabelEncoder, which is common but not guaranteed without explicit mapping)
workclass_map = {'Private': 3, 'Self-emp-not-inc': 5, 'Local-gov': 1, 'Others': 2, 'State-gov': 6, 'Self-emp-inc': 4, 'Federal-gov': 0}
marital_status_map = {'Never-married': 4, 'Married-civ-spouse': 2, 'Divorced': 0, 'Separated': 5, 'Widowed': 6, 'Married-spouse-absent': 3, 'Married-AF-spouse': 1}
occupation_map = {
    'Prof-specialty': 10, 'Craft-repair': 2, 'Exec-managerial': 3, 'Adm-clerical': 0,
    'Sales': 12, 'Other-service': 8, 'Machine-op-inspct': 6, 'Others': 9,
    'Transport-moving': 14, 'Handlers-cleaners': 5, 'Farming-fishing': 4,
    'Tech-support': 13, 'Protective-serv': 11, 'Priv-house-serv': 7, 'Armed-Forces': 1
}
relationship_map = {'Husband': 0, 'Not-in-family': 1, 'Own-child': 3, 'Unmarried': 4, 'Wife': 5, 'Other-relative': 2}
race_map = {'White': 4, 'Black': 2, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 0, 'Other': 3}
gender_map = {'Male': 1, 'Female': 0}
native_country_map = {
    'United-States': 39, 'Mexico': 27, 'Philippines': 31, 'Germany': 10, 'Puerto-Rico': 33,
    'Canada': 7, 'El-Salvador': 8, 'India': 18, 'Cuba': 5, 'England': 9, 'Jamaica': 22,
    'South': 36, 'China': 3, 'Italy': 20, 'Dominican-Republic': 6, 'Vietnam': 40,
    'Guatemala': 14, 'Japan': 23, 'Poland': 32, 'Columbia': 4, 'Taiwan': 37, 'Haiti': 15,
    'Iran': 19, 'Portugal': 30, 'Nicaragua': 28, 'Peru': 29, 'Greece': 13,
    'France': 11, 'Ecuador': 0, 'Ireland': 21, 'Hong': 16, 'Cambodia': 2, 'Trinadad&Tobago': 38,
    'Laos': 24, 'Thailand': 3, 'Yugoslavia': 41, 'Outlying-US(Guam-USVI-etc)': 2,
    'Hungary': 17, 'Honduras': 1, 'Scotland': 34, 'Holand-Netherlands': 1, 'Others': 25 # Assuming 'Others' for unknown countries
}


with st.sidebar:
    st.header("Input Employee Details")
    age = st.slider("Age", 17, 75, 30) # Adjusted min/max based on outlier removal
    workclass_label = st.selectbox("Workclass", list(workclass_map.keys()))
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=200000)
    educational_num = st.slider("Educational Number (Years of Education)", 5, 16, 9) # Adjusted min/max based on outlier removal
    marital_status_label = st.selectbox("Marital Status", list(marital_status_map.keys()))
    occupation_label = st.selectbox("Occupation", list(occupation_map.keys()))
    relationship_label = st.selectbox("Relationship", list(relationship_map.keys()))
    race_label = st.selectbox("Race", list(race_map.keys()))
    gender_label = st.selectbox("Gender", list(gender_map.keys()))
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0)
    hours_per_week = st.slider("Hours per week", 1, 99, 40) # Adjusted min/max based on typical ranges
    native_country_label = st.selectbox("Native Country", list(native_country_map.keys()))

# Map selected labels to their numerical encodings
workclass = workclass_map[workclass_label]
marital_status = marital_status_map[marital_status_label]
occupation = occupation_map[occupation_label]
relationship = relationship_map[relationship_label]
race = race_map[race_label]
gender = gender_map[gender_label]
native_country = native_country_map[native_country_label]


# Create input DataFrame for prediction
# Ensure column order matches the training data (x in the notebook)
input_data_single = pd.DataFrame([[
    age, workclass, fnlwgt, educational_num, marital_status,
    occupation, relationship, race, gender, capital_gain,
    capital_loss, hours_per_week, native_country
]], columns=[
    'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
    'occupation', 'relationship', 'race', 'gender', 'capital-gain',
    'capital-loss', 'hours-per-week', 'native-country'
])

st.write("### 🔎 Input Data for Prediction")
st.dataframe(input_data_single)

# Predict button
if st.button("Predict Salary Class for Individual"):
    try:
        # The loaded model is a Pipeline, so it handles scaling automatically
        prediction = model.predict(input_data_single)
        prediction_proba = model.predict_proba(input_data_single)

        st.success(f"**Predicted Salary Class:** `{prediction[0]}`")
        st.info(f"Confidence: <=50K: {prediction_proba[0][0]:.2f}, >50K: {prediction_proba[0][1]:.2f}")

        st.markdown("---")
        st.subheader("📊 Feature Importance for This Prediction")
        st.markdown("Understanding which factors most influenced this salary class prediction.")

        # Feature importance for tree-based models (like GradientBoosting)
        # Check if the model is a Pipeline before accessing named_steps
        if isinstance(model, Pipeline) and 'model' in model.named_steps and hasattr(model.named_steps['model'], 'feature_importances_'):
            feature_importances = model.named_steps['model'].feature_importances_
            feature_names = input_data_single.columns
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette='viridis')
            ax.set_title('Top Feature Importances for Salary Prediction')
            ax.set_xlabel('Relative Importance')
            ax.set_ylabel('Feature')
            st.pyplot(fig)
        elif hasattr(model, 'feature_importances_'): # Fallback if it's a direct classifier but not a Pipeline
            st.warning("Model is not a Pipeline, but has feature_importances_. Displaying raw importances. "
                       "For full functionality including scaling, ensure 'best_model.pkl' is a Pipeline.")
            feature_importances = model.feature_importances_
            feature_names = input_data_single.columns
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette='viridis')
            ax.set_title('Top Feature Importances for Salary Prediction')
            ax.set_xlabel('Relative Importance')
            ax.set_ylabel('Feature')
            st.pyplot(fig)
        else:
            st.warning("Feature importance visualization is not available for this model type or structure.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Batch Prediction ---
st.markdown("---")
st.header("📂 Batch Salary Assessment")
st.markdown("Upload a CSV file to get salary class predictions for multiple employees.")

uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.dataframe(batch_data.head())

    # Preprocess batch data using the same functions
    try:
        # Make a copy to avoid modifying the original dataframe in place if it's used elsewhere
        processed_batch_data = batch_data.copy()

        # Ensure all expected columns are present before preprocessing
        expected_cols = [
            'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
            'occupation', 'relationship', 'race', 'gender', 'capital-gain',
            'capital-loss', 'hours-per-week', 'native-country'
        ]
        missing_cols = [col for col in expected_cols if col not in processed_batch_data.columns]
        if missing_cols:
            st.error(f"Missing columns in uploaded CSV: {', '.join(missing_cols)}. Please ensure your CSV has all required features.")
        else:
            # Apply preprocessing steps to the batch data
            # NOTE: For a robust production system, LabelEncoders should be saved/loaded
            # and applied here, not re-fitted. This is a simplification for the demo.
            # For this demo, we'll manually apply the mappings.
            processed_batch_data['workclass'] = processed_batch_data['workclass'].map(workclass_map)
            processed_batch_data['marital-status'] = processed_batch_data['marital-status'].map(marital_status_map)
            processed_batch_data['occupation'] = processed_batch_data['occupation'].map(occupation_map)
            processed_batch_data['relationship'] = processed_batch_data['relationship'].map(relationship_map)
            processed_batch_data['race'] = processed_batch_data['race'].map(race_map)
            processed_batch_data['gender'] = processed_batch_data['gender'].map(gender_map)
            processed_batch_data['native-country'] = processed_batch_data['native-country'].map(native_country_map)

            # Handle any NaN values that might result from mapping unknown categories
            processed_batch_data.fillna(processed_batch_data.mean(numeric_only=True), inplace=True) # Fill numerical NaNs with mean
            processed_batch_data.fillna(processed_batch_data.mode().iloc[0], inplace=True) # Fill categorical NaNs with mode

            # Drop 'education' if present (as it was dropped in training)
            if 'education' in processed_batch_data.columns:
                processed_batch_data = processed_batch_data.drop(columns=['education'])

            # Ensure all columns are numeric after encoding
            for col in processed_batch_data.columns:
                if processed_batch_data[col].dtype == 'object':
                    st.warning(f"Column '{col}' still contains non-numeric data after mapping. This might indicate unmapped categories.")
                    # Fallback for unmapped categories if any remain
                    processed_batch_data[col] = LabelEncoder().fit_transform(processed_batch_data[col])


            # Predict
            batch_preds = model.predict(processed_batch_data[expected_cols]) # Predict using only expected columns
            batch_data['Predicted_Salary_Class'] = batch_preds

            st.write("### ✅ Predictions for Batch Data:")
            st.dataframe(batch_data.head())

            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions CSV",
                csv,
                file_name='predicted_salary_classes.csv',
                mime='text/csv'
            )
    except Exception as e:
        st.error(f"An error occurred during batch prediction preprocessing: {e}")
        st.info("Please ensure your uploaded CSV matches the expected format and contains valid categories.")

st.markdown("---")
st.markdown("""
### 💡 How this system aids HR:
* **Fairness Assessment:** By analyzing predictions across different demographic groups, HR can identify potential biases in compensation.
* **Anomaly Detection:** Discrepancies between predicted and actual (or expected) salary classes can highlight cases for further investigation.
* **Data-Driven Insights:** Feature importance helps HR understand which employee attributes are most influential in salary determination.
""")
'''

# Write the content to app.py
import os
try:
    with open('app.py', 'w') as f:
        f.write(app_content)
    print("app.py created/updated successfully!")
except Exception as e:
    print(f"Error writing app.py: {e}")


# --- Step 3: Train and save your model (if you haven't already) ---
# This is a placeholder. You would typically have a separate notebook or script
# where you train your model and save it as 'best_model.pkl'.
# If you run this cell without having trained your model, the Streamlit app
# will not work because 'best_model.pkl' will be missing.

# Example of how you might train and save a model (for demonstration purposes)
# This requires your 'adult.csv' to be in the same directory.
# This code block is designed to create a Pipeline and save it correctly.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

try:
    # Load the dataset
    data = pd.read_csv("adult.csv")

    # Apply the same preprocessing steps as in the preprocess_data function
    # (These steps are crucial for consistent data handling)
    data['workclass'].replace({'?': 'Others'}, inplace=True)
    data['occupation'].replace({'?': 'Others'}, inplace=True)
    data['native-country'].replace({'?': 'Others'}, inplace=True)

    data = data[data['workclass'] != 'Without-pay']
    data = data[data['workclass'] != 'Never-worked']
    data = data[(data['age'] <= 75) & (data['age'] >= 17)]
    data = data[(data['educational-num'] <= 16) & (data['educational-num'] >= 5)]

    if 'education' in data.columns:
        data = data.drop(columns=['education'])

    # Label Encode target variable 'income'
    le_salary = LabelEncoder()
    data['income'] = le_salary.fit_transform(data['income'])

    # Label Encode other categorical features
    categorical_cols = [
        'workclass', 'marital-status', 'occupation',
        'relationship', 'race', 'gender', 'native-country'
    ]
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Define features (X) and target (y)
    X = data.drop('income', axis=1)
    y = data['income']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with StandardScaler and GradientBoostingClassifier
    # This ensures both scaling and the model are saved together.
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(random_state=42))
    ])

    # Train the pipeline
    model_pipeline.fit(X_train, y_train)

    # Save the entire pipeline to 'best_model.pkl'
    joblib.dump(model_pipeline, "best_model.pkl")
    print("Model (Pipeline) 'best_model.pkl' trained and saved successfully!")
except FileNotFoundError:
    print("Warning: 'adult.csv' not found. Cannot train and save model. Please ensure it's in the same directory.")
except Exception as e:
    print(f"An error occurred during model training: {e}")


# --- Step 4: Run the Streamlit app from your Terminal ---
# You CANNOT run the Streamlit app directly in a Jupyter cell and see the interactive UI.
# You need to open your system's Terminal or Command Prompt.

# In a NEW Terminal window (not inside Jupyter):
# 1. Navigate to the directory where your app.py and best_model.pkl files are located.
#    Example: cd /path/to/your/project/folder
# 2. Run the Streamlit command:
#    streamlit run app.py

# This will open the Streamlit app in your web browser.
# Keep the Terminal window open while you are using the app.
