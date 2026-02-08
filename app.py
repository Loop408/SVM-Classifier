import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from datetime import datetime

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Logger
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    
# Session State
if 'clean saved' not in st.session_state:
    st.session_state['clean saved'] = False

# Foder Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
CLEAN_DIR = os.path.join(BASE_DIR, 'data', 'clean')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

log("Application started.")
log(f"Raw data directory: {RAW_DIR}")
log(f"Clean data directory: {CLEAN_DIR}")

# Page Configuration
st.set_page_config(page_title="End-to-End SVM Classifier", layout="wide")
st.title("Support Vector Machine (SVM) Classifier")

# Sidebar : Model Settings
st.sidebar.header("Model Settings")

kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])

C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])

log(f"SVM Settings-----> Kernel={kernel}, C={C}, Gamma={gamma}")

#Step 1 Data Ingestion
log("Step 1: Data Ingestion")

option = st.radio("Choose Data Source", ["Download Dataset", "Upload CSV"])


df = None
raw_path = None

if option == "Download Dataset":
    if st.button("Download Penguins Dataset"):
        log("Downloading Penguins dataset")
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
        response = requests.get(url)
         
        
        raw_path = os.path.join(RAW_DIR, "penguins.csv")
        with open(raw_path, 'wb') as f:
            f.write(response.content)
         
        df = pd.read_csv(raw_path)
        st.success("Dataset downloaded successfully")
        log(f"Dataset downloaded to {raw_path}")
        
if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        raw_path = os.path.join(RAW_DIR, uploaded_file.name)
        with open(raw_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        df = pd.read_csv(raw_path)
        st.success("File uploaded successfully")
        log(f"uploaded dataset saved to {raw_path}")

#Step 2 : EDA
if df is not None:
    st.header("Step 2: Exploratory Data Analysis (EDA)")
    log(" Step 2: EDA started")
    
    st.dataframe(df.head())
    st.write("Shape of dataset:", df.shape)
    st.write("Missing values:\n", df.isnull().sum())
    
    fig, ax = plt.subplots()
    sns.heatmap(df.select_dtypes(include=['number']).corr(),annot=True,cmap='coolwarm',ax=ax)
    st.pyplot(fig)
    
    log("EDA completed")
    
    # Step 3 : Data Cleaning
    
    if df is not None:
        st.header("Step 3: Data Cleaning")
        
        strategy = st.selectbox(
            "Missing Value Strategy",
            ["Mean", "Median", "Drop Rows"]
        )
        
        df_clean = df.copy()
        
        if strategy == "Drop Rows":
            df_clean = df_clean.dropna()
        else:
            for col in df_clean.select_dtypes(include=['number']).columns:
                if strategy == "Mean":
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif strategy == "Median":
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        st.session_state.df_clean = df_clean
        st.success("Data cleaning completed")

else:
    st.info("Please complete Step 1 to load a dataset before proceeding to EDA and Data Cleaning.")
        

#Step 4 : Save the Cleaned Data

if st.button("Save Cleaned Dataset"):
    
    if "df_clean" not in st.session_state or st.session_state.df_clean is None:
        st.error ("No cleaned data found.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename =  f"cleaned_data_{timestamp}.csv"
        clean_path = os.path.join(CLEAN_DIR, clean_filename)
        
        st.session_state.df_clean.to_csv(clean_path, index=False)
        
        st.success(f"Cleaned dataset saved to {clean_path}")
        log(f"Cleaned dataset saved to {clean_path}")
        
#step 5: Load Cleaned Data
st.header("Step 5: Load Cleaned Dataset")

clean_files = os.listdir(CLEAN_DIR)

if not clean_files:
    st.warning("No cleaned datasets found.")
    log("No cleaned datasets found in CLEAN_DIR.")

else:
    selected_file = st.selectbox("Select a cleaned dataset", clean_files)
    df_model = pd.read_csv(os.path.join(CLEAN_DIR, selected_file))
    
    st.success(f"Loaded cleaned dataset: {selected_file}")
    log(f"Loaded cleaned dataset: {selected_file}")
    
    st.dataframe(df_model.head())
    
# Step 6: Train SVM

st.header("Step 6: Train SVM Classifier")
log("Step 6: SVM Training started")

target = st.selectbox("Select Target Variable", df_model.columns)

y=df_model[target]

# validate taret for the classification
if y.dtype != 'object' and len(y.unique()) > 20:
    st.error("Invalid target selection"
             "SVM Classifier requires CATEGORICAL LABELS"
                "Please select a categorical target variable.")
    st.stop()
    
# encode target if categorical
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    log("Target variable encoded using LabelEncoder.")
    
# Select numeric features only

x=df_model.drop(columns=[target])
x=x.select_dtypes(include=[np.number])

if x.empty:
    st.error("No numeric features available for training.")
    log("No numeric features available for training. Stopping execution.")
    st.stop()

#Scale features

scaler = StandardScaler()
x = scaler.fit_transform(x)

#train test split

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Selection of the model

model = SVC(kernel=kernel, C=C, gamma=gamma)
model.fit(X_train, y_train)

#Evaluate

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.success(f"Model trained with accuracy: {accuracy:.2f}")


cm=confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)
