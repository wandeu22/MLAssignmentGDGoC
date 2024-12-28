import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose

# Set page config
st.set_page_config(page_title="Used Car Price Analysis", layout="wide")

# Title of the dashboard
st.title("Used Car Price Analysis Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('used_car_dataset.csv')
    return df

df = load_data()

# Data Overview Section
st.header("1. Data Overview")

# Display raw dataset
st.subheader("Raw Dataset")
st.dataframe(df.head())

# Data Gathering Summary
st.subheader("Data Gathering Summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Records", len(df))
with col2:
    st.metric("Total Columns", len(df.columns))

# Dataset Info
st.subheader("Dataset Information")
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

# Data Preprocessing
current_year = 2024
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df = df[df['Year'].dt.year >= current_year - 3]
df['AskPrice'] = pd.to_numeric(df['AskPrice'].str.replace('₹', '').str.replace(',', ''), errors='coerce')
df.drop_duplicates(subset=['kmDriven'], keep='first', inplace=True)
data_filtered = df[['FuelType', 'AskPrice']].dropna()

# Descriptive Statistics Section
st.header("2. Descriptive Statistics")

# Overall Statistics
st.subheader("Overall Statistics")
st.dataframe(df.describe())

# Fuel Type Statistics
st.subheader("Statistics by Fuel Type")
st.dataframe(df.groupby('FuelType')['AskPrice'].describe())

# Data Distribution
st.subheader("Fuel Type Distribution")
st.dataframe(data_filtered['FuelType'].value_counts())

# Visualizations Section
st.header("3. Data Visualizations")

# Correlation Heatmap
st.subheader("Correlation Heatmap")
numerical_features = df.select_dtypes(include=['number'])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numerical_features.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
plt.close()

# Price Trends
st.subheader("Price Trends")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x='Year', y='AskPrice', hue='FuelType', data=df, ci=None, ax=ax)
    plt.title('Average Price Trend by Fuel Type')
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='FuelType', data=data_filtered, palette='viridis', ax=ax)
    plt.title('Fuel Type Distribution')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()

# Price Distribution
st.subheader("Price Distribution Analysis")
col1, col2 = st.columns(2)

with col1:
    avg_price = data_filtered.groupby('FuelType')['AskPrice'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_price.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    plt.title('Average Price by Fuel Type')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=data_filtered, x='AskPrice', hue='FuelType', fill=True, alpha=0.5, ax=ax)
    plt.title('Price Distribution by Fuel Type')
    st.pyplot(fig)
    plt.close()
