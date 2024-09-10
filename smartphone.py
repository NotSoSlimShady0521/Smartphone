import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('smartphone.csv')  # Replace with your file path
    return df

# Preprocess the data
def preprocess_data(df):
    # Remove 'MYR ' from the price column and convert to numeric
    df['price'] = df['Price'].str.replace('MYR ', '', regex=False).str.replace(',', '', regex=False).astype(float)

    # Select numerical columns for similarity computation
    features = ['price', 'rating', 'battery_capacity', 'ram_capacity', 'internal_memory', 'screen_size']
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')  # Convert to numeric

    # Fill missing values with the mean
    df[features] = df[features].fillna(df[features].mean())

    # Save the original values for display later
    df_original = df.copy()

    # Normalize the feature values to a range of [0,1] for comparison
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df, df_original, features, scaler

# Recommend smartphones based on similarity
def recommend_smartphones(df, user_preferences, features, scaler, top_n=10):
    # Scale the user preferences using the same MinMaxScaler as the dataframe
    user_preferences_scaled = scaler.transform([user_preferences])  # Scale user preferences to match the range
    
    # Convert user preferences into a DataFrame with the same structure as the main dataset
    user_preferences_df = pd.DataFrame(user_preferences_scaled, columns=features)
    
    # Concatenate the user's preferences as a new row in the dataframe
    df_with_user = pd.concat([df, user_preferences_df], ignore_index=True)
    
    # Compute cosine similarity between user preferences and all smartphones
    similarity = cosine_similarity(df_with_user[features])
    
    # Get the top N most similar smartphones (excluding the user preference row)
    similar_indices = similarity[-1, :-1].argsort()[-top_n:][::-1]
    
    # Return the top recommended smartphones
    return similar_indices

# Streamlit App
def main():
    st.title('Smartphone Recommender System')
    
    # Load and preprocess the data
    df = load_data()
    df_scaled, df_original, features, scaler = preprocess_data(df)

    # User input: preferences for smartphone features
    st.sidebar.header('Set Your Preferences')
    price = st.sidebar.slider('Max Price (MYR)', min_value=int(df_original['price'].min()), max_value=int(df_original['price'].max()), value=1500)
    rating = st.sidebar.slider('Min Rating', min_value=0, max_value=100, value=80)
    battery_capacity = st.sidebar.slider('Min Battery Capacity (mAh)', min_value=int(df_original['battery_capacity'].min()), max_value=int(df_original['battery_capacity'].max()), value=4000)
    ram_capacity = st.sidebar.slider('Min RAM (GB)', min_value=int(df_original['ram_capacity'].min()), max_value=int(df_original['ram_capacity'].max()), value=6)
    internal_memory = st.sidebar.slider('Min Internal Memory (GB)', min_value=int(df_original['internal_memory'].min()), max_value=int(df_original['internal_memory'].max()), value=128)
    screen_size = st.sidebar.slider('Min Screen Size (inches)', min_value=float(df_original['screen_size'].min()), max_value=float(df_original['screen_size'].max()), value=6.5)
    
    # Store user preferences
    user_preferences = [price, rating, battery_capacity, ram_capacity, internal_memory, screen_size]
    
    # Recommend smartphones
    similar_indices = recommend_smartphones(df_scaled, user_preferences, features, scaler)
    
    # Display recommendations with original values
    recommendations = df_original.iloc[similar_indices]
    
    st.subheader('Recommended Smartphones for You')
    st.write(recommendations[['brand_name', 'model', 'price', 'rating', 'battery_capacity', 'ram_capacity', 'internal_memory', 'screen_size']])

if __name__ == "__main__":
    main()
