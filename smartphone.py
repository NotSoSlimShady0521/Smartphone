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
    features = ['price', 'battery_capacity', 'ram_capacity', 'internal_memory', 'screen_size', 'primary_camera_rear', 'primary_camera_front']
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
def recommend_smartphones(df, user_preferences, features, top_n=10):
    # Compute cosine similarity between user preferences and all smartphones
    similarity = cosine_similarity([user_preferences], df[features])
    
    # Get the top N most similar smartphones
    similar_indices = similarity[0].argsort()[-top_n:][::-1]
    
    return similar_indices

# Streamlit App
def main():
    st.title('Smartphone Recommender System')
    
    # Load and preprocess the data
    df = load_data()
    df_scaled, df_original, features, scaler = preprocess_data(df)

    # Sidebar: User input to filter by brand and processor
    st.sidebar.header('Set Your Preferences')

    # Allow the user to select a phone they like
    phone_options = df_original['model'].tolist()
    selected_phone = st.sidebar.selectbox('Select a phone you like', options=phone_options)

    # Get the features of the selected phone
    selected_phone_data = df_original[df_original['model'] == selected_phone].iloc[0]
    st.write(f"You selected: **{selected_phone}** with the following specs:")
    st.write(selected_phone_data[['brand_name', 'model', 'price', 'battery_capacity', 'ram_capacity', 
                                  'internal_memory', 'screen_size', 'primary_camera_rear', 'primary_camera_front']])
    
    # Get the normalized features of the selected phone for similarity computation
    selected_phone_features = df_scaled[df_original['model'] == selected_phone][features].iloc[0].tolist()

    # Recommend other smartphones similar to the selected phone
    similar_indices = recommend_smartphones(df_scaled, selected_phone_features, features)
    
    # Display recommendations, excluding the selected phone
    recommendations = df_original.iloc[similar_indices]
    recommendations = recommendations[recommendations['model'] != selected_phone]
    
    st.subheader('Phones with Similar Specifications:')
    st.write(recommendations[['brand_name', 'model', 
                              'price', 'battery_capacity', 'processor_brand', 
                              'ram_capacity', 'internal_memory', 'screen_size', 
                              'primary_camera_rear', 'primary_camera_front']])

if __name__ == "__main__":
    main()
