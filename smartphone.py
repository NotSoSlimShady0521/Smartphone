import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import requests
import json

# Set Streamlit to use wide layout
st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('smartphone.csv')  # Replace with your file path
    return df

# Preprocess the data
def preprocess_data(df):
    # Display dataset columns for debugging
    st.write("Columns in dataset:", df.columns)
    
    # Rename or preprocess the price column if it exists
    if 'Price' in df.columns:
        df['price'] = df['Price'].str.replace('MYR ', '', regex=False).str.replace(',', '', regex=False).astype(float)

    # Define features
    features = ['price', 'battery_capacity', 'ram_capacity', 'internal_memory', 'screen_size', 'rear_camera', 'front_camera']

    # Check for missing columns
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        st.error(f"Missing columns in dataset: {missing_features}")
        return df, df, [], None  # Return empty features if columns are missing

    df[features] = df[features].apply(pd.to_numeric, errors='coerce')  # Convert to numeric
    df[features] = df[features].fillna(df[features].mean())  # Fill NaN values with column means
    
    df_original = df.copy()  # Make a copy to keep original values for display
    scaler = MinMaxScaler()  # Initialize MinMaxScaler
    df[features] = scaler.fit_transform(df[features])  # Scale the features

    return df, df_original, features, scaler

# Recommend smartphones based on similarity
def recommend_smartphones(df, user_preferences, features, scaler, top_n=10):
    user_preferences_scaled = scaler.transform([user_preferences])
    user_preferences_df = pd.DataFrame(user_preferences_scaled, columns=features)
    df_with_user = pd.concat([df, user_preferences_df], ignore_index=True)
    similarity = cosine_similarity(df_with_user[features])
    similar_indices = similarity[-1, :-1].argsort()[-top_n:][::-1]

    # Return only valid indices (remove rows where indices exceed DataFrame length)
    similar_indices = [i for i in similar_indices if i < len(df)]
    
    return similar_indices

# Function to get smartphone images using a public image API
def get_image_url(query):
    # API call to get image, using an example free image search API
    search_url = f"https://api.unsplash.com/search/photos?query={query}&client_id=YOUR_UNSPLASH_API_KEY"  # Replace with actual API key
    response = requests.get(search_url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            return data['results'][0]['urls']['small']
    return None

# Streamlit App
def main():
    st.title('Smartphone Recommender System')
    
    # Load and preprocess the data
    df = load_data()
    df_scaled, df_original, features, scaler = preprocess_data(df)

    # Sidebar input: Set preferences
    st.sidebar.header('Set Your Preferences')

    # Add "Any Brand" option to the brand selection
    brand_list = ['Any Brand'] + df_original['brand_name'].unique().tolist()
    selected_brand = st.sidebar.selectbox('Choose a brand', options=brand_list, index=0)
    
    # Filter the dataframe based on selected brand
    if selected_brand != 'Any Brand':
        df_filtered = df_scaled[df_original['brand_name'] == selected_brand]
        df_original_filtered = df_original[df_original['brand_name'] == selected_brand]
    else:
        df_filtered = df_scaled
        df_original_filtered = df_original

    # Ensure no NaN values in the price column by filling with mean
    df_original_filtered['price'] = df_original_filtered['price'].fillna(df_original_filtered['price'].mean())
    
    # Processor brand options based on the selected smartphone brand
    if selected_brand == 'Any Brand':
        processor_list = df_original['processor_brand'].unique().tolist()  # Show all processor brands if "Any Brand" selected
    else:
        processor_list = ['Any Processor Brand'] + df_original_filtered['processor_brand'].unique().tolist()

    # Processor brand selection
    selected_processor_brand = st.sidebar.selectbox('Choose a Processor Brand', options=processor_list, index=0)
    
    # Filter by processor brand unless "Any Processor Brand" is selected
    if selected_processor_brand != 'Any Processor Brand':
        df_filtered = df_filtered[df_original_filtered['processor_brand'] == selected_processor_brand]
        df_original_filtered = df_original_filtered[df_original_filtered['processor_brand'] == selected_processor_brand]

    # User input: preferences for smartphone features
    price = st.sidebar.slider('Max Price (RM)', min_value=int(df_original_filtered['price'].dropna().min()), max_value=int(df_original_filtered['price'].dropna().max()), value=1500)
    battery_capacity = st.sidebar.slider('Min Battery Capacity (mAh)', min_value=int(df_original_filtered['battery_capacity'].min()), max_value=int(df_original_filtered['battery_capacity'].max()), value=4000)
    ram_capacity = st.sidebar.slider('Min RAM (GB)', min_value=int(df_original_filtered['ram_capacity'].min()), max_value=int(df_original_filtered['ram_capacity'].max()), value=6)
    internal_memory = st.sidebar.slider('Min Internal Memory (GB)', min_value=int(df_original_filtered['internal_memory'].min()), max_value=int(df_original_filtered['internal_memory'].max()), value=128)
    screen_size = st.sidebar.slider('Min Screen Size (inches)', min_value=float(df_original_filtered['screen_size'].min()), max_value=float(df_original_filtered['screen_size'].max()), value=6.5)
    
    # Dropdowns for camera megapixels
    rear_camera = st.sidebar.selectbox('Choose Min Rear Camera MP', sorted(df_original_filtered['primary_camera_rear'].unique()))
    front_camera = st.sidebar.selectbox('Choose Min Front Camera MP', sorted(df_original_filtered['primary_camera_front'].unique()))

    # Store user preferences
    user_preferences = [price, battery_capacity, ram_capacity, internal_memory, screen_size, rear_camera, front_camera]

    # Add a submit button to confirm the search
    if st.sidebar.button("Submit"):
        # Recommend smartphones
        similar_indices = recommend_smartphones(df_filtered, user_preferences, features, scaler)

        # Display recommendations with original values and units
        recommendations = df_original_filtered.iloc[similar_indices]

        st.subheader(f'Recommended Smartphones for Brand: {selected_brand} and Processor: {selected_processor_brand}')
             st.write(recommendations[['brand_name', 'model', 
                                  'price', 'battery_capacity', 
                                  'processor_brand', 'ram_capacity', 
                                  'internal_memory', 'screen_size', 
                                  'primary_camera_rear', 'primary_camera_front']])

if __name__ == "__main__":
    main()
