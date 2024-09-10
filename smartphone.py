import streamlit as st
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Constants for Google Custom Search API
API_KEY = 'YOUR_GOOGLE_API_KEY'  # Replace with your actual API key
SEARCH_ENGINE_ID = 'YOUR_SEARCH_ENGINE_ID'  # Replace with your search engine ID

# Function to get smartphone image from Google Custom Search API
def get_smartphone_image(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={SEARCH_ENGINE_ID}&key={API_KEY}&searchType=image&num=1"
    response = requests.get(url)
    try:
        results = response.json().get('items', [])
        if results:
            return results[0]['link']  # Return the first image link
    except Exception as e:
        print(f"Error fetching image: {e}")
    return None  # Return None if no image found

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

    # User input: Filter by brand
    st.sidebar.header('Set Your Preferences')
    
    # Dropdown for brand selection
    brand_list = df_original['brand_name'].unique().tolist()
    selected_brand = st.sidebar.selectbox('Choose a brand', options=brand_list, index=0)
    
    # Filter the dataframe based on selected brand
    df_filtered = df_scaled[df_original['brand_name'] == selected_brand]
    df_original_filtered = df_original[df_original['brand_name'] == selected_brand]

    # User input: preferences for smartphone features
    price = st.sidebar.slider('Max Price (MYR)', min_value=int(df_original_filtered['price'].min()), max_value=int(df_original_filtered['price'].max()), value=1500)
    rating = st.sidebar.slider('Min Rating', min_value=0, max_value=100, value=80)
    battery_capacity = st.sidebar.slider('Min Battery Capacity (mAh)', min_value=int(df_original_filtered['battery_capacity'].min()), max_value=int(df_original_filtered['battery_capacity'].max()), value=4000)
    ram_capacity = st.sidebar.slider('Min RAM (GB)', min_value=int(df_original_filtered['ram_capacity'].min()), max_value=int(df_original_filtered['ram_capacity'].max()), value=6)
    internal_memory = st.sidebar.slider('Min Internal Memory (GB)', min_value=int(df_original_filtered['internal_memory'].min()), max_value=int(df_original_filtered['internal_memory'].max()), value=128)
    screen_size = st.sidebar.slider('Min Screen Size (inches)', min_value=float(df_original_filtered['screen_size'].min()), max_value=float(df_original_filtered['screen_size'].max()), value=6.5)
    
    # Store user preferences
    user_preferences = [price, rating, battery_capacity, ram_capacity, internal_memory, screen_size]
    
    # Recommend smartphones
    similar_indices = recommend_smartphones(df_filtered, user_preferences, features, scaler)
    
    # Display recommendations with original values
    recommendations = df_original_filtered.iloc[similar_indices]
    
    st.subheader(f'Recommended Smartphones for Brand: {selected_brand}')
    
    for _, row in recommendations.iterrows():
        image_url = get_smartphone_image(f"{row['brand_name']} {row['model']}")
        st.write(f"**{row['brand_name']} {row['model']}**")
        st.write(f"Price: MYR {row['price']}")
        st.write(f"Rating: {row['rating']}")
        st.write(f"Battery Capacity: {row['battery_capacity']} mAh")
        st.write(f"RAM: {row['ram_capacity']} GB")
        st.write(f"Internal Memory: {row['internal_memory']} GB")
        st.write(f"Screen Size: {row['screen_size']} inches")
        
        if image_url:
            st.image(image_url, caption=f"{row['brand_name']} {row['model']}", use_column_width=True)
        else:
            st.write("Image not available")

if __name__ == "__main__":
    main()
