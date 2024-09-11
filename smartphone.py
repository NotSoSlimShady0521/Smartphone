import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import requests
from googleapiclient.discovery import build

# Set up Google Custom Search API
API_KEY = 'YOUR_GOOGLE_API_KEY'  # Replace with your API key
CSE_ID = 'YOUR_SEARCH_ENGINE_ID'  # Replace with your Search Engine ID

def google_search(query, api_key, cse_id, num=1):
    """Function to perform Google Custom Search and retrieve image URLs."""
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(
        q=query,
        cx=cse_id,
        searchType='image',  # Only retrieve images
        num=num,
        safe='off'
    ).execute()
    
    return res['items'][0]['link'] if 'items' in res else None

# Function to get smartphone image URL
def get_smartphone_image(model_name):
    query = f"{model_name} smartphone"
    image_url = google_search(query, API_KEY, CSE_ID)
    return image_url

# Set Streamlit to use wide layout
st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('smartphone.csv')  # Replace with your file path
    return df

# Preprocess the data
def preprocess_data(df):
    df['price'] = df['Price'].str.replace('MYR ', '', regex=False).str.replace(',', '', regex=False).astype(float)
    
    # Removed 'rating' from the features list
    features = ['price', 'battery_capacity', 'ram_capacity', 'internal_memory', 'screen_size']
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')  # Convert to numeric
    
    df[features] = df[features].fillna(df[features].mean())
    df_original = df.copy()
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
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

# Streamlit App
def main():
    st.title('Smartphone Recommender System')

    df = load_data()
    df_scaled, df_original, features, scaler = preprocess_data(df)

    st.sidebar.header('Set Your Preferences')
    
    brand_list = ['Every Brand'] + df_original['brand_name'].unique().tolist()
    selected_brand = st.sidebar.selectbox('Choose a brand', options=brand_list, index=0)

    processor_list = ['Every Processor'] + df_original['processor_brand'].unique().tolist()
    selected_processor = st.sidebar.selectbox('Choose a processor brand', options=processor_list, index=0)

    # Filter the original DataFrame first based on the selected brand and processor
    df_filtered = df_original.copy()
    
    if selected_brand != 'Every Brand':
        df_filtered = df_filtered[df_filtered['brand_name'] == selected_brand]
    
    if selected_processor != 'Every Processor':
        df_filtered = df_filtered[df_filtered['processor_brand'] == selected_processor]

    # Sidebar sliders for user preferences
    price = st.sidebar.slider('Max Price (MYR)', min_value=int(df_filtered['price'].min()), max_value=int(df_filtered['price'].max()), value=1500)
    battery_capacity = st.sidebar.slider('Min Battery Capacity (mAh)', min_value=int(df_filtered['battery_capacity'].min()), max_value=int(df_filtered['battery_capacity'].max()), value=4000)
    ram_capacity = st.sidebar.slider('Min RAM (GB)', min_value=int(df_filtered['ram_capacity'].min()), max_value=int(df_filtered['ram_capacity'].max()), value=6)
    internal_memory = st.sidebar.slider('Min Internal Memory (GB)', min_value=int(df_filtered['internal_memory'].min()), max_value=int(df_filtered['internal_memory'].max()), value=128)
    screen_size = st.sidebar.slider('Min Screen Size (inches)', min_value=float(df_filtered['screen_size'].min()), max_value=float(df_filtered['screen_size'].max()), value=6.5)
    
    # Apply filters to df_filtered based on the user's input
    df_filtered = df_filtered[
        (df_filtered['price'] <= price) &
        (df_filtered['battery_capacity'] >= battery_capacity) &
        (df_filtered['ram_capacity'] >= ram_capacity) &
        (df_filtered['internal_memory'] >= internal_memory) &
        (df_filtered['screen_size'] >= screen_size)
    ]

    # Scale the filtered DataFrame for recommendation
    df_filtered_scaled = df_filtered.copy()
    df_filtered_scaled[features] = scaler.transform(df_filtered[features])

    # Removed 'rating' from user preferences
    user_preferences = [price, battery_capacity, ram_capacity, internal_memory, screen_size]
    similar_indices = recommend_smartphones(df_filtered_scaled, user_preferences, features, scaler)

    # Display the filtered and recommended smartphones
    recommendations = df_filtered.iloc[similar_indices]
    
    st.subheader(f'Recommended Smartphones for Brand: {selected_brand} and Processor: {selected_processor}')
    
    # Display smartphone data along with images
    for i, row in recommendations.iterrows():
        # Use Google API to get image if URL is not available in the dataset
        image_url = row['image_url'] if 'image_url' in row and pd.notnull(row['image_url']) else get_smartphone_image(row['model'])
        st.image(image_url, width=150)
        st.write(f"**{row['brand_name']} {row['model']}**")
        st.write(f"Price: MYR {row['price']}")
        st.write(f"Processor: {row['processor_brand']}")
        st.write(f"Battery: {row['battery_capacity']} mAh")
        st.write(f"RAM: {row['ram_capacity']} GB")
        st.write(f"Internal Memory: {row['internal_memory']} GB")
        st.write(f"Screen Size: {row['screen_size']} inches")
        st.markdown("---")  # Separator

if __name__ == "__main__":
    main()
