import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import requests

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

    # Add 'rear_camera' and 'front_camera' to the features list
    features = ['price', 'battery_capacity', 'ram_capacity', 'internal_memory', 'screen_size', 'rear_camera', 'front_camera']
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

    similar_indices = [i for i in similar_indices if i < len(df)]
    return similar_indices

# Function to retrieve phone images (placeholder, to be filled with API setup)
def get_phone_image(brand, model):
    search_query = f"{brand} {model}"
    api_key = "YOUR_GOOGLE_CUSTOM_SEARCH_API_KEY"  # Replace with your API key
    search_engine_id = "YOUR_SEARCH_ENGINE_ID"     # Replace with your search engine ID
    url = f"https://www.googleapis.com/customsearch/v1?q={search_query}&cx={search_engine_id}&key={api_key}&searchType=image"
    
    try:
        response = requests.get(url)
        data = response.json()
        return data['items'][0]['link']  # Returning the first image link
    except Exception as e:
        st.error("Error retrieving image: " + str(e))
        return None

# Streamlit App
def main():
    st.title('Smartphone Recommender System')

    df = load_data()
    df_scaled, df_original, features, scaler = preprocess_data(df)

    st.sidebar.header('Set Your Preferences')

    # Dropdown filters for brand and processor
    brand_list = ['Every Brand'] + df_original['brand_name'].unique().tolist()
    selected_brand = st.sidebar.selectbox('Choose a brand', options=brand_list, index=0)

    processor_list = ['Every Processor'] + df_original['processor_brand'].unique().tolist()
    selected_processor = st.sidebar.selectbox('Choose a processor brand', options=processor_list, index=0)

    df_filtered = df_scaled.copy()
    df_original_filtered = df_original.copy()

    if selected_brand != 'Every Brand':
        df_filtered = df_filtered[df_original['brand_name'] == selected_brand]
        df_original_filtered = df_original[df_original['brand_name'] == selected_brand]

    if selected_processor != 'Every Processor':
        df_filtered = df_filtered[df_original['processor_brand'] == selected_processor]
        df_original_filtered = df_original[df_original['processor_brand'] == selected_processor]

    # User preferences through sliders
    price = st.sidebar.slider('Max Price (MYR)', min_value=int(df_original_filtered['price'].min()), max_value=int(df_original_filtered['price'].max()), value=1500)
    battery_capacity = st.sidebar.slider('Min Battery Capacity (mAh)', min_value=int(df_original_filtered['battery_capacity'].min()), max_value=int(df_original_filtered['battery_capacity'].max()), value=4000)
    ram_capacity = st.sidebar.slider('Min RAM (GB)', min_value=int(df_original_filtered['ram_capacity'].min()), max_value=int(df_original_filtered['ram_capacity'].max()), value=6)
    internal_memory = st.sidebar.slider('Min Internal Memory (GB)', min_value=int(df_original_filtered['internal_memory'].min()), max_value=int(df_original_filtered['internal_memory'].max()), value=128)
    screen_size = st.sidebar.slider('Min Screen Size (inches)', min_value=float(df_original_filtered['screen_size'].min()), max_value=float(df_original_filtered['screen_size'].max()), value=6.5)
    rear_camera = st.sidebar.slider('Min Rear Camera (MP)', min_value=int(df_original_filtered['rear_camera'].min()), max_value=int(df_original_filtered['rear_camera'].max()), value=12)
    front_camera = st.sidebar.slider('Min Front Camera (MP)', min_value=int(df_original_filtered['front_camera'].min()), max_value=int(df_original_filtered['front_camera'].max()), value=8)

    user_preferences = [price, battery_capacity, ram_capacity, internal_memory, screen_size, rear_camera, front_camera]
    similar_indices = recommend_smartphones(df_filtered, user_preferences, features, scaler)

    recommendations = df_original_filtered.iloc[similar_indices]

    st.subheader(f'Recommended Smartphones for Brand: {selected_brand} and Processor: {selected_processor}')
    
    for index, row in recommendations.iterrows():
        st.write(f"**{row['brand_name']} {row['model']}**")
        st.write(f"Price: MYR {row['price']}")
        st.write(f"Battery: {row['battery_capacity']} mAh, RAM: {row['ram_capacity']} GB, Internal Memory: {row['internal_memory']} GB, Screen Size: {row['screen_size']} inches")
        st.write(f"Rear Camera: {row['rear_camera']} MP, Front Camera: {row['front_camera']} MP")

        # Display phone image using API
        phone_image_url = get_phone_image(row['brand_name'], row['model'])
        if phone_image_url:
            st.image(phone_image_url, width=200)
        st.write('---')

if __name__ == "__main__":
    main()
