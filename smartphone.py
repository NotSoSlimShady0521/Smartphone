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
    df['price'] = df['Price'].str.replace('MYR ', 'RM', regex=False).str.replace(',', '', regex=False).astype(float)

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
    price = st.sidebar.slider('Max Price (RM)', min_value=int(df_original_filtered['price'].min()), max_value=int(df_original_filtered['price'].max()), value=1500)
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
                                  'price', 'battery_capacity', 'processor_brand', 
                                  'ram_capacity', 'internal_memory', 'screen_size', 
                                  'primary_camera_rear', 'primary_camera_front']].assign(
                                  price=lambda x: x['price'].apply(lambda p: f'RM {p:.2f}'),
                                  battery_capacity=lambda x: x['battery_capacity'].apply(lambda b: f'{int(b)} mAh'),
                                  ram_capacity=lambda x: x['ram_capacity'].apply(lambda r: f'{int(r)} GB'),
                                  internal_memory=lambda x: x['internal_memory'].apply(lambda m: f'{int(m)} GB'),
                                  screen_size=lambda x: x['screen_size'].apply(lambda s: f'{s:.1f} inches'),
                                  primary_camera_rear=lambda x: x['primary_camera_rear'].apply(lambda r: f'{int(r)} MP'),
                                  primary_camera_front=lambda x: x['primary_camera_front'].apply(lambda f: f'{int(f)} MP')
                                  ))

if __name__ == "__main__":
    main()
