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

    # Sidebar: User input to filter by brand and processor
    st.sidebar.header('Set Your Preferences')

    with st.sidebar.form(key='preferences_form'):
        # Add "Any Brand" option to the brand selection
        brand_list = ['Any Brand'] + df_original['brand_name'].unique().tolist()
        selected_brand = st.selectbox('Choose a brand', options=brand_list, index=0)
        
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
        selected_processor_brand = st.selectbox('Choose a Processor Brand', options=processor_list, index=0)
        
        # Filter by processor brand unless "Any Processor Brand" is selected
        if selected_processor_brand != 'Any Processor Brand':
            df_filtered = df_filtered[df_original_filtered['processor_brand'] == selected_processor_brand]
            df_original_filtered = df_original_filtered[df_original_filtered['processor_brand'] == selected_processor_brand]

        # Ensure there are no NaN values when calculating min/max for sliders
        df_original_filtered['price'] = df_original_filtered['price'].fillna(df_original_filtered['price'].mean())
        
        # User input: preferences for smartphone features
        price = st.slider('Max Price (MYR)', min_value=int(df_original_filtered['price'].dropna().min()), max_value=int(df_original_filtered['price'].dropna().max()), value=1500)
        battery_capacity = st.slider('Min Battery Capacity (mAh)', min_value=int(df_original_filtered['battery_capacity'].dropna().min()), max_value=int(df_original_filtered['battery_capacity'].dropna().max()), value=4000)
        ram_capacity = st.slider('Min RAM (GB)', min_value=int(df_original_filtered['ram_capacity'].dropna().min()), max_value=int(df_original_filtered['ram_capacity'].dropna().max()), value=6)
        internal_memory = st.slider('Min Internal Memory (GB)', min_value=int(df_original_filtered['internal_memory'].dropna().min()), max_value=int(df_original_filtered['internal_memory'].dropna().max()), value=128)
        screen_size = st.slider('Min Screen Size (inches)', min_value=float(df_original_filtered['screen_size'].dropna().min()), max_value=float(df_original_filtered['screen_size'].dropna().max()), value=6.5)
        
        # Dropdowns for camera megapixels
        rear_camera = st.selectbox('Choose Min Rear Camera MP', sorted(df_original_filtered['primary_camera_rear'].dropna().unique()))
        front_camera = st.selectbox('Choose Min Front Camera MP', sorted(df_original_filtered['primary_camera_front'].dropna().unique()))

        # Store user preferences
        user_preferences = [price, battery_capacity, ram_capacity, internal_memory, screen_size, rear_camera, front_camera]

        # Submit button
        submit_button = st.form_submit_button(label='Submit')

    # Only recommend smartphones when submit button is pressed
    if submit_button:
        # Recommend smartphones
        similar_indices = recommend_smartphones(df_filtered, user_preferences, features, scaler)
        
        # Display recommendations with original values
        recommendations = df_original_filtered.iloc[similar_indices]
        
        # Display result table with units
        st.subheader(f'Recommended Smartphones for Brand: {selected_brand} and Processor: {selected_processor_brand}')
        st.write(recommendations[['brand_name', 'model', 
                                  'price', 'battery_capacity', 
                                  'processor_brand', 'ram_capacity', 
                                  'internal_memory', 'screen_size', 
                                  'primary_camera_rear', 'primary_camera_front']])

if __name__ == "__main__":
    main()
