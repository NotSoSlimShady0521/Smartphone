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

# Recommender system 1: Select a phone and find similar phones
def recommender_system_1(df_original, df_scaled, features, scaler):
    st.subheader('Recommender System 1: Select a Phone and Find Similar Ones')

    # Select a phone from the dataset
    phone_selection = st.selectbox('Select a smartphone', df_original['model'].unique())

    # Get the selected phone's data
    selected_phone = df_original[df_original['model'] == phone_selection].iloc[0]
    
    # Extract the features of the selected phone
    user_preferences = [selected_phone['price'], 
                        selected_phone['battery_capacity'], 
                        selected_phone['ram_capacity'], 
                        selected_phone['internal_memory'], 
                        selected_phone['screen_size'], 
                        selected_phone['primary_camera_rear'], 
                        selected_phone['primary_camera_front']]
    
    # Find and display similar smartphones
    similar_indices = recommend_smartphones(df_scaled, user_preferences, features, scaler)
    recommendations = df_original.iloc[similar_indices]
    
    # Display result table with units
    st.write(recommendations[['brand_name', 'model', 
                              'price', 'battery_capacity', 
                              'processor_brand', 'ram_capacity', 
                              'internal_memory', 'screen_size', 
                              'primary_camera_rear', 'primary_camera_front']])

# Recommender system 2: Customize your preferences
def recommender_system_2(df_original, df_scaled, features, scaler):
    st.subheader('Recommender System 2: Customize Your Preferences')

    # Sidebar: User input to filter by brand and processor
    with st.sidebar.form(key='preferences_form'):
        # Get all unique brands and processors
        all_brands = df_original['brand_name'].unique().tolist()
        all_processors = df_original['processor_brand'].unique().tolist()

        # Initialize brand dropdown
        brand_list_with_any = ['Any Brand'] + all_brands
        selected_brand = st.selectbox('Choose a brand', options=brand_list_with_any)

        # Limit processor options based on the selected brand
        if selected_brand != 'Any Brand':
            filtered_processor_list = df_original[df_original['brand_name'] == selected_brand]['processor_brand'].unique().tolist()
            processor_list_with_any = ['Any Processor Brand'] + filtered_processor_list
        else:
            processor_list_with_any = ['Any Processor Brand'] + all_processors

        selected_processor_brand = st.selectbox('Choose a Processor Brand', options=processor_list_with_any)

        # Limit brand options based on the selected processor brand
        if selected_processor_brand != 'Any Processor Brand':
            filtered_brand_list = df_original[df_original['processor_brand'] == selected_processor_brand]['brand_name'].unique().tolist()
            brand_list_with_any = ['Any Brand'] + filtered_brand_list

        # Ensure the dropdown for brand is updated after processor selection
        selected_brand = st.selectbox('Choose a brand (updated after processor selection)', options=brand_list_with_any, index=brand_list_with_any.index(selected_brand))

        # Now filter the dataframe based on the selected brand and processor brand
        if selected_brand != 'Any Brand':
            df_filtered = df_scaled[df_original['brand_name'] == selected_brand]
            df_original_filtered = df_original[df_original['brand_name'] == selected_brand]
        else:
            df_filtered = df_scaled
            df_original_filtered = df_original

        if selected_processor_brand != 'Any Processor Brand':
            df_filtered = df_filtered[df_original_filtered['processor_brand'] == selected_processor_brand]
            df_original_filtered = df_original_filtered[df_original_filtered['processor_brand'] == selected_processor_brand]

        # Ensure the filtered dataframe is not empty
        if df_original_filtered.empty:
            st.error("No data available for the selected brand or processor. Please adjust your filters.")
            return  # Exit if no data is available

        # User input: preferences for smartphone features
        price = st.slider('Max Price (MYR)', min_value=int(df_original_filtered['price'].min()), max_value=int(df_original_filtered['price'].max()), value=1500)
        battery_capacity = st.slider('Min Battery Capacity (mAh)', min_value=int(df_original_filtered['battery_capacity'].min()), max_value=int(df_original_filtered['battery_capacity'].max()), value=4000)
        ram_capacity = st.slider('Min RAM (GB)', min_value=int(df_original_filtered['ram_capacity'].min()), max_value=int(df_original_filtered['ram_capacity'].max()), value=6)
        internal_memory = st.slider('Min Internal Memory (GB)', min_value=int(df_original_filtered['internal_memory'].min()), max_value=int(df_original_filtered['internal_memory'].max()), value=128)
        screen_size = st.slider('Min Screen Size (inches)', min_value=float(df_original_filtered['screen_size'].min()), max_value=float(df_original_filtered['screen_size'].max()), value=6.5)

        # Dropdowns for camera megapixels
        rear_camera = st.selectbox('Choose Min Rear Camera MP', sorted(df_original_filtered['primary_camera_rear'].unique()))
        front_camera = st.selectbox('Choose Min Front Camera MP', sorted(df_original_filtered['primary_camera_front'].unique()))

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
        st.write(recommendations[['brand_name', 'model', 
                                  'price', 'battery_capacity', 
                                  'processor_brand', 'ram_capacity', 
                                  'internal_memory', 'screen_size', 
                                  'primary_camera_rear', 'primary_camera_front']])

# Main function to run the Streamlit app
def main():
    st.title('Smartphone Recommender System')

    # Load and preprocess the data
    df = load_data()
    df_scaled, df_original, features, scaler = preprocess_data(df)

    # Select the recommender system mode
    system_choice = st.sidebar.selectbox('Choose a recommender system', 
                                         ['Recommender System 1: Select a phone and find similar', 
                                          'Recommender System 2: Customize preferences'])

    # Display the corresponding recommender system
    if system_choice == 'Recommender System 1: Select a phone and find similar':
        recommender_system_1(df_original, df_scaled, features, scaler)
    else:
        recommender_system_2(df_original, df_scaled, features, scaler)

if __name__ == "__main__":
    main()
