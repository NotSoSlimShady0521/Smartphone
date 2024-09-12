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
    
    # Return the top recommended smartphones' indices
    return similar_indices

# Recommender System 1: Recommend similar phones based on user selection
def recommender_system_1(df_original, df_scaled, features, scaler):
    st.subheader('Recommender System 1: Select a Phone')
    
    # Let the user select a phone
    selected_phone = st.selectbox('Choose a Phone', df_original['model'])
    
    # Find the phone in the original dataframe
    selected_phone_row = df_original[df_original['model'] == selected_phone].iloc[0]
    
    # Extract features of the selected phone
    selected_phone_features = selected_phone_row[features].values.reshape(1, -1)
    
    # Calculate similarity with all phones in the dataset
    similarity = cosine_similarity(selected_phone_features, df_scaled[features])
    
    # Get top 5 most similar phones
    top_indices = similarity.argsort()[0][-6:-1][::-1]  # Exclude the selected phone
    
    st.write("Other similar phones you might like:")
    st.write(df_original.iloc[top_indices][['brand_name', 'model', 'price', 'battery_capacity', 
                                            'ram_capacity', 'internal_memory', 'screen_size', 
                                            'primary_camera_rear', 'primary_camera_front']])

# Recommender System 2: Customized preference-based recommendation
def recommender_system_2(df_original, df_scaled, features, scaler):
    st.subheader('Recommender System 2: Customize Your Preferences')

    # Sidebar: User input to filter by brand and processor
    st.sidebar.header('Set Your Preferences')

    # Get unique brand and processor brand lists
    brand_list = df_original['brand_name'].unique().tolist()
    processor_list = df_original['processor_brand'].unique().tolist()

    # Add "Any Brand" and "Any Processor Brand" options
    brand_list_with_any = ['Any Brand'] + brand_list
    processor_list_with_any = ['Any Processor Brand'] + processor_list

    # Select brand and processor brand
    selected_brand = st.sidebar.selectbox('Choose a Smartphone Brand', options=brand_list_with_any)

    # Limit processor options based on the selected brand
    if selected_brand != 'Any Brand':
        filtered_processor_list = df_original[df_original['brand_name'] == selected_brand]['processor_brand'].unique().tolist()
        processor_list_with_any = ['Any Processor Brand'] + filtered_processor_list
    else:
        filtered_processor_list = processor_list_with_any

    selected_processor_brand = st.sidebar.selectbox('Choose a Processor Brand', options=filtered_processor_list)

    # User input: preferences for smartphone features
    with st.sidebar.form(key='preferences_form'):
        price = st.slider('Max Price (MYR)', min_value=int(df_original['price'].min()), max_value=int(df_original['price'].max()), value=4000)
        battery_capacity = st.slider('Min Battery Capacity (mAh)', min_value=int(df_original['battery_capacity'].min()), max_value=int(df_original['battery_capacity'].max()), value=4000)
        ram_capacity = st.slider('Min RAM (GB)', min_value=int(df_original['ram_capacity'].min()), max_value=int(df_original['ram_capacity'].max()), value=6)
        internal_memory = st.slider('Min Internal Memory (GB)', min_value=int(df_original['internal_memory'].min()), max_value=int(df_original['internal_memory'].max()), value=128)
        screen_size = st.slider('Min Screen Size (inches)', min_value=float(df_original['screen_size'].min()), max_value=float(df_original['screen_size'].max()), value=6.5)
        
        # Dropdowns for camera megapixels
        rear_camera = st.selectbox('Choose Min Rear Camera MP', sorted(df_original['primary_camera_rear'].unique()))
        front_camera = st.selectbox('Choose Min Front Camera MP', sorted(df_original['primary_camera_front'].unique()))

        # Submit button
        submit_button = st.form_submit_button(label='Submit')

    # Store user preferences
    user_preferences = [price, battery_capacity, ram_capacity, internal_memory, screen_size, rear_camera, front_camera]

    # Only recommend smartphones when submit button is pressed
    if submit_button:
        # Start filtering the dataframe based on selected brand, processor brand, and user preferences
        df_filtered = df_original.copy()

        if selected_brand != 'Any Brand':
            df_filtered = df_filtered[df_filtered['brand_name'] == selected_brand]

        if selected_processor_brand != 'Any Processor Brand':
            df_filtered = df_filtered[df_filtered['processor_brand'] == selected_processor_brand]

        # Filter based on the max price and other user preferences
        df_filtered = df_filtered[
            (df_filtered['price'] <= price) &  # Filter by price
            (df_filtered['battery_capacity'] >= battery_capacity) &  # Filter by battery
            (df_filtered['ram_capacity'] >= ram_capacity) &  # Filter by RAM
            (df_filtered['internal_memory'] >= internal_memory) &  # Filter by internal memory
            (df_filtered['screen_size'] >= screen_size) &  # Filter by screen size
            (df_filtered['primary_camera_rear'] >= rear_camera) &  # Filter by rear camera
            (df_filtered['primary_camera_front'] >= front_camera)  # Filter by front camera
        ]

        # If df_filtered is empty after filtering, display a message
        if df_filtered.empty:
            st.subheader(f'No smartphones found for the selected filters.')
            return

        # Now that the filtering is done, normalize the filtered data for similarity computation
        df_filtered_scaled = df_scaled.loc[df_filtered.index]  # Get scaled version of filtered data

        # Prepare the user's preferences for similarity computation
        user_preferences_scaled = scaler.transform([user_preferences])  # Scale user preferences

        # Run the recommendation based on cosine similarity
        similarity = cosine_similarity(user_preferences_scaled, df_filtered_scaled[features])

        # Get the top N recommendations
        top_indices = similarity.argsort()[0][-10:][::-1]

        # Extract the recommended smartphones from the original dataframe using the indices
        recommendations = df_filtered.iloc[top_indices]

        # Display recommendations with original values
        st.subheader(f'Recommended Smartphones for Your Preferences:')
        st.write(recommendations[['brand_name', 'model', 'price', 'battery_capacity', 
                                  'processor_brand', 'ram_capacity', 'internal_memory', 
                                  'screen_size', 'primary_camera_rear', 'primary_camera_front']])

# Main function to choose between the recommender systems
def main():
    st.title('Smartphone Recommender System')

    # Load and preprocess the data
    df = load_data()
    df_scaled, df_original, features, scaler = preprocess_data(df)

    # Menu for user to select the recommender system
    st.sidebar.title('Choose Recommender System')
    system_choice = st.sidebar.selectbox('Select Recommender System', ['Recommender System 1', 'Recommender System 2'])

    if system_choice == 'Recommender System 1':
        recommender_system_1(df_original, df_scaled, features, scaler)
    elif system_choice == 'Recommender System 2':
        recommender_system_2(df_original, df_scaled, features, scaler)


if __name__ == "__main__":
    main()
