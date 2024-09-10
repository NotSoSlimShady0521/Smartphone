import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import requests

# Function to load the dataset
@st.cache
def load_data():
    df = pd.read_csv('smartphone.csv')  # Ensure your CSV file has 'Brand', 'Price', 'Battery', 'Camera', 'RAM', 'Storage'
    return df

# Preprocess the data
def preprocess_data(df):
    # Convert the price to numeric (assuming the format is like 'MYR 1,000')
    df['price'] = df['Price'].str.replace('MYR ', '').str.replace(',', '').astype(float)
    
    # Select numerical features
    df_features = df[['price', 'Battery', 'Camera', 'RAM', 'Storage']]
    
    # Scale the features
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)
    
    return df, df_scaled

# Google API function to get images
def get_image_url(query, api_key, cx):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={cx}&key={api_key}&searchType=image&num=1"
    response = requests.get(url).json()
    try:
        return response['items'][0]['link']
    except (KeyError, IndexError):
        return None

# Function to recommend smartphones
def recommend_smartphones(df, df_scaled, selected_index, top_n=5):
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(df_scaled)
    
    # Get similarity scores for the selected smartphone
    similarity_scores = similarity_matrix[selected_index]
    
    # Sort smartphones based on similarity scores
    similar_smartphones_indices = similarity_scores.argsort()[::-1][1:top_n+1]  # Exclude the selected smartphone
    
    return df.iloc[similar_smartphones_indices]

# Main function
def main():
    # Load and preprocess the data
    df = load_data()
    df, df_scaled = preprocess_data(df)
    
    st.title("Smartphone Recommender System")

    # Select brand filter
    brand_options = df['Brand'].unique()
    selected_brand = st.selectbox("Select a phone brand", options=brand_options)
    
    # Input price range filter
    min_price = st.number_input("Minimum price (MYR)", min_value=0, value=1000)
    max_price = st.number_input("Maximum price (MYR)", min_value=0, value=5000)
    
    # Filter data based on user inputs
    filtered_df = df[(df['Brand'] == selected_brand) & (df['price'] >= min_price) & (df['price'] <= max_price)]
    
    if filtered_df.empty:
        st.write("No smartphones found matching the criteria.")
    else:
        # Select a smartphone from the filtered list
        selected_smartphone = st.selectbox("Select a smartphone", filtered_df['Model'])
        selected_index = df[df['Model'] == selected_smartphone].index[0]
        
        # Show selected smartphone details
        st.write(f"**Selected Smartphone: {selected_smartphone}**")
        
        # Recommend similar smartphones
        st.write("Recommended Smartphones:")
        recommended_smartphones = recommend_smartphones(df, df_scaled, selected_index)
        
        # Set Google Custom Search API key and CX (custom search engine ID)
        google_api_key = 'YOUR_GOOGLE_API_KEY'
        google_cx = 'YOUR_CUSTOM_SEARCH_ENGINE_ID'
        
        for _, row in recommended_smartphones.iterrows():
            # Display smartphone details
            st.write(f"**Model**: {row['Model']}")
            st.write(f"**Price**: MYR {row['price']}")
            st.write(f"**Battery**: {row['Battery']} mAh")
            st.write(f"**Camera**: {row['Camera']} MP")
            st.write(f"**RAM**: {row['RAM']} GB")
            st.write(f"**Storage**: {row['Storage']} GB")
            
            # Get smartphone image using Google Custom Search API
            image_query = f"{row['Model']} smartphone"
            image_url = get_image_url(image_query, google_api_key, google_cx)
            
            # Display smartphone image if found
            if image_url:
                st.image(image_url, width=200)
            else:
                st.write("Image not available.")
                
if __name__ == '__main__':
    main()
