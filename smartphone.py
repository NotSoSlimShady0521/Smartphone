import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv('smartphone.csv')  # Ensure the correct file path
    print(df.columns)  # Add this line to print the column names
    return df


# Preprocess the data
def preprocess_data(df):
    # Remove 'MYR ' from the price column and convert to numeric
    df['price'] = df['Price'].str.replace('MYR ', '').str.replace(',', '').astype(float)
    
    # Select numerical features for similarity (customize with features relevant to smartphones)
    df_features = df[['Price', 'Battery', 'Camera', 'RAM', 'Storage']]  # Corrected names
    
    # Scale the features
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)
    
    return df, df_scaled

# Compute similarity and recommend smartphones
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
    # Load and preprocess data
    df = load_data()
    df, df_scaled = preprocess_data(df)
    
    st.title("Smartphone Recommender System")
    
    # Select a smartphone from the dataset
    selected_smartphone = st.selectbox("Select a smartphone", df['Model'])
    selected_index = df[df['Model'] == selected_smartphone].index[0]
    
    # Display selected smartphone details
    st.write(f"**Selected Smartphone: {selected_smartphone}**")
    
    # Recommend similar smartphones
    st.write("Recommended Smartphones:")
    recommended_smartphones = recommend_smartphones(df, df_scaled, selected_index)
    
    for index, row in recommended_smartphones.iterrows():
        # Display smartphone details
        st.write(f"**Model**: {row['Model']}")
        st.write(f"**Price**: MYR {row['price']}")
        st.write(f"**Battery**: {row['Battery']} mAh")
        st.write(f"**Camera**: {row['Camera']} MP")
        st.write(f"**RAM**: {row['RAM']} GB")
        st.write(f"**Storage**: {row['Storage']} GB")
        
        # Display smartphone image
        st.image(row['Image URL'], width=200)  # Assuming 'Image URL' column contains URLs of smartphone images

if __name__ == '__main__':
    main()
