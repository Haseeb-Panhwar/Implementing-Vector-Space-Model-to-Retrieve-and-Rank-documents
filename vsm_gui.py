import streamlit as st
from vsm import vsm  # Adjust this import statement based on your actual module and VSM class location

# Instantiate the VSM object
vector_space_model = vsm()
vector_space_model.prep()
vector_space_model.build_temp_vsm_index()
vector_space_model.build_idf()
vector_space_model.loadtfidf()

def search(query, alpha):
    vector_space_model.getquery(query)
    vector_space_model.getscore(alpha)  # Pass alpha to the getscore method
    return vector_space_model.score

# Streamlit page config
st.set_page_config(page_title="VSM Information Retrieval", layout="wide")

# Streamlit UI elements
st.title("K214889 Muhammad Qasim Alias Haseeb")
st.header("Information Retrieval Assignment # 2")
st.subheader("Vector Space Models")

# Query input
user_query = st.text_input("Enter your search query here:", "")

# Alpha input
alpha_value = st.number_input("Enter the alpha threshold for filtering scores (e.g., 0.05):", min_value=0.0, value=0.02, step=0.01)

# Search button
if st.button("Search"):
    if user_query:
        results = search(user_query, alpha_value)
        if results:
            # Display results
            st.write("Search Results:")
            for result in results:
                st.write(f"Document ID: {result[0]}, Score: {result[1]}")
        else:
            st.write("No results found for the given query.")
    else:
        st.write("Please enter a query to search.")
