# Implementing-Vector-Space-Model-to-Retrieve-and-Rank-documents

 ## -------- How to run the GUI files? --------------

Run the Streamlit Application:
Open a command prompt or terminal.
Navigate to the directory where your app.py file is located.

1. Run the application by typing in terminal: streamlit run app.py

2. Open a web browser and go to the URL provided by Streamlit (http://localhost:8501).


You will see input fields for entering a search query and setting an alpha threshold value.
Enter your search query and the alpha threshold in the respective fields.
Click the "Search" button to retrieve and display documents based on their relevance to the query.
Interacting with the Application:

The application allows you to adjust the alpha threshold to filter the search results based on their scores.
After performing a search, the results are displayed on the same page, listing document IDs and their corresponding scores if they exceed the alpha threshold.


## ----------- VSM Model -----------------

### Overview

This code implements a basic Vector Space Model (VSM) for text processing and retrieval in Python using natural language processing (NLP) techniques. 
The VSM is utilized to compute Term Frequency-Inverse Document Frequency (TF-IDF) scores which are crucial in ranking and retrieving documents based on 
their relevance to a given query. This implementation is particularly useful in understanding the fundamental concepts of text indexing, stemming, and 
relevance scoring in information retrieval systems.

### Features

Text Preprocessing: Includes cleaning of text, stemming using the PorterStemmer, and removal of stopwords.
TF-IDF Calculation: Computes the term frequency-inverse document frequency for terms in the corpus.
Query Processing: Processes a text query and computes its vector representation based on the indexed terms.
Relevance Scoring: Scores and ranks documents based on their cosine similarity to the query.

### Modules Required

numpy: For handling large arrays and matrices.
json: For storing and retrieving the TF-IDF vectors in JSON format.
nltk: Specifically, PorterStemmer is used for stemming words.
re: Regular expression operations for text processing.
streamlit: for running vsm_gui.py

### File Structure

Stopword-List.txt: Contains a list of stopwords.
ResearchPapers/: Directory containing text documents as .txt files, each named with a unique identifier.
doc_tdidf.json : stores td.idf of document in form of vectors in json format. 
