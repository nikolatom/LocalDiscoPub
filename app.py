import streamlit as st
import torch
import numpy as np
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import faiss
import os
from heapq import nlargest
import tempfile

# Initialize the model and tokenizer
model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

def load_embeddings_and_pmids(embeds_file, pmids_file):
    embeds = np.load(embeds_file)
    with open(pmids_file, 'r') as f:
        pmids = json.load(f)
    return embeds, pmids

def load_abstracts_for_pmids(pmids_list, chunks_dir):
    abstracts = {}
    for file in os.listdir(chunks_dir):
        if file.startswith("pubmed_chunk") and file.endswith(".json"):
            try:
                with open(os.path.join(chunks_dir, file), 'r') as f:
                    data = json.load(f)
                    for pmid in pmids_list:
                        pmid_str = str(pmid)
                        if pmid_str in data:
                            abstracts[pmid_str] = data[pmid_str].get('a', 'No abstract available')
            except json.JSONDecodeError as e:
                st.error(f"Problem with decoding JSON file {file}: {e}. Some abstracts/papers from this file might not be shown and need to be found manually based on PMID.")
                continue
            except Exception as e:
                st.error(f"An unexpected error occurred while processing file {file}: {e}")
                continue
    return abstracts

def load_embeddings_and_pmids(embeds_file, pmids_file):
    print(f"Loading embeddings from {embeds_file} and PMIDs from {pmids_file}")
    embeds = np.load(embeds_file)
    with open(pmids_file, 'r') as f:
        pmids = json.load(f)
    print(f"Loaded {len(embeds)} embeddings and {len(pmids)} PMIDs")
    return embeds, pmids

def load_abstracts_for_pmids(pmids_list, chunks_dir):
    print(f"Loading abstracts for PMIDs: {pmids_list} from {chunks_dir}")
    abstracts = {}
    for file in os.listdir(chunks_dir):
        print(f"Checking file: {file}")
        if file.startswith("pubmed_chunk") and file.endswith(".json"):
            try:
                with open(os.path.join(chunks_dir, file), 'r') as f:
                    data = json.load(f)
                    for pmid in pmids_list:
                        pmid_str = str(pmid)
                        if pmid_str in data:
                            abstracts[pmid_str] = data[pmid_str].get('a', 'No abstract available')
            except json.JSONDecodeError as e:
                print(f"JSON decoding error for file {file}: {e}")
                st.error(f"Problem with decoding JSON file {file}: {e}. Some abstracts/papers from this file might not be shown and need to be found manually based on PMID.")
                continue
            except Exception as e:
                print(f"Unexpected error while processing file {file}: {e}")
                st.error(f"An unexpected error occurred while processing file {file}: {e}")
                continue
    print(f"Loaded abstracts for {len(abstracts)} PMIDs")
    return abstracts

def search_pubmed(query, chunks_dir):
    print(f"Searching PubMed for query: {query}")
    with torch.no_grad():
        status_text.text('Encoding query...')
        print("Encoding query")
        encoded = tokenizer([query], truncation=True, padding=True, return_tensors='pt', max_length=64)
        query_embeds = model(**encoded).last_hidden_state[:, 0, :].numpy()

    results = []
    chunk_files = [file for file in sorted(os.listdir(chunks_dir)) if file.startswith("embeds_chunk") and file.endswith(".npy")]
    total_chunks = len(chunk_files)
    print(f"Total chunks to process: {total_chunks}")
    
    for i, file in enumerate(chunk_files, start=1):
        status_text.text(f'Loading embeddings and PMIDs: Processing chunk {i}/{total_chunks}...')
        print(f"Processing chunk {i}/{total_chunks}")
        chunk_number = file.split("_")[-1].split(".")[0]
        embeds_file = f"{chunks_dir}/{file}"
        pmids_file = f"{chunks_dir}/pmids_chunk_{chunk_number}.json"
        
        embeds, pmids = load_embeddings_and_pmids(embeds_file, pmids_file)
        index = faiss.IndexFlatIP(768)
        index.add(embeds)
        scores, inds = index.search(query_embeds, k=10)
        
        for score, ind in zip(scores[0], inds[0]):
            results.append((score, pmids[ind]))

        progress_bar.progress(i / total_chunks)

    print(f"Found {len(results)} results")
    top_results = nlargest(10, results, key=lambda x: x[0])
    top_pmids = [str(pmid) for _, pmid in top_results]
    
    print(f"Top PMIDs: {top_pmids}")
    # Proceed with loading abstracts and other steps...

    
    # Prepare for loading abstracts
    progress_bar.empty()  # Clear the progress bar from the previous phase
    status_text.text('Loading abstracts for top results...')
    
    abstracts = {}
    total_pmids = len(top_pmids)
    for i, pmid in enumerate(top_pmids, start=1):
        abstracts.update(load_abstracts_for_pmids([pmid], chunks_dir))
        progress_bar.progress(i / total_pmids)
    
    status_text.text('Search completed.')
    
    return top_results, abstracts


def generate_results_file(results, abstracts):
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as tmpfile:
        for result in results:
            score, pmid = result
            tmpfile.write(f"PMID: {pmid}; Score: {score:.2f}\n")
            tmpfile.write(f"Abstract: {abstracts.get(str(pmid), 'No abstract available')}\n\n")
        return tmpfile.name

# Streamlit UI setup
st.image('logo/disco_pubmed.jpg', caption='logo by DALL-E', width=100, use_column_width='always')  # This will center the image


# Display the header with HTML for extra space
st.markdown("""
    <h1 style='font-size: 40px; text-align: center;'>LocalDiscoPub</h1>
    <h2 style='text-align: center;'>DISCOver PUBmed LOCALly!</h2>
    <br>  <!-- This adds a bit of extra space -->
    """, unsafe_allow_html=True)

# Allow users to input the
chunks_dir = st.text_input('Enter the path to the PubMed chunks directory:', 'pubmed/test')

# User input for query with an option to upload a file
st.write("Enter your query:")
user_query = st.text_input("Type your query here:", "")
query_file = st.file_uploader("Or upload a file with your query (TXT or CSV):", type=['txt', 'csv'])

# Handling file uploads and determining the query based on file type
if query_file is not None:
    # Determine the file type
    file_type = query_file.type
    
    # Process a TXT file
    if file_type == "text/plain":
        user_query = query_file.getvalue().decode("utf-8").strip()
    
    # Process a CSV file
    elif file_type == "text/csv":
        # Assuming the query is in the first row and first column of the CSV
        df = pd.read_csv(query_file)
        user_query = df.iloc[0, 0]
        
# Placeholders for status and progress should be initialized here to appear below the 'Search' button
status_text = st.empty()
progress_bar = st.progress(0)

if st.button('Search'):
    # First, check if both the directory and query inputs are provided
    if not chunks_dir or not user_query.strip():
        st.error("Please enter both a valid PubMed chunks directory and a query to search.")
    elif not os.path.exists(chunks_dir):  # Then, verify the directory exists
        st.error("The specified directory does not exist. Please enter a valid path.")
    else:
        # If all conditions are met, perform the search
        top_results, abstracts = search_pubmed(user_query, chunks_dir)
        status_text.text('Search completed.')

        # Clear the progress bar after completing the search and loading abstracts
        progress_bar.empty()

        if top_results:
            for score, pmid in top_results:
                abstract = abstracts.get(str(pmid), "No abstract available")
                # Use HTML tags within Markdown for smaller font size and concise display
                st.markdown(f"<span style='font-size: 90%'><b>PMID:</b> {pmid}  <b>Score:</b> {score:.2f}<br><b>Abstract:</b> {abstract}</span>", unsafe_allow_html=True)
                st.markdown("---")  # Separator line for readability

        # Optionally, allow users to download the results
        if top_results:
            results_file_path = generate_results_file(top_results, abstracts)
            with open(results_file_path, "rb") as file:
                st.download_button(
                    label="Download Search Results",
                    data=file,
                    file_name="PubMed_Search_Results.txt",
                    mime="text/plain"
                )

# The final line outside of the button click logic
st.markdown('<p style="text-align: right;">created by Nikola Tom</p>', unsafe_allow_html=True)
