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

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

def search_pubmed(query, chunks_dir):
    with torch.no_grad():
        status_text.text('Encoding query...')
        encoded = tokenizer([query], truncation=True, padding=True, return_tensors='pt', max_length=64)
        query_embeds = model(**encoded).last_hidden_state[:, 0, :].numpy()

    results = []
    chunk_files = [file for file in sorted(os.listdir(chunks_dir)) if file.startswith("embeds_chunk") and file.endswith(".npy")]
    total_chunks = len(chunk_files)
    
    for i, file in enumerate(chunk_files, start=1):
        status_text.text(f'Loading embeddings and PMIDs: Processing chunk {i}/{total_chunks}...')
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

    top_results = nlargest(top_n, results, key=lambda x: x[0])
    top_pmids = [str(pmid) for _, pmid in top_results]

    progress_bar.empty()
    status_text.text('Loading abstracts for top results...')
    
    abstracts = {}
    total_pmids = len(top_pmids)

    # THIS WAS DEACTIVATED IN ORDER TO MAKE THE SEARCH FAST
    # for i, pmid in enumerate(top_pmids, start=1):
    #     abstracts.update(load_abstracts_for_pmids([pmid], chunks_dir))
    #     progress_bar.progress(i / total_pmids)
    
    status_text.text('Search completed.')
    
    return top_results, abstracts

def generate_results_file(results, abstracts, directory, gene_name=""):
    results_file_path = os.path.join(directory, f"{gene_name}_results.txt")
    with open(results_file_path, 'w') as tmpfile:
        for result in results:
            score, pmid = result
            tmpfile.write(f"PMID: {pmid}; Score: {score:.2f}\n")
            tmpfile.write(f"Abstract: {abstracts.get(str(pmid), 'No abstract available')}\n\n")
    return results_file_path
        
def generate_pmids_file(results, directory, gene_name=""):
    pmids_file_path = os.path.join(directory, f"{gene_name}_pmids.txt")
    with open(pmids_file_path, 'w') as tmpfile:
        for _, pmid in results:
            tmpfile.write(f"{pmid}\n")
    return pmids_file_path

def concatenate_files(file_paths, output_path):
    with open(output_path, 'w') as outfile:
        for file_path in file_paths:
            with open(file_path, 'r') as infile:
                outfile.write(infile.read())

# Streamlit UI setup
st.image('logo/disco_pubmed.jpg', caption='logo by DALL-E', width=100, use_column_width='always')

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

# New Feature: Gene list input and query generation
st.write("Or enter a list of genes to generate a query:")
genes = st.text_input("Enter a comma-separated list of genes:", "")  # Input field for genes

# Option to upload a file containing the list of genes
gene_file = st.file_uploader("Or upload a file with the list of genes (TXT or CSV):", type=['txt', 'csv'])

# Checkbox for individual gene searches
individual_gene_search = st.checkbox("Search for each gene individually")

# Parameter for number of top results
top_n = st.number_input("Number of top results to retrieve", min_value=1, value=10)

# Placeholders for status and progress should be initialized here to appear below the 'Search' button
status_text = st.empty()
progress_bar = st.progress(0)

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

# Handling gene file upload
if gene_file is not None:
    file_type = gene_file.type
    
    # Process a TXT file
    if file_type == "text/plain":
        genes = gene_file.getvalue().decode("utf-8").strip().replace('\n', ',')
    
    # Process a CSV file
    elif file_type == "text/csv":
        df = pd.read_csv(gene_file)
        genes = ','.join(df.iloc[:, 0].astype(str))  # Assuming genes are in the first column

# Automatically combine user query and genes if available
if genes:
    gene_list = genes.split(',')

if st.button('Search'):
    # First, check if both the directory and query inputs are provided
    if not chunks_dir or not user_query.strip():
        st.error("Please enter both a valid PubMed chunks directory and a query to search.")
    elif not os.path.exists(chunks_dir):  # Then, verify the directory exists
        st.error("The specified directory does not exist. Please enter a valid path.")
    else:
        top_results = []
        abstracts = {}
        base_dir = tempfile.mkdtemp()
        results_files = []
        pmids_files = []
        if individual_gene_search:
            for gene in gene_list:
                gene_query = f"{user_query} AND {gene}"  # Combine user query with each gene
                st.text(f"Generated Query for {gene}: {gene_query}")  # Optionally display the generated query
                results, gene_abstracts = search_pubmed(gene_query, chunks_dir)
                top_results.extend(results)
                abstracts.update(gene_abstracts)
                gene_dir = os.path.join(base_dir, gene)
                os.makedirs(gene_dir, exist_ok=True)
                results_file = generate_results_file(results, gene_abstracts, gene_dir, gene_name=gene)
                pmids_file = generate_pmids_file(results, gene_dir, gene_name=gene)
                results_files.append(results_file)
                pmids_files.append(pmids_file)
                # st.write(f"Results for {gene}: {len(results)} PMIDs found")
                # st.write([pmid for _, pmid in results])
                
            # Concatenate all individual files into one
            concatenated_results_file = os.path.join(base_dir, "concatenated_results.txt")
            concatenated_pmids_file = os.path.join(base_dir, "concatenated_pmids.txt")
            concatenate_files(results_files, concatenated_results_file)
            concatenate_files(pmids_files, concatenated_pmids_file)
        else:
            if genes:
                gene_query = " AND ".join(gene_list)  # Constructs a query that searches for all of the genes
                if user_query.strip():
                    user_query = f"{user_query} AND {gene_query}"  # Combines the user query with the gene query
                else:
                    user_query = gene_query
            st.text(f"Generated Query: {user_query}")  # Optionally display the generated query
            top_results, abstracts = search_pubmed(user_query, chunks_dir)
            results_file = generate_results_file(top_results, abstracts, base_dir)
            pmids_file = generate_pmids_file(top_results, base_dir)
            # st.write(f"Results: {len(top_results)} PMIDs found")
            # st.write([pmid for _, pmid in top_results])
            concatenated_results_file = results_file
            concatenated_pmids_file = pmids_file

        status_text.text('Search completed.')

        # Clear the progress bar after completing the search and loading abstracts
        progress_bar.empty()

        # # Optionally, allow users to download the results
        # with open(concatenated_results_file, "rb") as file:
        #     st.download_button(
        #         label="Download Search Results",
        #         data=file,
        #         file_name="PubMed_Search_Results.txt",
        #         mime="text/plain"
        #     )

        with open(concatenated_pmids_file, "rb") as file:
            st.download_button(
                label="Download PMIDs",
                data=file,
                file_name="PubMed_PMIDs.txt",
                mime="text/plain"
            )

# The final line outside of the button click logic
st.markdown('<p style="text-align: right;">created by Nikola Tom</p>', unsafe_allow_html=True)
