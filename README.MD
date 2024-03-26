# LocalDiscoPub - DISCOver PUBmed LOCALly

Welcome to LocalDiscoPub, a powerful tool designed to revolutionize the way researchers and professionals conduct semantic searches through the PubMed database, right from the comfort of their local machines. Leveraging cutting-edge AI, LocalDiscoPub facilitates deep, insightful searches across PubMed's vast repository of medical literature, offering unparalleled access to abstracts, full texts, and other critical knowledge bases.

## Features

- **Semantic Search**: Utilize advanced natural language processing (NLP) techniques to conduct semantic searches through the PubMed database based on user queries.
- **Local Database Utilization**: Access PubMed data locally, ensuring speed and efficiency without the need for constant internet access.
- **Streamlit GUI**: Enjoy a user-friendly graphical interface designed with Streamlit, making it easy to input queries and review results.
- **Continuous Updates**: Keep your local PubMed database current with the option to update abstracts, full texts, and other knowledge bases.
- **Comprehensive Results**: Retrieve detailed search outcomes, including PubMed IDs (PMID), relevance scores, and abstracts, perfect for further research or analysis.

## Components

LocalDiscoPub is built using several key components:

1. **Local PubMed Database**: Should be downloaded from [NCBI FTP](https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/), this database is the backbone of the application, containing the necessary data for semantic searches.
2. **Article Encoder**: Embedded with an article encoder from [Hugging Face - MedCPT-Article-Encoder](https://huggingface.co/ncbi/MedCPT-Article-Encoder), enabling the conversion of articles into machine-readable formats for processing. This doesn't need to be used when using default pubmed database downloaded from the link above.
3. **Model and Tokenizer**: The core AI components, including the model and tokenizer, are sourced from [ncbi/MedCPT-Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder), ensuring accurate understanding and processing of user queries. These are downloaded into the cash during the 1st run.

## Getting Started

To begin using LocalDiscoPub, follow these simple steps:

1. **Installation**: Clone this repository to your local machine. Follow the commands in `bash install.sh`
2. **Database Setup**: Download the initial PubMed database from the provided NCBI FTP link and place it in the specified directory within your local setup.
4. **Launch**: Run the Streamlit application by navigating to the application directory and executing `streamlit run app.py`.

## Usage

1. **Enter path to pubmed database**
2. **Enter Query**: Launch the Streamlit GUI and input your search query into the provided text box.
3. **Search**: Initiate the search process. The system will process your query against the local PubMed database using advanced NLP techniques.
4. **Review Results**: Browse through the search results, which include PMIDs, relevance scores, and abstracts. Utilize these results for further research, downloading full texts, or additional data processing.
5. **Download the results**

## Updating the Database

To ensure your searches are as current as possible, regularly update your local PubMed database with new abstracts, full texts, or other relevant knowledge bases. Follow the instructions provided in the [`update_guide.md`](https://github.com/ncbi/MedCPT) for detailed steps on how to perform updates.

## Contribution

We welcome contributions from the community, whether it's through improving the codebase, expanding the database, or enhancing the GUI. 

## License

LocalDiscoPub is released under the MIT License. See the `LICENSE` file for more details.

## Support

If you encounter any issues or have questions, please open an issue on the GitHub repository, or contact the maintainers directly.

Happy searching!