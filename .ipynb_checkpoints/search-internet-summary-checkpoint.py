import requests
from bs4 import BeautifulSoup
from googlesearch import search
import openai
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document
#from langchain.chat_models import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from reference import encoder, dims, getIndex, llm

# Function to find downloadable files from a URL
def find_downloadable_links(url):
    try:
        # Send HTTP request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        # Parse the page content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all links in the page
        links = soup.find_all('a', href=True)

        downloadable_files = []
        
        # Check if the link points to a downloadable file (e.g., .pdf, .docx, .txt)
        for link in links:
            href = link['href']
            if href.endswith(('.pdf', '.docx', '.txt', '.xls', '.pptx', '.csv')):
                downloadable_files.append(href)

        return downloadable_files
    except Exception as e:
        print(f"Error with URL {url}: {e}")
        return []

# Function to download the document content
def download_document(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error downloading document from {url}: {e}")
        return None

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_data):
    try:
        pdf_reader = PdfReader(BytesIO(pdf_data))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Function to extract text from a Word document (.docx)
def extract_text_from_docx(docx_data):
    try:
        doc = Document(BytesIO(docx_data))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

# Function to extract text from a text file
def extract_text_from_txt(txt_data):
    return txt_data.decode('utf-8')

# Function to summarize text using LangChain and GPT-4
def summarize_text_with_langchain(text):
    try:
        # Create a prompt template for summarization
        prompt_template = "Please summarize the following text:\n\n{text}"
        prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
        
        # Create the LLM chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Get the summary from the model
        summary = chain.run(text)
        return summary
    except Exception as e:
        print(f"Error summarizing with LangChain: {e}")
        return "Error generating summary."

# Function to get a summary of the document
def get_document_summary(url):
    # Download the document based on its extension
    document_data = download_document(url)
    
    if not document_data:
        return "Unable to download the document."

    # Check the file type by extension and extract text accordingly
    if url.endswith(".pdf"):
        text = extract_text_from_pdf(document_data)
    elif url.endswith(".docx"):
        text = extract_text_from_docx(document_data)
    elif url.endswith(".txt"):
        text = extract_text_from_txt(document_data)
    else:
        return "Unsupported file type for summarization."

    # If no text extracted, return a message
    if not text.strip():
        return "No text extracted from the document."

    # Summarize the extracted text using LangChain and GPT-4
    return summarize_text_with_langchain(text)

# Function to perform the search and get the links to documents
def search_documents(query, num_results=5):
    print(f"Searching for '{query}'...")
    
    # Perform Google search and get the URLs of the top results
    search_results = search(query, num_results=num_results)

    all_downloadable_links = {}

    # Check each URL for downloadable files
    for result in search_results:
        print(f"Checking {result}...")
        downloadable_links = find_downloadable_links(result)
        if downloadable_links:
            all_downloadable_links[result] = downloadable_links

    return all_downloadable_links

# Main function to search and display downloadable file links
def main():
    query = input("Enter the search query: ")
    num_results = 10  #int(input("How many search results do you want to check? "))

    downloadable_files = search_documents(query, num_results)

    if downloadable_files:
        print("\nFound downloadable files at these links:")
        for url, files in downloadable_files.items():
            print(f"\nURL: {url}")
            for file in files:
                print(f"  - {file}")
                # Get a summary of each file
                summary = get_document_summary(file)
                print(f"    Summary: {summary}")
    else:
        print("No downloadable files found.")

# Run the script
if __name__ == "__main__":
    main()
