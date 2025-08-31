import streamlit as st

from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = InMemoryVectorStore(embeddings)

# Initialize LLM
model = OllamaLLM(model="llama3.2")

# Predefined URL for Freshservice API
url = "https://api.freshservice.com/#ticket_attributes"

# Load and process documents
def load_page(url):
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# Streamlit UI
st.title("AI Crawler")

# Load and index docs automatically
documents = load_page(url)
chunked_documents = split_text(documents)
index_docs(chunked_documents)

# User question input
question = st.chat_input("Ask your question:")

if question:
    st.chat_message("user").write(question)
    retrieved_docs = retrieve_docs(question)
    context_from_docs = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Add manual curl command context for creating tickets
    curl_context = """
    To create a ticket in Freshservice using curl, use this command:

    curl -v -u your_api_key:X -H 'Content-Type: application/json' \\
    -d '{
        "subject": "Example Ticket",
        "description": "Details of the ticket",
        "email": "requester@example.com",
        "priority": 1,
        "status": 2
    }' \\
    -X POST 'https://yourdomain.freshservice.com/api/v2/tickets'
    """

    # Combine scraped docs and manual curl context
    full_context = curl_context + "\n\n" + context_from_docs
    answer = answer_question(question, full_context)
    st.chat_message("assistant").write(answer)
