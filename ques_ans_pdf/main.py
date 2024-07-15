from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
class langchain.chains.retrieval_qa.base.RetrievalQA
import streamlit as st
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PDFReader

def generate_response(uploaded_file, query_question, openai_api_key):
    reader = PDFReader(uploaded_file)
    formatted_document = []
    for page in reader.pages:
        formatted_document.append(page.extract_text())
    text_splitter = CharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 0
    )
    docs = text_splitter.create_documents(formatted_document)
    embeddings = OpenAIEmbeddings(openai_api_type=openai_api_key)
    store = FAISS.from_documents(docs, embeddings)
    retrieval_chain = RetrievalQA.from_chain_type(
        llm = OpenAI(openai_api_key = user_key),
        chain_type = "stuff",
        retriever = store.as_retriever()
    )
    return retrieval_chain.run(query_question)

st.set_page_config("Question from your PDF")
st.title("Question answer from PDF")
st.markdown("Do question answering on your uploaded PDF HERE")
st.markdown("# Upload your PDF file here")

uploaded_file = st.file_uploader(
    "Upload a .pdf document here",
    type = "pdf"
)
query_text = st.text_input(label = "Your query question", placeholder = "Enter your question here...")
result = []

with st.form("myform", clear_on_submit=True):
    user_key = st.text_input(label = "OpenAI API key", placeholder="Enter your OPENAI API KEY", type = "password", disabled= not (query_text and uploaded_file))
    submitted = st.form_submit_button("Submit", disabled = not(query_text and uploaded_file))
    if submitted and user_key.startswith("sk-"):
        with st.spinner("Wait, please while I generate the response"):
            response = generate_response(
                uploaded_file, 
                query_text,
                user_key
            )
            result.append(response)
            del user_key

if len(result):
    st.info(result)
