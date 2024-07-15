import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import os
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader

def load_llm(openai_api_key):
    return OpenAI(openai_api_key = openai_api_key, temperature = 0)

st.set_page_config("Question from CSV Files")
st.title("CSV File Question and Answer")

def get_openai_api_key_user():
    key_text = st.text_input(label = "API KEY", placeholder = "Enter your API Key", type = "password")
    return key_text

openai_api_key = get_openai_api_key_user()
if openai_api_key:
   embedding = OpenAIEmbeddings(openai_api_key = openai_api_key)

   vectordb_file_path = "my_vector_db"

   def create_db_for_csv():
      loader_config = {
         'file_path' : './sample.csv',
         'source_column' : list(pd.read_csv('sample.csv').columns),
         'encoding': 'utf-8'
      }
      loader = CSVLoader(loader_config)
      documents = loader.load()
      vectordb = FAISS.from_documents(documents, embedding)
      vectordb.save_local(vectordb_file_path)
    
   def execute_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embedding)
    retriever = vectordb.as_retriever(score_threshold = 0.7)
    
    template = """ Given the following contet and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from the "response" section in the source document context without making much changes.
    If the answer is not found in the context, respond "I don't know." Don't try to make up an answer.
    CONTEXT: {context}

    QUESTION: {question}
    """

    prompt = PromptTemplate(
       template= template,
       input_variables = ["context", "question"]
    )

    llm_loaded = load_llm(openai_api_key=openai_api_key)

    chain = RetrievalQA.from_chain_type(
       llm = llm_loaded,
       chain_type="stuff",
       retriever = retriever,
       input_key = "query",
       return_source_documents = True,
       chain_type_kwargs={"prompt":prompt}       
    )

    return chain

   
if __name__ == "main":
   create_db_for_csv()
   chain = execute_chain()

button = st.button("Private button: Re create database")
if button:
   create_db_for_csv()

ques = st.text_input("Question: ")

if ques:
   chain = execute_chain()
   response = chain(ques)

   st.header("Answer: ")
   st.write(response["result"])



   
