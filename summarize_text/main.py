from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

def generate_response(text_input, openai_api_key):
    llm = OpenAI(
        openai_api_key = openai_api_key
    )
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(text_input)
    docs = [Document(page_content = t) for t in texts]
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce"
    )    
    return chain.run(docs)


st.set_page_config("Generate Text Summary")

st.title("Generate Text Summary")

text_input = st.text_area(
    "Enter your text",
    height = 200
)

result = []

with st.form("summarize-form", clear_on_submit = True, border = True):
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type = "password",
        disabled = not text_input

    )
    submitted = st.form_submit_button("Submit")
    if submitted and openai_api_key.startswith("sk-"):
        response = generate_response(text_input, openai_api_key)
        result.append(response)
        del openai_api_key
    
    if len(result):
       st.info(response)