from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from io import StringIO
import utils




# template = """

# """

st.set_page_config("Long Text Summarizer")
st.header("Generate Summary for long texts files through OpenAI GPT LLM")
st.write("ChatGPT can summarize your long texts")


st.markdown("Enter your OPENAI API Key")
user_openai_api_key = utils.get_openai_api_key()



st.markdown("Upload the text you want to summarize")


file_uploaded = st.file_uploader("Choose your file", type = "txt")
if file_uploaded is not None:
    stringio = StringIO(file_uploaded.getvalue().decode("utf-8"))
    string_data = stringio.read()

    if len(string_data.split(" "))>20000:
        st.write("Please upload a small file, max length is 20000 words.")
        st.stop()

    if string_data:
        if not user_openai_api_key:
            st.warning("Please specify you OPEN AI API Key")
            st.stop()
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size = 5000,
        chunk_overlap = 350
    )

    splitted_documents = text_splitter.create_documents([])

    llm = utils.load_llm(user_openai_api_key)

    summarize_chain = load_summarize_chain(
        llm = llm,
        chain_type = "map_reduce"
    )

    summary_output = summarize_chain.run(splitted_documents)
    st.markdown("Here is your summary")
    st.write(summary_output)




