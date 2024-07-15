import streamlit as st
from langchain_openai import OpenAI



def get_openai_api_key():
    input_text_openai_key = st.text_input(label = "OpenAI Key", placeholder = "Youre key goes here...", key = "openai_api_key_input", type = "password")
    return input_text_openai_key


def load_LLM(openai_api_key):
    llm = OpenAI(model = "gpt-3.5-turbo-0125", temperature = 0.8, openai_api_key = openai_api_key)
    return llm

def get_text_input_draft():
    input_user_text_content = st.text_input(label = "Text to rewrite", label_visibility = "collapsed", placeholder = "Your Text...", key = "user_text_input")
    return input_user_text_content


