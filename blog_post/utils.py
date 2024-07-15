import streamlit as st
from langchain_openai import OpenAI


def get_openai_api_key():
    openai_api_key_input = st.sidebar.text_input(label = "OpenAI API Key", type = "password")
    return openai_api_key_input

def generate_blog(prompt_with_input, openai_api_key_user):
    llm = OpenAI(temperature = 0.9, openai_api_key = openai_api_key_user)
    blog = llm(prompt_with_input, max_tokens = 2048)
    return st.write(blog)
