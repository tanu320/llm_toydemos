import streamlit as st
from langchain_openai import OpenAI
from io import StringIO



def get_openai_api_key():
    user_api_key = st.text_input(label = "API KEY", placeholder = "Your OpenAI API Key", type = "password")
    return user_api_key




def load_llm(user_openai_api_key):
    return OpenAI(temperature = 0.4, openai_api_key = user_openai_api_key)
