from langchain.prompts import PromptTemplate
import streamlit as st
import utils

template = """

As experienced blog technology and innovation writer,
generate a 400-word blog post about {topic}

Your response should be in the format:
First, print the blog post.
Then, sum the total number of words on it and print the result like this : This post has X words.

"""

prompt = PromptTemplate(template=template, input_variables=["topic"])

st.set_page_config("Write your Blog Post")

st.title("Generate your blogpost...")

openai_api_key_user = utils.get_openai_api_key()

st.markdown("Specify your topic:")
blog_topic = st.text_input(label = "Blog Topic", max_chars = 80)

prompt_with_input = prompt.format(topic = blog_topic)
st.markdown("Your blog goes here :")

if not openai_api_key_user.startswith("sk-"):
    st.warning("Please give valid OpenAI Key")

if openai_api_key_user.startswith("sk-"):
    utils.generate_blog(prompt_with_input, openai_api_key_user)



