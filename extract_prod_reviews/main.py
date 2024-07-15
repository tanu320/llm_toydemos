from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_openai import OpenAI
template = """\
For the following text, extract the following \
information:

sentiment: Is the customer happy with the product? 
Answer Positive if yes, Negative if \
not, Neutral if either of them, or Unknown if unknown.

delivery_days: How many days did it take \
for the product to arrive? If this \
information is not found, output No information about this.

price_perception: How does it feel the customer about the price? 
Answer Expensive if the customer feels the product is expensive, 
Cheap if the customer feels the product is cheap,
not, Neutral if either of them, or Unknown if unknown.

Format the output as bullet-points text with the \
following keys:
- Sentiment
- How long took it to deliver?
- How was the price perceived?

Input example:
This dress is pretty amazing. It arrived in two days, just in time for my wife's anniversary present. It is cheaper than the other dresses out there, but I think it is worth it for the extra features.

Output example:
- Sentiment: Positive
- How long took it to deliver? 2 days
- How was the price perceived? Cheap

text: {review}
"""

temp = PromptTemplate(
    template= template,
    input_variables="review"
)

st.set_page_config("Extract Key information from product Reviews")
st.title("Extract Key Information from a Product Review")
col1, col2 = st.columns(2)

with col1:
    st.markdown("Extract Key information from a product review")
    st.markdown("""
        - Sentiment
        - How long took it to deliver ?
        - How was its price perceived ?
    """)

with col2:
    st.markdown("Made from OpenAI's LLM models")

st.markdown("Enter your OpenAI API Key")

def get_openai_key():
    openai_api_key = st.text_input(label = "OpenAI API Key", placeholder= "Enter your key...", type = "password")
    return openai_api_key

openai_api_key = get_openai_key()

st.markdown("Enter the product Review")

def get_review():
    review_text = st.text_area(label = "Product Review", label_visibility="collapsed", placeholder="Your Product Review...", key = "review_input")
    return review_text

review_input = get_review()

if len(review_input.split(" "))>700:
    st.write("Please enter a shorter product review. The maximum length is 700 words")
    st.stop()

st.markdown("### Key Data Extracted")

if review_input:
    if not openai_api_key:
        st.warning("Please intern an OpenAI API Key")
        st.stop()
    
    llm = OpenAI(openai_api_key=openai_api_key)
    prompt_with_input_review = temp.format(review = review_input)
    key_data_extraction = llm(prompt_with_input_review)
    st.write(key_data_extraction)