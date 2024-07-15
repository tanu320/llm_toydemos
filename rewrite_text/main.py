import streamlit as st
from langchain_core.prompts import PromptTemplate
import utils
# from utils import get_openai_api_key, load_LLM, get_text_input_draft

template = """
    Below is a draft text that may be poorly worded.
    Your goal is to:
    - Properly redact the draft text
    - Convert the draft text to a specified tone
    - Convert the draft text to a specified dialect

    Here are some examples different Tones:
    - Formal: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After a period of five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.
    - Informal: Hey everyone, it's been a wild week! We've got some exciting news to share - Sam Altman is back at OpenAI, taking up the role of chief executive. After a bunch of intense talks, debates, and convincing, Altman is making his triumphant return to the AI startup he co-founded.  

    Here are some examples of words in different dialects:
    - American: French Fries, cotton candy, apartment, garbage, \
        cookie, green thumb, parking lot, pants, windshield
    - British: chips, candyfloss, flag, rubbish, biscuit, green fingers, \
        car park, trousers, windscreen

    Example Sentences from each dialect:
    - American: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After a period of five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.
    - British: On Wednesday, OpenAI, the esteemed artificial intelligence start-up, announced that Sam Altman would be returning as its Chief Executive Officer. This decisive move follows five days of deliberation, discourse and persuasion, after Altman's abrupt departure from the company which he had co-established.

    Please start the redaction with a warm introduction. Add the introduction \
        if you need to.
    
    Below is the draft text, tone, and dialect:
    DRAFT: {draft}
    TONE: {tone}
    DIALECT: {dialect}

    YOUR {dialect} RESPONSE:

"""

prompt = PromptTemplate(
    template = template,
    input_variables = ["draft", "tone", "dialect"]
)


st.set_page_config("Rewrite your Text")
st.header("Rewrite your Text content")


col1, col2 = st.columns(2)

with col1:
    st.markdown("Rewrite your text in different styles")

with col2:
    st.markdown("Using LLM from OpenAI")

st.markdown("Enter your OpenAI API Key")

openai_api_key = utils.get_openai_api_key()

user_text_input = utils.get_text_input_draft()

if len(user_text_input.split(" ")) > 700:
    st.write("Enter shorter text, max limit is 700 words")
    st.stop()


col1, col2 = st.columns(2)

with col1:
    option_tone = st.selectbox(
        'Which tone would you like your text to have ?',
        ('Formal','Informal')
    )

with col2:
    option_dialect = st.selectbox(
        'Which English Dialect would you like ?',
        ('American','British')
    )

st.markdown("Your rewritten text :")

if user_text_input:
    if not openai_api_key:
        st.warning('Please give API Key')
        st.stop()
    
    llm = utils.load_LLM(openai_api_key = openai_api_key)

    prompt_with_input = prompt.format(
        tone = option_tone,
        draft = user_text_input,
        dialect = option_dialect
    )


    rewritten_input = llm(prompt_with_input)

    st.write(rewritten_input)


