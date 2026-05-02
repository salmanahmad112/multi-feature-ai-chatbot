import os
import re
import streamlit as st
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==============================
# Load API Key
# ==============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ==============================
# Languages Dictionary
# ==============================
LANGUAGES = {
    "Original": "original",
    "English": "en",
    "Urdu": "ur",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Chinese": "zh-cn",
    "Arabic": "ar",
    "Hindi": "hi",
    "Turkish": "tr",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko"
}


# ==============================
# Helper Functions
# ==============================
def clean_text(text):
    """Remove unwanted HTML tags from text."""
    return re.sub(r"<.*?>", "", text)


def translate_text(text, target_language):
    """Translate text into the selected language."""
    if target_language == "original":
        return text

    translator = GoogleTranslator(source="auto", target=target_language)
    return translator.translate(text)


def create_chat_history():
    """Create previous chat history for chatbot memory."""
    history = ""

    for message in st.session_state.messages[-8:]:
        role = message["role"]
        content = message["content"]

        if role == "user":
            history += f"User: {content}\n"
        else:
            history += f"Assistant: {content}\n"

    return history


def get_ai_response(user_question, model_name, temperature):
    """Generate chatbot response using Groq LLM."""
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=GROQ_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI chatbot.
You answer questions clearly and simply.
Use the previous chat history to understand context.

Previous Chat History:
{chat_history}
"""
        ),
        ("human", "{question}")
    ])

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    chat_history = create_chat_history()

    response = chain.invoke({
        "chat_history": chat_history,
        "question": user_question
    })

    return clean_text(response)


# ==============================
# Streamlit Page Settings
# ==============================
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Multi-Feature AI Chatbot")
st.write("This chatbot has Q&A, memory, and translation features.")


# ==============================
# Check API Key
# ==============================
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please add your API key in the .env file.")
    st.stop()


# ==============================
# Sidebar Settings
# ==============================
st.sidebar.title("Settings")

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "groq/compound-mini"
    ]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1
)

selected_language = st.sidebar.selectbox(
    "Translate Response To",
    list(LANGUAGES.keys())
)

target_language = LANGUAGES[selected_language]

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# ==============================
# Session State for Memory
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []


# ==============================
# Display Previous Messages
# ==============================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ==============================
# User Input
# ==============================
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Save and display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                ai_response = get_ai_response(
                    user_question=user_input,
                    model_name=model_name,
                    temperature=temperature
                )

                final_response = translate_text(ai_response, target_language)

                st.write(final_response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response
                })

            except Exception as e:
                error_message = f"Error occurred: {e}"
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
