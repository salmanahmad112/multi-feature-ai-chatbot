# Multi-Feature AI Chatbot

This project is a multi-feature AI chatbot built using Python, Streamlit, and LangChain. It can answer user questions, remember chat history during the current session, and translate responses into different languages.

## Features

- AI-based question answering
- Chat memory using Streamlit session state
- Multi-language translation
- Simple Streamlit web interface
- Error handling using try-except
- Secure API key handling using `.env`

## Technologies Used

- Python
- Streamlit
- LangChain
- ChatGroq
- Deep Translator
- python-dotenv

## How It Works

The user enters a question in the chatbot. The input is sent to the Groq language model using LangChain. The model generates a response, and `StrOutputParser` converts it into simple text. The response is displayed in the Streamlit app and stored in `st.session_state` as chat history. If a language is selected, the response is translated using `GoogleTranslator`.

## How to Run

Install required libraries:

```bash
pip install -r requirements.txt
