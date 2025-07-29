from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

import streamlit as st

load_dotenv()

model= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature= 0.4
)

st.header("Text Summarization")

user_input= st.text_input("enter your text here: ")

if st.button:
    st.text("summarizing....")
    response= model.invoke(user_input)
    st.write(response.content)
    print(response.content)



