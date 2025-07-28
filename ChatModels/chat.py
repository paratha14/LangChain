from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
api= os.getenv('GOOGLE_API_KEY')

chat_genai = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    #max_tokens= 1000
    #api_key=api
)

def chat(prompt):
    response= chat_genai.invoke(prompt)
    return response.text()

print(chat("tell me some og games of 2010-2017 innn pc world like openworld types"))


