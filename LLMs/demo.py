import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()

api=os.getenv('GOOGLE_API_KEY')

llm= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

response= llm.invoke("hello")
print(response.text())