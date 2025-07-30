from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

loader= TextLoader('demo.txt')
docs= loader.load()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

splitter= RecursiveCharacterTextSplitter(
    chunk_size= 25,
    chunk_overlap=0,
    
)

chunks= splitter.split_text(docs[0].page_content)
print(chunks)