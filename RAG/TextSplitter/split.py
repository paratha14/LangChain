from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)
parser = StrOutputParser()
#prompt = PromptTemplate()

loader= PyPDFLoader('demo.pdf')
docs= loader.load()

splitter= CharacterTextSplitter(
    chunk_size= 150,
    chunk_overlap= 0,
    separator=''
)

chunks= splitter.split_documents(docs)

print(chunks[0].page_content)

