from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

llm= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
)

embedder= HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

parser= StrOutputParser()

loader= PyPDFLoader('demo.pdf')
docs=loader.load()

splitter= RecursiveCharacterTextSplitter(
    chunk_size=170,
    chunk_overlap=4
)

chunks= splitter.split_documents(docs)

print(len(chunks))

vector_store= Chroma(
    embedding_function= embedder,
    collection_name="qna_collection",
    persist_directory="chroma_db"
    
)

vector_store.add_documents(documents=chunks)

query= input("Enter Your Question Here! ")

base_ret= vector_store.as_retriever(
    embeddings=embedder,
    search_type="mmr",
    search_kwargs={"k": 2}
)

comp= LLMChainExtractor.from_llm(llm)

compresser_ret= ContextualCompressionRetriever(
    base_retriever=base_ret,
    base_compressor=comp,
    )

result= compresser_ret.invoke(query)
print(result)
for idx, ans in enumerate(result):
    print(f"Answer {idx+1}, \n")
    print(f"Content: {ans.page_content}, \n")


