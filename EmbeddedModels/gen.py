from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

embedding= GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    dimensions=40
    
    #max_tokens= 1000
    #api_key=
)
result= embedding.embed_query("sleeping dog is an og game")
print(str(result))
