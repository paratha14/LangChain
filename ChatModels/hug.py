from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

llm= HuggingFacePipeline.from_model_id(
    model_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task= "text-generation",
    
)

model= ChatHuggingFace(llm=llm)

result= model.invoke("tell me some og games of 2010-2017 innn pc world like open world")
print(result.text())