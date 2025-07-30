from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import SystemMessage, HumanMessage, AIMessage
load_dotenv()

model= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature= 0.6,
    
)

class Review(BaseModel):
    summary: str = Field(..., description="Summary of the review in less than 25 words")
    senti: str= Field(..., description="Sentiment of the review as positive , negative or neutral")
    topics: List[str]= Field(..., description="List of topics discussed in the review")
    pros: List[str]= Field(..., description="List of pros mentioned in the review")
    cons: List[str]= Field(..., description="List of cons mentioned in the review")

structured_model= model.with_structured_output(Review)
result= structured_model.invoke("Beautiful bat, great balance and ping. 6-7 grains were present. Could add a thin grip over the existing grip to enhance the balance even more but that might just be personal preference. Overall a great bat!")

print(f'Summary of the the review includes {result.summary}, having topics like: {result.topics}, with pros like: {result.pros} and cons like: {result.cons}. The sentiment of the review is {result.senti}.')






