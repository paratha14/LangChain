from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature= 0.4
)

messages= [
    SystemMessage(content="You are a helpful assistant that summarizes text and provides fielding strategies in cricket."),
    HumanMessage(content="Summarize the following cricket shot and provide a fielding approach for a fast bowler: 'The batsman played a cover drive with precision, sending the ball racing to the boundary.'"),
    
]

result= model.invoke(messages)
messages.append(AIMessage(content=result.content)) 

print(messages)
