from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
model= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature= 0.6,
    max_output_tokens= 100
)
messages=[
    SystemMessage(content="you are a very helpful and co-operative chatbot")
]

user_name = input("What would you like to be called? ")
while True:
    
    user_input= input(f"{user_name}: ")
    if (user_input=='exit'):
        break
    messages.append(HumanMessage(content=user_input))
    

    response= model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print(f'ChatBot: {response.content}')
print(messages)