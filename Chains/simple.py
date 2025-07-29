from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm= GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
)
prompt = PromptTemplate(
    template="give me 5 interesting facts about {topic}",
    input_variables=["topic"]
)

parser= StrOutputParser()

chain = prompt |llm |parser
#tar= input("enter target: ")
#result= chain.invoke({"topic": tar})
#print(result)

chain.get_graph().print_ascii()


