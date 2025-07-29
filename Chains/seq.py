from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm= GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
)
prompt1 = PromptTemplate(
    template="give me detailed report on the topic: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="give me a 5 pointer summary of the the following report: {report}",
    input_variables=["report"]

)


parser= StrOutputParser()

chain= prompt1 | llm | parser | prompt2 | llm | parser
target= input("enter topic: ")

result= chain.invoke({"topic": target})
print(result)




