from langchain_core.runnables import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

prompt = PromptTemplate(
    template="Give me a X post on the following topic {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Give me LinkedIn post n=on the topic: {topic}",
    input_variables=["topic"]
)

parser= StrOutputParser()

parallel_chain= RunnableParallel(
    Tweet= prompt | llm | parser,
    Linkedin= prompt2 | llm | parser
)

result= parallel_chain.invoke({"topic":"My Cricket Journey"})

print(result['Linkedin'])