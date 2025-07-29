from langchain_core.runnables import RunnablePassthrough, RunnableParallel
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
    template="write a joke of 2 lines on the given title: {title}",
    input_variables=["title"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke in 2 lines: {joke}",
    input_variables=["joke"]
)

parser= StrOutputParser()

joke_generator= prompt| llm| parser

branch_chain= RunnableParallel(
    joke= RunnablePassthrough(),
    explanation= prompt2 | llm | parser
)

chain= joke_generator | branch_chain

result = chain.invoke({"title": "SRK"})
print(result)
