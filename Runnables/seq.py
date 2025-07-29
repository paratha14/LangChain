from langchain_core.runnables import RunnableSequence
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
    template="write a joke on the given title: {title}",
    input_variables=["title"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke: {joke}",
    input_variables=["joke"]
)

parser= StrOutputParser()

chain= RunnableSequence(
    prompt,
    llm,
    parser,
    prompt2,
    llm,
    parser
)

result = chain.invoke({"title": "sidhu paaji"})
print(result)
