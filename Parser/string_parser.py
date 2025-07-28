from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv()

parser= StrOutputParser()

llm= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
)

prompt1= PromptTemplate(
    template="write a detailed report on the {topic}",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template="write a 5 summary on the following  report on the {report}",
    input_variables=["report"]
)

chain = prompt1 | llm |parser | prompt2 | llm | parser
result= chain.invoke({"topic": "mca cricketing rules"})
print("report is: ", result)