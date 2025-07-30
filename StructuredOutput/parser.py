from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

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

topic= input("Enter the topic for the report: ")
prompt_1 = prompt1.invoke({"topic": topic})
result_1= llm.invoke(prompt_1)
prompt_2 = prompt2.invoke({"report": llm.invoke(prompt_1).content})
result_2= llm.invoke(prompt_2)

print("report is: ", result_1.content)
print("summary is: ", result_2.content)