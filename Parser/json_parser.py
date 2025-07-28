from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

llm= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
)

parser= JsonOutputParser()

template= PromptTemplate(
    template= "give me a name, age, city of a fictional character / person \n {format_instruction}",
    input_variables=[],
    partial_variables= {"format_instruction":parser.get_format_instructions()}

)

prompt= template.format()
chain = template |llm |parser

result= chain.invoke({})
print(result)


