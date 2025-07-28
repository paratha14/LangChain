from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv()

llm= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
)
 
schema= [
    ResponseSchema(name="name", desciption="name of the person"),
    ResponseSchema(name="age", desciption="age of the person"),
    ResponseSchema(name="city", desciption="city of the person"),
    
]

parser = StructuredOutputParser.from_response_schemas(schema)

template= PromptTemplate(
    template= "give me a name, age, city of a fictional character / person \n {format_instruction}",
    input_variables=[],
    partial_variables= {"format_instruction":parser.get_format_instructions()}



)

prompt= template.format()
chain= prompt | llm | parser
result= chain.invoke({})
print(result)
