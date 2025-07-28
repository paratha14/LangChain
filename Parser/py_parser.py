from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.6,
)

class Person(BaseModel):
    name: str = Field(...,description="description of the person")

    age: int= Field(...,description="age of the person")

    country: str = Field(...,description="Country of the person")

parser= PydanticOutputParser(pydantic_object=Person)

template= PromptTemplate(
    template="give me a name, age, Country of a fictional character / person \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

prompt= template.format()

chain= prompt | llm |parser
result= chain.invoke({})
print(result)


