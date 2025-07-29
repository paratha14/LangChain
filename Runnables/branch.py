from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

class Review(BaseModel):
    review: Literal["positive", "negative"] = Field(..., description="The review text to be analyzed")

parser2 = PydanticOutputParser(pydantic_object=Review)

prompt1 = PromptTemplate(
    template="analyse the following reviews/ feebacks and then state whether it is positive or negative: {review} \n{format_instruction}",
    input_variables=["review"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="write a thanking note for the following positive review: {review}",
    input_variables=["review"]
)

prompt3 = PromptTemplate(
    template="write a apology note for the following negative review and promise improvements: {review}",
    input_variables=["review"]
)

parser = StrOutputParser()

checking_chain = prompt1 | llm | parser2

def route_branch(data):
    return data["review"] == "positive"

branch_chain = RunnableBranch(
    (lambda x: x["review"] == "positive", prompt2 | llm | parser),
    (lambda x: x["review"] == "negative", prompt3 | llm | parser),
)

full_chain = (
    checking_chain | branch_chain | parser
)

result = full_chain.invoke({"review": "The product was great and I loved it!"})
print(result)
