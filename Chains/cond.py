from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
)

# Output parser for plain string responses
parser = StrOutputParser()

# Pydantic model for sentiment
class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Analyze the sentiment of the text")

# Output parser for sentiment classification
parser2 = PydanticOutputParser(pydantic_object=Sentiment)

# Sentiment classification prompt
prompt = PromptTemplate(
    template="Analyze the sentiment of the following text: {text}\n{formatted_instructions}",
    input_variables=["text"],
    partial_variables={"formatted_instructions": parser2.get_format_instructions()}
)

# Chain to classify sentiment
checker_chain = prompt | llm | parser2

# Positive feedback response prompt
prompt2 = PromptTemplate(
    template="Write a short text about positive feedback from the user and assure them that you'll send the feedback form. Feedback: {feedback}",
    input_variables=["feedback"],
)

# Negative feedback response prompt
prompt3 = PromptTemplate(
    template="Write a short text apologizing for a negative review. Feedback: {feedback}",
    input_variables=["feedback"],
)

# Chains to handle responses
chain1 = prompt2 | llm | parser
chain2 = prompt3 | llm | parser

# Branch chain to route based on sentiment
branch_chain = RunnableBranch(
    (lambda x: x['sentiment'] == "positive", RunnableLambda(lambda x: {"feedback": x["text"]}) | chain1),
    (lambda x: x['sentiment'] == "negative", RunnableLambda(lambda x: {"feedback": x["text"]}) | chain2),
    RunnableLambda(lambda x: "No sentiment detected")
)

# Full chain: classify → branch based on sentiment → respond
chain = checker_chain | (lambda s: {"sentiment": s.sentiment, "text": s.dict().get("text", "")}) | branch_chain

# Invoke chain with input
input_text = "This is a great product, I really love it!"
result = chain.invoke({"text": input_text})
print(result)
