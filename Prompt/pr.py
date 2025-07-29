from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()
model= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature= 0.4

)
shot= input("enter shot type: ")
bowler_type= input("enter bowler type: ")

template= PromptTemplate(
    template="""summarize or explain me:{shot} and give me a fielding approach for a {bowler_type} bowler in cricket""",
    input_variables=["shot", "bowler_type"]

)


prompt= template.invoke({
    'shot': shot,
    'bowler_type': bowler_type
})

result=  model.invoke(prompt)
print(result.content)