from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm= GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
)
prompt1 = PromptTemplate(
    template="give me Notes on the given document {document}",
    input_variables=["document"]
)

prompt2 = PromptTemplate(
    template="give me 5 quiz based on the following Document: {document}",
    input_variables=["document"]
)

prompt3= PromptTemplate(
    template="combine the notes and questions into a single document with the following format:\n\nNotes:\n{notes}\n\nQuestions:\n{questions}",
    input_variables=["notes", "questions"]

)

parser= StrOutputParser()

parallel_chains = RunnableParallel(
    notes=prompt1 | llm | parser,
    questions=prompt2 | llm | parser
)
merge_chain= prompt3 | llm | parser

chain= parallel_chains | merge_chain



document="""Perth Pitch – 2013 Ashes (Australia vs England)
Overview:
The 2013 Ashes Test match at the WACA Ground in Perth, held from December 13–17, was a pivotal third Test in the five-match series. Known for its bouncy and fast surface, the Perth pitch lived up to its reputation and played a crucial role in Australia's series-clinching victory.

🏟 The Perth Pitch – Key Characteristics
Surface: Hard, dry, and bouncy.

Pace & Bounce: Traditionally favored fast bowlers due to steep bounce.

Cracks: As the match progressed, widening cracks caused unpredictable bounce.

Spin Later: Offered some turn for spinners late in the match due to dryness.

🔥 Match Summary
Australia 1st Innings: 385 (Warner 60, Smith 111)

England 1st Innings: 251 (Johnson 4/61)

Australia 2nd Innings: 369 (Watson 103, Warner 112)

England 2nd Innings: 353 (Harris 5/92)

🎉 Result: Australia won by 150 runs and regained the Ashes with a 3–0 lead.

🌟 Star Performers
Mitchell Johnson: Ferocious spells exploiting Perth’s bounce (9 wickets in the match).

David Warner & Shane Watson: Dominated with aggressive batting.

Ryan Harris: Took key wickets on a deteriorating pitch.

💬 Impact of the Pitch
The fast, cracking pitch challenged England’s batsmen, especially against Johnson's pace.

Australian pacers exploited the conditions better than England’s attack.

The pitch became more hostile by Day 4, aiding reverse swing and variable bounce.

🎯 Legacy
The 2013 Perth Test was iconic:

Marked Australia’s Ashes redemption after a 3-0 loss in England earlier that year.

Johnson’s fiery comeback redefined Australia’s bowling dominance."""



result= chain.invoke({"document": document})
print(result)
