from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

splitter= RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size= 30,
    chunk_overlap=0,
)

text= """def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Example usage
number = int(input("Enter a number: "))
if is_prime(number):
    print(f"{number} is a prime number ✅")
else:
    print(f"{number} is not a prime number ❌")"""

chunks = splitter.split_text(text)
print(chunks)

