from langchain_huggingface import HuggingFaceEmbeddings
import os
from sklearn.metrics.pairwise import cosine_similarity
embedding= HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    #dimensions=300
)

docs= ["Virat Kohli is known for his aggressive batting and has captained India across all formats.",
       "MS Dhoni led India to victory in the 2007 T20 World Cup and the 2011 ODI World Cup.",
       "Rohit Sharma is the only player to have scored three double centuries in ODI cricket.",
       "Sachin Tendulkar is called the God of Cricket and has 100 international centuries.",
       "Jasprit Bumrah is known for his deadly yorkers and is India's premier fast bowler."]

docs_embed= embedding.embed_documents(docs)
#print(docs_embed)

query="Who is called the God of Cricket"

query_embed= embedding.embed_query(query)
#print(query_embed)
scores = cosine_similarity([query_embed], docs_embed)[0]
idx, scores = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]
print(scores)
print(idx)
#print(f"the most relevant document is: {docs[idx]} with a score of {scores}")
print(type(docs[idx]))