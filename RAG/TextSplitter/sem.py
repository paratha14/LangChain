sample= """
Mountains are towering natural structures formed over millions of years by tectonic forces. They provide a home to diverse wildlife and are a source of freshwater for many regions. People often visit mountains for hiking, skiing, and breathtaking views. Despite their beauty, mountains can be dangerous due to avalanches and harsh weather conditions.

Popping bubble wrap is oddly satisfying and often used as a stress reliever during packaging. Meanwhile, philosophy dives deep into the nature of existence, morality, and consciousness. While one offers mindless fun, the other demands mindful reflection. Yet in some strange way, both can bring peace — one to the fingers, the other to the mind.
"""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
embedding= HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    #dimensions=300
)

splitter= SemanticChunker(
    embedding=embedding,
    
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)

chunks= splitter.split_text(sample)
print(chunks)
print(len(chunks))
