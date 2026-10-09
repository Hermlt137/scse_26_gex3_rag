from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag_tool import (
    search_university_knowledge
)


## USE THE QWEN 0.6B MODEL (not the 8B model from Lecture 11)
model = ChatOllama(
    model="qwen3:0.6b",
    temperature=0
)


## THE AGENT ALWAYS USES THE RAG TOOL TO RETRIEVE THE UNIVERSITY
## DOCUMENTATION BEFORE IT ANSWERS.
##
## The qwen3-0.6b model is too small to reliably decide on its own when to
## call tools, so the retrieval step is done first and its result is given
## to the model as the only allowed source of facts.
question = """

What exact phone number should I call for the University IT Service Desk?

"""


## STEP 1: RETRIEVE THE RELEVANT UNIVERSITY POLICIES WITH THE RAG TOOL
context = search_university_knowledge.invoke({
    "question": question
})


print("\nRETRIEVED CONTEXT")
print(context)


## STEP 2: ANSWER USING ONLY THE RETRIEVED CONTEXT
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a university support assistant.

        Answer the question using only the supplied university documentation.

        Never invent phone numbers, contact details, university policies, or
        university procedures.

        If the documentation does not contain the answer, reply only that the
        information is not available in the university documentation.
        Do not suggest other sources, phone numbers, or instructions.
        """
    ),
    (
        "human",
        """
        UNIVERSITY DOCUMENTATION:

        {context}

        QUESTION:

        {question}
        """
    )
])


agent_chain = prompt | model


result = agent_chain.invoke({
    "context": context,
    "question": question
})


print("\nFINAL ANSWER")

print(result.content)
