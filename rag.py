from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from policy_loader import load_policy_documents


## LOAD THE POLICY DOCUMENT (KNOWLEDGE BASE)
## The exercise uses only 1 file: university_it_policies.txt
documents = load_policy_documents()

print(f"Loaded {len(documents)} policy documents.")


## CREATE EMBEDDING MODEL using the OLLAMA embedding model "nomic-embed-text"
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


## CREATE VECTOR STORE using CHROMA
vector_store = Chroma(
    collection_name="university_it_policies",
    embedding_function=embeddings
)


## ADD ABOVE DOCUMENTS TO THE STORE
## The knowledge base has many policy records, so add them in batches.
BATCH_SIZE = 64

for start in range(0, len(documents), BATCH_SIZE):

    vector_store.add_documents(
        documents[start:start + BATCH_SIZE]
    )


## ASK A QUESTION
question = (
    "What exact phone number should I call "
    "for the University IT Service Desk?"
)


## RETRIEVE RELEVANT DOCUMENTS FROM THE STORE
## k=3 means we want the top 3 most relevant documents
results = vector_store.similarity_search(
    question,
    k=3
)


print("\nRETRIEVED DOCUMENTS")

for document in results:

    print("\n--------------------")

    print(
        "Policy:",
        document.metadata["policy_id"],
        "| Topic:",
        document.metadata["topic"]
    )

    print(document.page_content)


## THE RETRIEVED DOCUMENTS ARE THE CONTEXT THAT WE PROVIDE TO THE LLM.
## COMBINE ALL THE DOCUMENTS INTO A SINGLE STRING.
context = "\n\n".join(
    f"[{document.metadata['policy_id']} | {document.metadata['topic']}]\n"
    f"{document.page_content}"
    for document in results
)


## CREATE THE QWEN 0.6B CHAT MODEL (not the 8B model from Lecture 11)
model = ChatOllama(
    model="qwen3:0.6b",
    temperature=0
)


## CREATE A PROMPT THAT WILL PROVIDE THE CONTEXT AND THE QUESTION TO THE MODEL
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a university IT support assistant.

        Answer the question using only the supplied university documentation.

        If the documentation does not contain the answer, say that the
        information is not available in the university documentation.

        Never invent phone numbers, contact details, or university policies.
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


## BUILD A CHAIN THAT WILL FIRST FORMAT THE PROMPT AND THEN CALL THE MODEL
rag_chain = prompt | model


## invoke() the chain with the context and the question
response = rag_chain.invoke({
    "context": context,
    "question": question
})


print("\nFINAL ANSWER")
print(response.content)
