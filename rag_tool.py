from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool

from policy_loader import load_policy_documents


## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()


## CREATE EMBEDDING MODEL using the OLLAMA embedding model "nomic-embed-text"
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


## CREATE VECTOR STORE using CHROMA
vector_store = Chroma(
    collection_name="university_it_policies_agent",
    embedding_function=embeddings
)


## ADD ABOVE DOCUMENTS TO THE STORE
## The knowledge base has many policy records, so add them in batches.
BATCH_SIZE = 64

for start in range(0, len(documents), BATCH_SIZE):

    vector_store.add_documents(
        documents[start:start + BATCH_SIZE]
    )


## RAG tool for searching the university IT knowledge base
@tool
def search_university_knowledge(
    question: str
) -> str:
    """
    Search the university IT documentation for information
    about university procedures, policies and instructions.
    """

    results = vector_store.similarity_search(
        question,
        k=3
    )

    retrieved = []

    for document in results:

        metadata = document.metadata

        retrieved.append(
            f"SOURCE: {metadata['source']} "
            f"({metadata['policy_id']}, topic={metadata['topic']})\n"
            f"{document.page_content}"
        )

    return "\n\n".join(retrieved)
