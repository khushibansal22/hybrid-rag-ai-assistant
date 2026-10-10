
from langchain_community.vectorstores import FAISS


def create_vector_store(documents, embedding_model):
    """
    Create a FAISS vector store from LangChain Document objects.

    Args:
        documents: List of chunked LangChain Document objects.
        embedding_model: The embedding model used to convert text into vectors.

    Returns:
        A FAISS vector store containing the embeddings and documents.
    """

    vector_store = FAISS.from_documents(
        documents,
        embedding_model
    )

    return vector_store

def save_vector_store(vector_store, path="data/vector_db/faiss_index"):
    """Save the FAISS index and its associated documents locally."""
    vector_store.save_local(path)


def load_vector_store(embedding_model, path="data/vector_db/faiss_index"):
    """Load a previously saved FAISS index."""
    return FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
