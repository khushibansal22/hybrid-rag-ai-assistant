from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "BAAI/bge-base-en-v1.5"


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME
    )


def embed_documents(documents):
    embedding_model = get_embedding_model()

    texts = [document.page_content for document in documents]

    embeddings = embedding_model.embed_documents(texts)

    return embeddings