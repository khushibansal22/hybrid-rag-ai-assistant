from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):
    """
    Splits LangChain Document objects into smaller chunks.

    Args:
        documents (list):
            List of LangChain Document objects.

    Returns:
        list:
            List of chunked LangChain Document objects.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(documents)

    return chunks