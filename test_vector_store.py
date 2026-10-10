
from src.loader.pdf_loader import load_pdf
from src.text_splitter import chunk_documents
from src.embeddings import get_embedding_model
from src.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store
)


# Step 1: Load the PDF
pdf_path = "data/uploads/handwrittencircuit.pdf"
documents = load_pdf(pdf_path)

# Step 2: Split documents into chunks
chunks = chunk_documents(documents)

# Step 3: Load the embedding model
embedding_model = get_embedding_model()

# Step 4: Create the FAISS vector store
vector_store = create_vector_store(chunks, embedding_model)

print("Original documents:", len(documents))
print("Total chunks:", len(chunks))
print("FAISS vector store created successfully!")


# Step 5: Save the FAISS vector store
save_vector_store(vector_store)
print("FAISS vector store saved successfully!")

# Step 6: Load the saved FAISS vector store
loaded_vector_store = load_vector_store(embedding_model)
print("FAISS vector store loaded successfully!")

# Step 7: Test retrieval using the loaded vector store
query = "What dataset does this paper use?"

results = loaded_vector_store.similarity_search(query, k=3)

print("\nTop 3 retrieved chunks:")

for i, document in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Source:", document.metadata.get("source"))
    print("Page:", document.metadata.get("page"))
    print("Content:", document.page_content[:300])
