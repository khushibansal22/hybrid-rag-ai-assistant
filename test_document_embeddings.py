from src.loader.pdf_loader import load_pdf
from src.text_splitter import chunk_documents
from src.embeddings import embed_documents


pdf_path = "data/uploads/handwrittencircuit.pdf"

documents = load_pdf(pdf_path)

chunks = chunk_documents(documents)

print("Original Documents:", len(documents))
print("Total Chunks:", len(chunks))

embeddings = embed_documents(chunks)

print("Total Embeddings:", len(embeddings))
print("Dimensions of Each Embedding:", len(embeddings[0]))

print("\nFirst Embedding:")
print(embeddings[0][:10])