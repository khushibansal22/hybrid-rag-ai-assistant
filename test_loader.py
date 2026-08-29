from src.loader.pdf_loader import load_pdf
from src.text_splitter import chunk_documents

pdf_path = "data/uploads/handwrittencircuit.pdf"

documents = load_pdf(pdf_path)

chunks = chunk_documents(documents)

print(f"Original Documents : {len(documents)}")
print(f"Total Chunks       : {len(chunks)}")

print("\n------------------------------------")
print("FIRST CHUNK")
print("------------------------------------")

print(chunks[0].page_content)

print("\n------------------------------------")
print("METADATA")
print("------------------------------------")

print(chunks[0].metadata)