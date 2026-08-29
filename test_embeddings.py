from src.embeddings import get_embedding_model


embedding_model = get_embedding_model()

text = "Attention mechanisms allow a model to focus on relevant information."

vector = embedding_model.embed_query(text)

print("Vector type:", type(vector))
print("Vector dimensions:", len(vector))
print("First 10 values:", vector[:10])