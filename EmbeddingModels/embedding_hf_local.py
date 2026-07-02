import os

os.environ["HF_HOME"] = "D:/huggingface_cache"

from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace

embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

documents = [
    "Delhi is the capital of India",
    "My name is Chaitanya Zunzurkar",
    "Paris is the capital of France"
]

text = "Delhi is the capital of India."

result = embedding.embed_query(text)

document_result = embedding.embed_documents(documents)

print(str(document_result))
print(str(result))