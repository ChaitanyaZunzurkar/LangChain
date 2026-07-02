from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
    model='text-embedding-3-large',    
    dimensions=32
)

documents = [
    "Delhi is the capital of India",
    "My name is Chaitanya Zunzurkar",
    "Paris is the capital of France"
]

result = embedding.embed_query("What is the capital of India ?")
document_result = embedding.embed_documents(documents)

print(str(document_result))
print(str(result))