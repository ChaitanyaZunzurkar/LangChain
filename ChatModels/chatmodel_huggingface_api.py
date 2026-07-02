from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

chat = ChatHuggingFace(llm=llm)

response = chat.invoke("What is the capital of India?")
print(response.content)