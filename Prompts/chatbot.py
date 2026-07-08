from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

while True:
    user_input = input("You : ")

    if user_input == 'exit':
        break

    result = model.invoke(user_input)
    print("AI: " ,result.content)

