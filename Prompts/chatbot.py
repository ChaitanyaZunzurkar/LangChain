from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    
]

while True:
    user_input = input("You : ")
    chat_history.append(user_input)

    if user_input == 'exit':
        break

    result = model.invoke(chat_history)
    chat_history.append(result.content)

    print("AI: " ,result.content)

