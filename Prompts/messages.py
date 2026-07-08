from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from model import model

messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='tell me about langchain'),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)