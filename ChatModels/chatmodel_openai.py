from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(
    model='gpt-4', 
    temprature=0.3,
    max_completion_token=100    
)

result = chat_model.invoke("What is the capital of India ?")

print(result.content)