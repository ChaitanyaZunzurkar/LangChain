import os
from pathlib import Path 
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("API key not found.")

client = Groq(api_key=api_key)

model = 'llama-3.3-70b-versatile'
role="user"
prompt="Do you know MSD ?"
message={
    "role":role,
    "content":prompt
}

messages=[message]

response = client.chat.completions.create(model=model, messages=messages)
print(response.choices[0].message.content)