from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field
from typing import Literal, Optional

class Review(BaseModel):
    key_theams: list[str] = Field(description='tell the key theams from the review.')

    summary: str = Field(description='A brief summary of the review.')

    sentiment: Literal["pos", "neg"] = Field(description='Return the sentiment of the review either negative, positive or neutral')

    pros: Optional[list[str]] = Field(default=None, description='Write down all the pros inside the list')

    cons: Optional[list[str]] = Field(default=None, description='Write down all the cons inside the list')


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(result)