from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

Review = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_theams": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "description": "Write down all the key themes discussed in the review in a list"
        }, 
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "Return a sentiment of the review either negative, positive or neutral"
        },
    }
}


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(result)