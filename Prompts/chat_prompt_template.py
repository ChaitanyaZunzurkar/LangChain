from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    # MessagesPlaceholder(variable_name='chat_history')
    ('human', 'Explain in simple terms, what is {topic}')
    # ('human', '{query}')
])

prompts = chat_template.invoke({'domain': 'cricket', 'topic': 'No ball'})

print(prompts)