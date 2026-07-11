from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me 3 facts about {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
) 

# chain = template | model | parser 

prompt = template.invoke({
    'topic': 'black hole',
})

result = model.invoke(prompt)

final_result = parser.parse(result.content)
print(final_result)

# result = chain.invoke({})

# print(result.content)