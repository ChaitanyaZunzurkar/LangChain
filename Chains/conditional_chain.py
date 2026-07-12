from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

class Review(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Return the sentiement of the following review either postive or negative.')

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Review)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback. {format_instructions} \n Feedback:{feedback}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser1),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser1),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({
    'feedback': "Absolutely worth the purchase! The product exceeded my expectations in terms of quality and performance. It's easy to use, well-built, and works exactly as described. I've been using it every day for the past few weeks without any issues. The packaging was excellent, delivery was on time, and the overall experience was fantastic. I would definitely recommend this product to anyone looking for reliability and great value for money."
})

chain.get_graph().print_ascii()
print(result)