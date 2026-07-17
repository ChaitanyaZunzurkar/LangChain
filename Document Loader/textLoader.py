from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Write a summary for the following poem : \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('cricket.txt', encoding='utf8')

docs = loader.load()

# print(type(docs)) # List class 
# print(docs) # list of document object
# print(docs[0]) # way to access the list of documents 
# print(type(docs[0])) # type of element in the list ==> document object

chain = prompt | model | parser

result = chain.invoke({
    'poem': docs[0].page_content
})

print(result)

