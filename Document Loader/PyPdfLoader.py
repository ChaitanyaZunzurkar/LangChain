from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Give me 10 most important question from the given pdf : \n {pdf}',
    input_variables=['pdf']
)

parser = StrOutputParser()

loader = PyPDFLoader('OS_Full_Notes.pdf')

docs = loader.load()

# print(type(docs)) # List class 
# print(docs) # list of document object
# print(docs[0]) # way to access the list of documents 
# print(type(docs[0])) # type of element in the list ==> document object

chain = prompt | model | parser

result = chain.invoke({
    'pdf': docs[0].page_content
})

print(result)

