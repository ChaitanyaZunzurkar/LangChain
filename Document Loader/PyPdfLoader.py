from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader, WebBaseLoader, SeleniumURLLoader
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

loader = PyPDFLoader('OS_Full_Notes.pdf') # just works with the textual content pdf. It does not work with scanned image pdf

docs = loader.load() # use this when working with small number of pdfs
# use laze_load when working with multiple pdfs. 
# lazy_load loads pdf one by one and perform operation one by one and get removed from the memory

# print(type(docs)) # List class 
# print(docs) # list of document object
# print(docs[0]) # way to access the list of documents 
# print(type(docs[0])) # type of element in the list ==> document object

chain = prompt | model | parser

result = chain.invoke({
    'pdf': docs[0].page_content
})

print(result)

