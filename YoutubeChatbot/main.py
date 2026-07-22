from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from model import model

url = 'https://youtu.be/5Y5TktzkeTI?si=CUUutWH1mu6w-jDe'
loader = YoutubeLoader.from_youtube_url(
    url,
    add_video_info=False
)

transcript = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10
)

split_docs = splitter.split_documents(transcript)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    collection_name="youtube_transcripts",
    persist_directory='./chroma_db'
)

retriever = vectordb.as_retriever(
    search_type='similarity',
    search_kwargs={"k": 5}
)

query = "What is taught in this video ?"

docs = retriever.invoke(query)

context = ""

for doc in docs:
    context += doc.page_content + "\n\n"

prompt = PromptTemplate(
    template="""
        You are an assistant.
        Answer only using the provided context.

        If answer is not present in the provided context, just say you don't know.

        Context: {context}
        Question: {question}
    """,
    input_variables=['context', "question"],
)

formatted_prompt = prompt.format(
    context=context,
    question=query
)

result = model.invoke(formatted_prompt)

print(result)
print(result.content)
