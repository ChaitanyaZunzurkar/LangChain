from langchain_text_splitter import CharacterTextSplitter, RecursiveCharacterSplitter

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

text = "If you crack Nutanix, we can have a very detailed discussion about whether accepting a 22 LPA SRE role is better than taking the risk of waiting for Juspay, DE Shaw, Porter, or Nvidia SDE. At that point, we'll be comparing a real offer against real upcoming opportunities—not making the decision based on assumptions. That means you are competitive, but I wouldn't say you're guaranteed to crack companies like Nvidia or DE Shaw. If Nutanix SRE (22 LPA) is one of the strongest opportunities your campus is likely to offer this season, then I would treat it very differently than if Microsoft or Atlassian SDE were guaranteed to come." 

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

length_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

recursive_splitter = RecursiveCharacterSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

semantic_splitter = SemanticChunker(
    embedding_model,
    breakpoint_type="standard_deviation",
    breakpoint_threshold_amount=2
)

result = length_splitter.split_text(text)
# splitter.split_document(docs) if using document 

recursive_result = recursive_splitter.split_text(text)

semantic_result = semantic_splitter.split_text(text)

print(result)
print(recursive_result)
print(semantic_result)