from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import streamlit as st
from model import model

load_dotenv()

st.header("Research Tool")

paper_input = st.selectbox("Select Research Paper Name", ["Select...", "Attention is all you need", "BERT: Pre-training Deep Bidirectional Transformers", "GPT-3: Language Models are few-shot learners", "Diffusion models beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explnation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation length", ["Short (1-2 paragrpahs)", "Medium (3-5 paragrpahs)", "large (Detailed Explanation)"])

prompt = PromptTemplate(
    template="""
You are an expert AI researcher and technical educator.

Your task is to explain the following research paper in a clear, accurate, and engaging manner.

Research Paper:
{paper}

Explanation Style:
{style}

Explanation Length:
{length}

Instructions:
- Provide a concise overview of the paper.
- Explain the problem the paper aims to solve.
- Describe the core idea or methodology.
- Explain the model architecture or algorithm if applicable.
- Highlight the key innovations and contributions.
- Mention the main experimental results and findings.
- Discuss real-world applications.
- Mention any important limitations of the paper.
- Use headings and bullet points wherever appropriate.
- Keep the explanation appropriate for the selected style.
- Do not invent information that is not present in the paper.
- Avoid unnecessary repetition.

Answer:
""",
    input_variables=["paper", "style", "length"],
    validate_template=True
)

formatted_prompt = prompt.format(
    paper=paper_input,
    style=style_input,
    length=length_input
)

if st.button("Summarize"):
    result = model.invoke(formatted_prompt)
    output = result.content

    if "<|assistant|>" in output:
        output = output.split("<|assistant|>")[-1].strip()
    
    st.write(output)