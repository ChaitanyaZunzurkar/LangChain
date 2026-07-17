from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_Variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short questions from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    notes=prompt1 | model | parser,
    quiz=prompt2 | model | parser
)

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

result = chain.invoke({
    'text': "Support Vector Machine (SVM) is a supervised machine learning algorithm used for both classification and regression tasks, though it is most commonly applied to classification problems. The primary objective of SVM is to identify the optimal hyperplane that separates data points of different classes with the maximum possible margin, thereby improving the model's generalization performance. Data points that lie closest to the decision boundary, known as support vectors, play a crucial role in defining the hyperplane. For datasets that are not linearly separable, SVM employs the kernel trick, which transforms the input data into a higher-dimensional feature space using kernels such as linear, polynomial, radial basis function (RBF), or sigmoid, enabling effective separation of complex patterns. SVM is particularly effective in high-dimensional spaces, works well with small- to medium-sized datasets, and is widely used in applications such as image classification, text categorization, bioinformatics, and medical diagnosis due to its high accuracy and robustness against overfitting."
})

print(result)

chain.get_graph().print_ascii()