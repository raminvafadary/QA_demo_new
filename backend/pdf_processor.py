from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
# Global variable to hold the initialized QA chain
qa_chain = None
qa_chain_sgg=None
def initialize_qa_system(pdf_directory):
    global qa_chain
    global qa_chain_sgg

    # Load PDF documents
    loader = PyPDFDirectoryLoader(pdf_directory)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Generate embeddings for document chunks
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key="3DJoDZKyuT3isbrDu4Les4ydIcZ29KnWi5Mpg9kj")
    vectorstore_faiss = FAISS.from_documents(docs, embeddings)

    # Initialize the language model
    llm = OpenAI(api_key="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")

    # Create a custom prompt template
    prompt_template_qa = PromptTemplate(
    template="""Text: {context}\n\nQuestion: {question}\n\nAnswer the question based on the text provided. If the text doesn't contain the answer, reply that the answer is not available.\n\n""",
    input_variables=["context", "question"]
    )
    # Set up the question-answering chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template_qa},
        return_source_documents=True
    )
    # Create a custom prompt template
    prompt_template_sgg = PromptTemplate(
    template="""Text: {context}\n\nQuestion: {question}\n\nBased on the context and the question, suggest three related questions that might be of interest as next questions.""",
    input_variables=["context", "question"]
    )
    # Set up the question-answering chain
    qa_chain_sgg = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template_sgg},
        return_source_documents=True
    )

def ask_question(query):
    if qa_chain is None:
        return "QA system is not initialized.", None
    answer = qa_chain({"query": query})
    result = answer["result"].replace("\n", "").replace("Answer:", "").strip()
    # citation
    document = answer['source_documents'][-1]  # This is the last document in the list
    source_directory = document.metadata['source']  # Extracting the 'source' from metadat
    page_number = document.metadata['page']
    combined_info = f"{source_directory}, Page {page_number}"

    # next suggestions
    answer_sgg = qa_chain_sgg({"query": query})
    result_sgg = answer_sgg["result"].replace("\n", "").replace("Answer:", "").strip()


    return result,result_sgg, combined_info
