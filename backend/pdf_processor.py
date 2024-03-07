from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
load_dotenv()  # This loads the environment variables from a .env file

# Global variable to hold the initialized QA chain
qa_chain = None
qa_chain_sgg=None
# Instead of hardcoding the keys, use environment variables
cohere_api_key = os.getenv('COHERE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

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
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    vectorstore_faiss = FAISS.from_documents(docs, embeddings)

    # Initialize the language model
    llm = OpenAI(api_key=openai_api_key)

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
    template="""Text: {context}\n\nQuestion: {question}\n\nBased on the context and the question, suggest three related questions that might be of interest as next questions.keep the questions short and show them in separate lines""",
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
    # source_directory = document.metadata['source']  # Extracting the 'source' from metadat
    original_path = document.metadata['source']  # This is 'pdf_directory/YourDocument.pdf'
    file_name = os.path.basename(original_path)  # This extracts 'YourDocument.pdf' from the path

    # Now construct the new path with correct base directory and original file name
    source_directory = os.path.join('pdfs', file_name)
    page_number = document.metadata['page']
    combined_info=f"{source_directory}"
    docs_text = ""
    with fitz.open(source_directory) as doc_sum1:  # Ensure 'fitz' is imported and working correctly
        for page in doc_sum1:
            docs_text += page.get_text()

    context_s = docs_text[:3000]
    combined_input = f"Document: {context_s}\ Create a very very short summary of the text."
    llm = OpenAI(api_key=openai_api_key)
    doc_summary = llm.invoke(combined_input)


    # next suggestions
    answer_sgg = qa_chain_sgg({"query": query})
    result_sgg = answer_sgg["result"].replace("\n", "").replace("Answer:", "").strip()


    return result,result_sgg, combined_info,doc_summary







def process_uploaded_document_and_ask(file_path, question):
    # Initialize your OpenAI LLM with your API key
    llm = OpenAI(api_key=openai_api_key)
    # Extract text from the uploaded PDF document
    doc_text = ""
    with fitz.open(file_path) as doc:  # Ensure 'fitz' is imported and working correctly
        for page in doc:
            doc_text += page.get_text()

    # Ensure the document text is appropriately truncated or processed for your use case

    context = doc_text[:3000]  # Adjust as needed based on your model's limitations

    # Combine the extracted text (context) and the question
    combined_input = f"Document: {context}\nQuestion: {question}\n\nAnswer the question based on the document provided. If nothing found say I could not find anything related."

    # Send the combined input to the LLM and get the response
    response = llm.invoke(combined_input)  # Adjust this if your method of invoking the LLM differs

    # Assuming the response is structured with 'choices', extract the text from the response
    # This part needs to match the structure of your LLM's response
    answer = response.strip()

    return answer

