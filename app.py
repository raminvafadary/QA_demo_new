import streamlit as st
import sys
import os
import tempfile
import re

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.pdf_processor import initialize_qa_system, ask_question, process_uploaded_document_and_ask

# Initialize the QA system for internal documents
initialize_qa_system('./pdfs')  # Assuming this is the path to your internal documents

def main():
    st.title('Question Answering System')
       # Display logos
    col1, col2 = st.columns(2)
    with col1:
        st.image('./logo2.png')  # Adjust path as necessary

    # UI for internal document based QA
    st.header('Internal Document Based QA')
    query_internal = st.text_input("Enter your question for the internal documents:")

    if st.button('Submit for Internal Documents'):
        if query_internal:
            answer_internal, answer_sgg_internal, citation_internal, doc_summary_internal = ask_question(query_internal)
            if answer_internal:
                st.subheader('Answer from Internal Documents')
                st.write(answer_internal)
                st.subheader('Document Summary for Internal Documents')  # Added part
                st.write(doc_summary_internal)  # Display the document summary for internal documents
                st.subheader('Citation')
                st.write(citation_internal)
                st.subheader('Suggested Next Questions for Internal Documents')
                cleaned_questions = re.sub(r'^\.\s*', '', answer_sgg_internal)
                question_list = re.split(r'\d+\.', cleaned_questions)
                questions_internal = [question.strip() for question in question_list if question.strip()]
                for question in questions_internal:
                    st.write(question)
            else:
                st.write('Sorry, no answer found from internal documents.')
        else:
            st.write('Please enter a question for internal documents.')

    # Divider
    st.markdown("---")

    # UI for user-uploaded document based QA
    st.header('Uploaded Document Based QA')
    uploaded_file = st.file_uploader("Upload a PDF document for your questions:", type=["pdf"])
    query_uploaded = st.text_input("Enter your question for the uploaded document:")

    if uploaded_file is not None and st.button('Submit for Uploaded Document'):
        # Process the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            processed_file_path = tmp.name  # Path to the temporary saved file
            # Get the answer for the uploaded document directly
            answer_uploaded = process_uploaded_document_and_ask(processed_file_path, query_uploaded)

        if query_uploaded:
            if answer_uploaded:  # Just use the answer from the uploaded document processing
                st.subheader('Answer from Uploaded Document')
                st.write(answer_uploaded)
                # If you had citation and suggested questions for the uploaded docs, add them here
            else:
                st.write('Sorry, no answer found from uploaded document.')
        else:
            st.write('Please enter a question for the uploaded document.')

if __name__ == '__main__':
    main()
