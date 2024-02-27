import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from backend.pdf_processor import initialize_qa_system, ask_question  # Ensure this path is correctly pointing to your modules

# Initialize the QA system (replace 'path_to_your_pdfs' with the actual path to your PDFs)
initialize_qa_system('./pdfs')




def main():
    st.title('Question Answering System')
    query = st.text_input("Enter your question here:")

    if st.button('Submit'):
        if query:  # Check if there is a query input
            answer, answer_sgg, citation = ask_question(query)
            if answer:
                st.subheader('Answer')
                st.write(answer)
                st.subheader('Citation')
                st.write(citation)
                st.subheader('Suggested Next Questions')
                # Assuming 'answer_sgg' contains questions separated by a new line
                questions = answer_sgg.split('\n')  # Split by new line or change '\n' to the appropriate separator
                for question in questions:
                    st.write(question)  # This will display each question on a new line
            else:
                st.write('Sorry, no answer found.')
        else:
            st.write('Please enter a question.')

if __name__ == '__main__':
    main()

