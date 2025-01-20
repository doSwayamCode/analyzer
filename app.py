import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY is missing. Please set it in the environment variables.")
else:
    genai.configure(api_key=api_key)


def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text


def get_text_chunks(text):
    """Split text into manageable chunks for processing."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []


def get_vector_store(text_chunks):
    """Create and return a vector store using text chunks."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return FAISS.from_texts(text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None


def get_conversational_chain():
    """Set up the conversational chain using a custom prompt."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say, "Answer is not available in the context." 
    Do not provide a wrong answer.\n\n
    Context:\n {context}\n
    Question: {question}\n
    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        st.error(f"Error setting up conversational chain: {e}")
        return None


def user_input(user_question, vector_store):
    """Handle user questions by retrieving relevant documents and providing answers."""
    try:
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        if chain:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("Reply:", response["output_text"])
        else:
            st.error("Error in generating response.")
    except Exception as e:
        st.error(f"Error handling user input: {e}")


def main():
    """Streamlit app main function."""
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.title("Chat with PDF")
    st.markdown("Upload your PDFs and ask questions based on their content.")

    vector_store = None

    with st.sidebar:
        st.header("📂 Upload PDF File")
        pdf_docs = st.file_uploader(
            "Select your PDF file",
            accept_multiple_files=True,
            type=["pdf"],
        )
        if st.button("Submit PDF"):
            if pdf_docs:
                with st.spinner("Processing PDF..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                            vector_store = get_vector_store(text_chunks)
                            if vector_store:
                                st.success("PDF submitted successfully!")
                        else:
                            st.error("Failed to split text")
                    else:
                        st.error("No text extracted from PDF.")
            else:
                st.error("Please upload at least one PDF")

    user_question = st.text_input("Ask a question about your PDFs:")
    if user_question:
        if vector_store:
            user_input(user_question, vector_store)
        else:
            st.warning("Please submit PDF file before asking questions.")


if __name__ == "__main__":
    main()
