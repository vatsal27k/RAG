import os
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-pro"
FAISS_INDEX_DIR = "faiss_index"
DEFAULT_CHUNK_SIZE = 50_000
DEFAULT_CHUNK_OVERLAP = 1_000


def extract_pdf_text(pdf_files: List) -> str:
    """Extract and concatenate text from uploaded PDF files.

    Args:
        pdf_files: List of Streamlit UploadedFile objects (or file-like objects).

    Returns:
        Concatenated text from all pages of all readable PDFs.
    """
    parts: List[str] = []
    for uploaded in pdf_files:
        try:
            reader = PdfReader(uploaded)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                parts.append(page_text)
        except Exception as exc:
            # Skip files that cannot be read but continue processing others
            st.warning(f"Failed to read a PDF file: {getattr(uploaded, 'name', 'unknown')} - {exc}")
            continue

    return "\n".join(parts)


def split_text_into_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks using a recursive splitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def create_and_save_vector_store(chunks: List[str], index_dir: str = FAISS_INDEX_DIR) -> None:
    """Create a FAISS vector store from text chunks and persist it locally."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(index_dir)


def load_vector_store(index_dir: str = FAISS_INDEX_DIR) -> FAISS:
    """Load a persisted FAISS vector store. Raises if not present."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(index_dir, embeddings)


def build_qa_chain(model_name: str = LLM_MODEL, temperature: float = 0.3):
    """Construct the QA chain with a clear prompt template and the chosen model."""
    prompt_template = (
        """
    Answer the question as detailed as possible from the provided context. Provide all available details.
    If the answer is not present in the context, respond exactly: "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def answer_user_question(user_question: str) -> Optional[str]:
    """Retrieve similar docs from the vector store and answer the question.

    Returns the answer string or None on failure.
    """
    try:
        db = load_vector_store()
    except Exception:
        st.error("No vector store found. Please upload PDFs and click 'Submit & Process' first.")
        return None

    docs = db.similarity_search(user_question)
    if not docs:
        st.info("No relevant content found in the uploaded PDFs.")
        return None

    chain = build_qa_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    # Different versions may return different keys; try common ones
    answer = response.get("output_text") or response.get("answer") or ""
    return answer


def main() -> None:
    st.set_page_config(page_title="Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")
    if user_question:
        answer = answer_user_question(user_question)
        if answer:
            st.write("Reply:", answer)

    with st.sidebar:
        st.image("img/Robot.jpg")
        st.write("---")

        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF before processing.")
            else:
                with st.spinner("Processing..."):
                    raw_text = extract_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text could be extracted from the uploaded PDFs.")
                    else:
                        chunks = split_text_into_chunks(raw_text)
                        create_and_save_vector_store(chunks)
                        st.success("Done")

        st.write("---")
        st.image("img/gkj.jpg")
        st.write("AI App created by @ Gurpreet Kaur")

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/gurpreetkaurjethra" target="_blank">Gurpreet Kaur Jethra</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
