import streamlit as st
from dotenv import load_dotenv
import os
from core.document_processor import DocumentProcessor
from core.rag_summarizer import RAGSummarizer
from core.translation_manager import TranslationManager

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def main():
    st.set_page_config(page_title="Multilingual Document Summarizer", layout="wide")
    st.title("Multilingual Document Summarizer (RAG)")

    doc_processor = DocumentProcessor()
    rag_summarizer = RAGSummarizer(google_api_key=GOOGLE_API_KEY)
    translation_manager = TranslationManager(google_api_key=GOOGLE_API_KEY)

    st.subheader("Upload Document or Paste Text")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"]
    )
    pasted_text = st.text_area("Or paste text here", height=200)

    summarization_type = st.radio("Summarization Type", ("Extractive", "Abstractive"))

    target_language = st.selectbox(
        "Translate Summary", ["None", "English", "Hindi", "Tamil", "Marathi"]
    )

    if st.button("Summarize"):
        document_content = ""
        if uploaded_file:
            with st.spinner("Processing document..."):
                document_content = doc_processor.load_document(uploaded_file)
        elif pasted_text:
            document_content = pasted_text

        if document_content:
            with st.spinner("Generating summary..."):
                summary = rag_summarizer.summarize(document_content, summarization_type)

            st.subheader("Original Document")
            st.text_area("Original Text", document_content, height=300)

            st.subheader("Generated Summary")
            st.write(summary)

            if target_language != "None":
                with st.spinner(f"Translating summary to {target_language}..."):
                    translated_summary = translation_manager.translate(
                        summary, target_language
                    )
                    st.subheader(f"Translated Summary ({target_language})")
                    st.write(translated_summary)
        else:
            st.warning("Please upload a document or paste text to summarize.")


if __name__ == "__main__":
    main()
