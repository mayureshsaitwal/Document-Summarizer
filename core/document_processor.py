import os
import io
import docx
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect_langs, DetectorFactory
import streamlit as st


class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        DetectorFactory.seed = 0

    def load_document(self, uploaded_file):
        """
        Loads content from an uploaded file (PDF, DOCX, or TXT) and returns it as a string.
        """
        document_content = ""
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension == ".pdf":
            document_content = self._extract_text_from_pdf(uploaded_file)
        elif file_extension == ".docx":
            document_content = self._extract_text_from_docx(uploaded_file)
        elif file_extension == ".txt":
            document_content = uploaded_file.getvalue().decode("utf-8")
        else:
            raise ValueError("Unsupported file type")
        
        return document_content

    def _extract_text_from_pdf(self, uploaded_file):
        """
        Extracts text content from a PDF file.
        """
        text = ""
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def _extract_text_from_docx(self, uploaded_file):
        """
        Extracts text content from a DOCX file.
        """
        doc = docx.Document(uploaded_file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)

    def detect_document_language(self, text):
        """
        Detects the dominant language of the input text using `langdetect`.
        Returns the ISO 639-1 code of the detected language.
        """
        try:
            languages = detect_langs(text)
            return languages[0].lang
        except Exception as e:
            st.warning(f"Could not detect language: {e}. Defaulting to English.")
            return "en" #Default to English if detection fails or is unreliable

    def chunk_document(self, document_content):
        """
        Splits the document content into smaller chunks using the initialized text splitter.
        """
        chunks = self.text_splitter.split_text(document_content)
        return chunks
