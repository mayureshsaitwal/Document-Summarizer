import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from core.document_processor import DocumentProcessor
from langchain.chains.summarize.chain import load_summarize_chain


class CustomEmbedding(Embeddings):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def embed_documents(self, texts):
        # Tokenize the texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        # Get embeddings from the model
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)  # Adjust as needed

        # Convert to list of lists (make sure it's a list format)
        embeddings_list = embeddings.cpu().numpy().tolist()  # Convert ndarray to list
        return embeddings_list  # Return as list of embeddings (list of lists)

    def embed_query(self, query):
        # Embedding for a single query
        inputs = self.tokenizer(query, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        
        # Convert to list (single embedding)
        embedding_list = embeddings.cpu().numpy().tolist()  # Convert ndarray to list
        return embedding_list  # Return as list


class RAGSummarizer:
    def __init__(self, google_api_key, llm_model="gemini-2.0-flash"):
        # Load the model and tokenizer manually with transformers
        model_name = "nomic-ai/nomic-embed-text-v2-moe"
        
        # Manually load the model and tokenizer with the trust_remote_code argument
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,  # Enable custom code execution
            device_map="auto"  # Automatically use available device (GPU/CPU)
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create the custom embeddings object
        self.embeddings = CustomEmbedding(self.model, self.tokenizer)

        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model, google_api_key=google_api_key
        )
        self.vector_db = None

    def _initialize_vector_db(self, chunks):
        """Initialize the vector database with document chunks and embeddings."""
        # Pass the chunks and embeddings (which is now a list) to Chroma
        embeddings = self.embeddings.embed_documents(chunks)
        self.vector_db = Chroma.from_texts(chunks, embeddings)

    def summarize(self, document_content, summarization_type="Abstractive"):
        document_processor = DocumentProcessor()
        chunks = document_processor.chunk_document(document_content)

        # Initialize the vector DB with the chunks
        self._initialize_vector_db(chunks)

        if summarization_type == "Extractive":
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_db.as_retriever(),
                return_source_documents=True,
            )
            extractive_prompt = PromptTemplate(
                template="Extract the most important sentences from the following text to form a summary:\n{context}\nSummary:",
                input_variables=["context"],
            )
            response = qa_chain(
                {"query": "Summarize the document.", "prompt": extractive_prompt}
            )
            summary = response["result"]

        elif summarization_type == "Abstractive":
            docs = [Document(page_content=t) for t in chunks]
            summary_chain = load_summarize_chain(self.llm, chain_type="map_reduce")
            summary = summary_chain.run(docs)
        else:
            raise ValueError("Invalid summarization type")

        return summary
