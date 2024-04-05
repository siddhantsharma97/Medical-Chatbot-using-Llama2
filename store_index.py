from src.helper import load_pdf, text_split , download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

extracted_data=load_pdf("data/")
text_chunks=text_split(extracted_data)
embeddings=download_hugging_face_embeddings()

index_name="test"

#Creating Embeddings for Each of The Text Chunks & storing
docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)