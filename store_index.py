from src.helper import load_pdf,text_split,download_hugging_face_embeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
index_name = "medical-chatbot"

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pinecone = PineconeVectorStore(embedding=embeddings,pinecone_api_key=PINECONE_API_KEY, index_name=index_name)
docsearch = pinecone.from_documents(text_chunks,embedding=embeddings,index_name=index_name)
