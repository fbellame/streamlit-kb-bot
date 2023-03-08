from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os


def embed_doc(directory_path):
    if len(os.listdir(directory_path)) > 0:
        loader = DirectoryLoader(directory_path, glob="**/*.*")

        raw_documents = loader.load()
        print(len(raw_documents))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=0, length_function= len)
        documents = text_splitter.split_documents(raw_documents)

        # Load Data to vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Save vectorstore
        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)

if os.path.exists("vectorstore.pkl"):
    with open("vectorstore.pkl", "rb") as f:
        docsearch = pickle.load(f)