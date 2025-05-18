import os
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool as lc_create_retriever_tool
from langchain_core.tools.simple import Tool
from langchain.docstore.document import Document
from verifia.utils.helpers.io import find_pdf_file_paths
from verifia.utils import CHECKPOINTS_DIRPATH


def _split_documents(
    docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100
) -> List[Document]:
    """
    Split a list of documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        docs (List[Document]): List of Document objects to split.
        chunk_size (int, optional): Maximum size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): Number of overlapping characters between chunks. Defaults to 100.

    Returns:
        List[Document]: List of split Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splitted_docs: List[Document] = []
    for doc in docs:
        splitted_docs.extend(text_splitter.split_documents([doc]))
    return splitted_docs


def _get_persist_dir(persist: bool) -> Optional[str]:
    """
    Determine the persistence directory if persistence is enabled.

    Args:
        persist (bool): Whether to persist the vector store.

    Returns:
        Optional[str]: The persist directory path (or None if not persisting).
    """
    if persist:
        persist_dir = os.path.join(CHECKPOINTS_DIRPATH, "embeddings")
        os.makedirs(persist_dir, exist_ok=True)
    else:
        persist_dir = None
    return persist_dir


def create_chroma_vectorstore_from_pdfs(
    pdf_folder_path: os.PathLike, persist: bool = False
) -> Chroma:
    """
    Load PDF documents from a folder, split them into chunks, and create a Chroma vector store.

    Args:
        pdf_folder_path (os.PathLike): The folder path containing PDF files.
        persist (bool, optional): Whether to persist the vector store on disk. Defaults to False.

    Returns:
        Chroma: A Chroma vector store built from the document chunks.
    """
    pdf_file_paths: List[str] = find_pdf_file_paths(pdf_folder_path)

    all_docs: List[Document] = []
    for pdf_file_path in pdf_file_paths:
        loader = PyPDFLoader(pdf_file_path)
        pdf_docs = loader.load()
        all_docs.extend(pdf_docs)

    splitted_docs: List[Document] = _split_documents(all_docs)

    embeddings = OpenAIEmbeddings()

    persist_dir = _get_persist_dir(persist)

    vectordb = Chroma.from_documents(
        documents=splitted_docs, embedding=embeddings, persist_directory=persist_dir
    )
    return vectordb


def create_chroma_vectorstore_from_string(
    content: str, persist: bool = False
) -> Chroma:
    """
    Convert a long string into a document, split it into chunks, and create a Chroma vector store.

    Args:
        content (str): A long string containing the document content.
        persist (bool, optional): Whether to persist the vector store on disk. Defaults to False.

    Returns:
        Chroma: A Chroma vector store built from the document chunks.
    """
    doc = Document(page_content=content, metadata={"source": "string_input"})

    splitted_docs: List[Document] = _split_documents([doc])

    embeddings = OpenAIEmbeddings()

    persist_dir = _get_persist_dir(persist)

    vectordb = Chroma.from_documents(
        documents=splitted_docs, embedding=embeddings, persist_directory=persist_dir
    )
    return vectordb


def create_pdf_retriever_tool(vectordb: Chroma) -> Tool:
    """
    Create a retriever tool for PDFs using a Chroma vector store.

    Args:
        vectordb (Chroma): The Chroma vector store containing document embeddings.

    Returns:
        Tool: A tool that wraps the retriever for PDF text retrieval.
    """
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    retriever_tool = lc_create_retriever_tool(
        retriever, "retrieve_pdf_files", "Search and return relevant PDF text"
    )
    return retriever_tool
