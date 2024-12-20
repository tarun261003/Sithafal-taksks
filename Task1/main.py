import streamlit as st
import subprocess
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

load_dotenv()

st.title("RAG Application for Comparison Queries using Gemini Model")

# Radio buttons for single or multiple PDF handling
option = st.radio("Select PDF handling mode:", ("Single PDF", "Multiple PDFs"))

if option == "Single PDF":
    uploaded_file = st.file_uploader("Upload a single PDF file", type="pdf")
    
    if uploaded_file:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())

        # OCR Processing
        ocr_output = f"ocr_{uploaded_file.name}"
        subprocess.run(["ocrmypdf", "--skip-text", uploaded_file.name, ocr_output])

        # Load and split documents
        loader = PyPDFLoader(ocr_output)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)

        # Create vector store with documents
        vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        # Initialize the language model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

        # Define prompts
        comparison_prompt = (
            "You are an assistant designed for answering tasks. "
            "Use the retrieved context to extract and compare relevant information. "
            "Provide the answer in a concise format."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", comparison_prompt),
                ("human", "{input}"),
            ]
        )

        query = st.chat_input("Enter your query: ")

        if query:
            # Create chains for processing and answering
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # Retrieve and process the query
            response = rag_chain.invoke({"input": query})

            # Display the response
            st.write(response["answer"])

elif option == "Multiple PDFs":
    uploaded_files = st.file_uploader("Upload multiple PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        all_docs = []

        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.read())

            # OCR Processing
            ocr_output = f"ocr_{uploaded_file.name}"
            subprocess.run(["ocrmypdf", "--skip-text", uploaded_file.name, ocr_output])

            # Load and split documents
            loader = PyPDFLoader(ocr_output)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            split_docs = text_splitter.split_documents(data)

            # Add PDF name as metadata to each chunk
            for doc in split_docs:
                doc.metadata["source"] = uploaded_file.name

            all_docs.extend(split_docs)

        # Combine documents with metadata
        combined_docs = "\n\n".join([f"Source: {doc.metadata['source']}\n{doc.page_content}" for doc in all_docs])

        # Create vector store with documents
        vectorstore = Chroma.from_documents(documents=all_docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        # Initialize the language model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

        # Define prompts
        comparison_prompt = (
            "You are an assistant designed for comparison tasks. "
            "Use the retrieved context to extract and compare relevant information. "
            "Provide the comparison in a structured format like a table or bullet points. "
            "If relevant terms or fields are not found, indicate that you need more specific information."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", comparison_prompt),
                ("human", "{input}"),
            ]
        )

        query = st.chat_input("Enter a comparison query: ")

        if query:
            # Create chains for processing and answering
            comparison_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, comparison_chain)

            # Retrieve and process the comparison query
            response = rag_chain.invoke({"input": query})

            # Display the structured comparison response
            st.write(response["answer"])

