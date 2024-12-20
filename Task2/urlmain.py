from langchain_community.document_loaders import UnstructuredURLLoader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

load_dotenv()
# Streamlit application title
st.title("RAG Application built on Gemini Model")

# Prompt user to enter URLs
user_url_input = st.text_input("Enter a URL (e.g., https://example.com):", "https://www.washington.edu/")
urls = user_url_input.split(',')

# Load documents from the user-provided URL
if urls:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Create a vector store for document retrieval
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )

    # Configure retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Load the language model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None
    )

    # User query input
    query = st.chat_input("Ask your question:")

    if query:
        # System prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Create the RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        with st.spinner("Processing..."):
            response = rag_chain.invoke({"input": query})


        # Display the answer
        st.write(response["answer"])