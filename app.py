import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_core.documents import BaseDocumentTransformer
from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
from htmlTemplates import bot_template, user_template, css
from transformers import AutoTokenizer
import re
import time

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_pdf_text(pdf_files):
    total_files = len(pdf_files)
    all_text = ""
    status_area = st.empty()
    for i, pdf_file in enumerate(pdf_files):
        reader = PdfReader(pdf_file)
        num_pages = len(reader.pages)
        st.info(f"Processing file: {pdf_file.name} ({i+1}/{total_files})")
        file_text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                file_text += page_text
            progress = (page_num + 1) / num_pages
            status_area.progress(progress, text=f"Processing page {page_num + 1}/{num_pages} of {pdf_file.name}")
        all_text += clean_text(file_text) + " "
    status_area.success("PDFs processed successfully!")
    return all_text.strip()

def get_chunk_text(text):
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100,
    separator = "\n", length_function = len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # For OpenAI Embeddings
    #openai_api_key = st.secrets["OPENAI_API_KEY"]
    # embeddings = OpenAIEmbeddings()
    # For Huggingface Embeddings with instructor model more accuracy, more computational resources, therefore slower
    # embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")

    # Huggingface Embeddings with all-MiniLM-L6-v2 less quality, more velocity
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def truncate_text(text, max_tokens):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return text

class TruncateDocumentsTransformer(BaseDocumentTransformer):
    def transform_documents(self, documents: list[Document]) -> list[Document]:
        truncated_docs = []
        for doc in documents:
            truncated_content = truncate_text(doc.page_content, 500)
            new_doc = Document(page_content=truncated_content, metadata=doc.metadata)
            truncated_docs.append(new_doc)
        return truncated_docs
    

def get_conversation_chain(vector_base):
    # OpenAI Model
    # llm = ChatOpenAI()

    # HuggingFace Model
    llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"temperature": 0.7, "max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vector_base.as_retriever()

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain


def handle_user_input(question):
    if st.session_state.conversation:
        if question:  
            response = st.session_state.conversation({'question': question})
            st.session_state.chat_history = response['chat_history']
            for i, message in enumerate(st.session_state.chat_history):
                template = user_template if i % 2 == 0 else bot_template
                st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.warning("Please enter a question.")
    else:
        st.warning("Please process the PDF files first to start the conversation.")

def main():
    load_dotenv()

    st.set_page_config(page_title='Pdf Chatbot Assistant')
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Let's chat about your Pdf")
    question = st.text_input("Ask me anything you want about your Pdf: ")

    if question:
        handle_user_input(question)

    with st.sidebar:
        st.subheader("Upload your Documents Here: ")
        pdf_files = st.file_uploader("Choose your Pdf Files and Press OK", type=['pdf'], accept_multiple_files=True)

        if st.button("Process Pdf"):
            with st.spinner("Processing your Pdf..."):
                if pdf_files:
                    raw_text = get_pdf_text(pdf_files)
                    text_chunks = get_chunk_text(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    if vector_store: 
                        st.session_state.conversation = get_conversation_chain(vector_store)
                        st.success("Pdf processed and knowledge base created!")
                        if question:
                            try:
                                results = vector_store.similarity_search(question, k=1)
                                st.write("Resultados de búsqueda directa:", results)
                            except Exception as e:
                                st.error(f"Error en búsqueda directa: {e}")
                    else:
                        st.error("Failed to create the knowledge base.")
                else:
                    st.warning("Please upload PDF files first.")

        st.divider()
        with st.expander("Use instructions"):
            st.markdown("1. Upload your Pdf files using the selector on the left.")
            st.markdown("2. Click the 'Process PDFs' button.")
            st.markdown("3. Write your questions in the main text box.")

if __name__ == '__main__':
    main()