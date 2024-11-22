import streamlit as st
import random
import json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
# from langchain.chains import (
#     create_history_aware_retriever,
#     create_retrieval_chain,
# )
from langchain.chains import ConversationalRetrievalChain

from openai import OpenAI

from htmlTemplates import css, bot_template, user_template


# client = OpenAI()

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # conversation_chain = create_history_aware_retriever(llm, retriever=vector_store.as_retriever, memory=memory)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vector_store.as_retriever(), memory=memory)
    
    return conversation_chain

# def generate_questions(vector_store, num_questions=5):
#     retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
#     relevant_docs = retriever.get_relevant_documents("Resumen del documento para preguntas.")
#     context = " ".join([doc.page_content for doc in relevant_docs])

#     prompt = f"""
#     Basándote en el siguiente contexto:
#     {context}
    
#     Genera {num_questions} preguntas sobre el contenido proporcionado. Formatea cada pregunta de manera clara y precisa.
#     """
#     try:
#         llm = ChatOpenAI(model="gpt-4", temperature=0.7)
#         response = llm(prompt, max_tokens=300)
#         questions = response["choices"][0]["text"].strip().split("\n")
        
#         return {f"question_{i+1}": q for i, q in enumerate(questions)}
#     except Exception as e:
#         st.error(f"Error al generar preguntas: {e}")
        
#         return {}




def handle_user_input(user_question):
    resp = st.session_state.conversation({"question": user_question})

    st.session_state.chat_history = resp['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():

    load_dotenv()

    st.set_page_config(page_title='Chat with multiple PDFs')
    st.write(css, unsafe_allow_html=True)
    st.markdown("Sube un documento PDF para interactuar con el contenido y generar preguntas.")

    # Initialize the session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # if "questions_json" not in st.session_state:
    #     st.session_state.questions_json = None


    # Upload documents
    pdf_docs = st.file_uploader("Sube tus PDFs aquí", accept_multiple_files=True)
    if st.button("Procesar Documento"):
        if pdf_docs:
            with st.spinner("Procesando documentos..."):
                # Obtener texto del PDF
                raw_text = get_pdf_text(pdf_docs)

                # Dividir texto en chunks
                text_chunks = get_text_chunks(raw_text)

                # Crear vector store
                vector_store = get_vector_store(text_chunks)

                # Crear cadena de conversación
                st.session_state.conversation = get_conversation_chain(vector_store)

                # Generar preguntas
                # st.session_state.questions_json = generate_questions(vector_store, num_questions=5)
        else:
            st.error("Por favor, sube al menos un documento en formato PDF.")

    # Display generated questions
    # if st.session_state.questions_json:
    #     st.subheader("Preguntas Generadas")
    #     for key, question in st.session_state.questions_json.items():
    #         st.markdown(f"**{key}:** {question}")

    #     # Save questions in JSON format
    #     with open("questions.json", "w") as json_file:
    #         json.dump(st.session_state.questions_json, json_file, indent=4)
    
    # Chatbot
    if st.session_state.conversation:
        st.subheader("Interacción con el Bot")
        user_question = st.text_input("Haz una pregunta sobre el documento:")
        if user_question:
            handle_user_input(user_question)



if __name__ == '__main__':
    main()