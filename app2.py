import streamlit as st
import random
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
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


client = OpenAI()

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
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # conversation_chain = create_history_aware_retriever(llm, retriever=vector_store.as_retriever, memory=memory)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vector_store.as_retriever(), memory=memory)
    
    return conversation_chain


def handle_user_input(user_question):
    resp = st.session_state.conversation({"question": user_question})

    st.session_state.chat_history = resp['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# def generate_questions(text, level="medium", num_questions=3):
#     prompt = f"""
#         Basándote en el siguiente texto del documento {text}

#         Genera {num_questions} preguntas de opción múltiple de nivel {level}.
#         Para cada pregunta, incluye 4 opciones y especifica cuál es la respuesta correcta.
#         """
    
#     # prompt = f"""
#     #     Explícame que es el AI-102.
#     #     """
    
#     try:
#         completion = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "Eres un asistente amable. Responde la consulta utilizando únicamente las fuentes que se proporcionan a continuación de manera amistosa y concisa. Responde ÚNICAMENTE con los datos que se enumeran en la lista de fuentes que aparece a continuación. Si no hay suficiente información a continuación, indica que no sabes. No generes respuestas que no utilicen las fuentes que se indican a continuación."},
#                 {
#                     "role": "user",
#                     "content": f"{prompt}"
#                 }
#             ],
#             temperature=0,
#             max_tokens=500
#         )

#         # print(prompt)
#         return completion.choices[0].message.content.split("\n\n")
#     except Exception as e:
#         st.error(f"Error al generar preguntas: {e}")
#         return []





def main():

    load_dotenv()

    st.set_page_config(page_title='Chat with multiple PDFs', page_icon=':books:')
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat With Multiple PDFs :books:')
    user_question = st.text_input('Ask a question about your document:')

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # Create vector store
                vector_store = get_vector_store(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)


    # st.set_page_config(page_title="Generador de Preguntas", page_icon=":books:")
    # st.title("Generador de Preguntas de Opción Múltiple")
    # st.markdown("Sube un documento en formato PDF para generar preguntas basadas en su contenido.")

    # if "questions" not in st.session_state:
    #     st.session_state.questions = None

    # # Subida y procesamiento de documentos
    # pdf_docs = st.file_uploader("Sube tus PDFs aquí", accept_multiple_files=True)
    # if st.button("Procesar Documento"):
    #     if pdf_docs:
    #         with st.spinner("Procesando documentos..."):
    #             # Leer el texto de los documentos
    #             raw_text = get_pdf_text(pdf_docs)

    #             # Dividir el texto en chunks (puede ser útil para documentos grandes)
    #             text_chunks = get_text_chunks(raw_text)

    #             # Combinar los chunks para generar preguntas basadas en todo el contenido
    #             combined_text = " ".join(text_chunks)

    #             # Generar preguntas
    #             st.session_state.questions = generate_questions(combined_text, level="medium")
                
    #     else:
    #         st.error("Por favor, sube al menos un documento en formato PDF.")

    # # Mostrar las preguntas generadas
    # if st.session_state.questions:
    #     st.subheader("Preguntas Generadas")
    #     for i, question in enumerate(st.session_state.questions, start=1):
    #         st.markdown(f"**Pregunta {i}:** {question}")
    


if __name__ == '__main__':
    main()