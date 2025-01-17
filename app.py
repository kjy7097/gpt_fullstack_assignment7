import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.storage import LocalFileStore
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler

import os

os.environ['OPENAI_API_KEY'] = 'dummy_api_key'

st.set_page_config(page_title="Advanced Quiz GPT", page_icon="📖")
st.title("Welcome! Advanced Quiz GPT")

with st.sidebar:
    st.page_link(page="https://github.com/kjy7097/gpt_fullstack_assignment7.git",label="Click! Go to Github Repo.")
    api_key = st.text_input("Enter OpenAI API Key....")
    if api_key:
        st.selectbox("Difficulty level", ["Beginner", "Intermediate", "Advanced"], placeholder="Choose the difficulty level for the quiz.")
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"]
        )

if api_key:
    st.markdown(
        """
                Please upload a document to generate a quiz.

                You can adjust the difficulty level in sidebar"""
    )
    os.environ['OPENAI_API_KEY'] = api_key
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
else:
    st.markdown(
        """
                Please enter API key first.
        """
    )



def map_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])



def invoke_chain(message, input_chain):
    result = input_chain.invoke(message)
    st.session_state["memory"].save_context(
        {"input": message}, {"output": result.content}
    )
    return result.content


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up,
            --------
            {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# if api_key:
    # if file:
        # retriever = embed_file(file)
        # send_message("I am ready to answer!", "ai", save=False)
        # chain = (
        #     {
        #         "context": retriever | RunnableLambda(map_docs),
        #         "question": RunnablePassthrough(),
        #     }
        #     | RunnablePassthrough.assign(history=load_memory)
        #     | prompt
        #     | llm
        # )
        # message = st.chat_input("Ask anything about the book...")
        # if message:
        #     if len(st.session_state["history"]) > 0:
        #         print_history()
        #     send_message(message, "Human")
        #     with st.chat_message("ai"):
        #         invoke_chain(message, chain)
    # else:
    #     st.session_state["history"] = []
