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

import json

import os

os.environ["OPENAI_API_KEY"] = "dummy_api_key"

st.set_page_config(page_title="Advanced Quiz GPT", page_icon="📖")
st.title("Welcome! Advanced Quiz GPT")
function = {
    "name": "gen_quiz",
    "description": "this function convert array of quiz to json format",
    "parameters": {
        "type": "object",
        "properties": {
            "problems": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                        "description": "Answers of the question",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                        "description": "True if the answer is correct",
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


with st.sidebar:
    st.page_link(
        page="https://github.com/kjy7097/gpt_fullstack_assignment7.git",
        label="Click! to go Github Repo.",
    )
    api_key = st.text_input(
        "Enter OpenAI API Key....",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        difficulty = st.selectbox(
            "Difficulty level",
            ["Beginner", "Intermediate", "Advanced"],
            placeholder="Choose the difficulty level for the quiz.",
        )
        num_quiz = st.number_input("The number of quiz", 0, 20, step=1, value=5)
        user_req = st.text_area(
            "Additional requirements",
            placeholder="""Enter anything you want....
Ex) Please generate the quiz in Korean
            """,
        )
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"]
        )


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs["context"]])


@st.cache_data(show_spinner="Loading file....")
def load_file(file):
    file_content = file.read()
    file_path = f".cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()
    return docs


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
).bind(function_call="auto", functions=[function])


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(
    num_quiz,
    difficulty,
    user_req,
    _docs,
    topic,
):
    os.environ["OPENAI_API_KEY"] = api_key
    chain = (
        {
            "num_quiz": lambda input: input["num_quiz"],
            "difficulty": lambda input: input["difficulty"],
            "requirements": lambda input: input["requirements"],
            "context": format_docs,
        }
        | prompt
        | llm
    )
    return chain.invoke(
        {
            "num_quiz": str(num_quiz),
            "difficulty": difficulty,
            "requirements": user_req,
            "context": _docs,
        }
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.

            Based ONLY on the following context make {num_quiz} questions to test the user's knowledge about the text.

            The diffculty level of the quiz is {difficulty}

            Each question should have 4 answers, three of them must be incorrect and one should be correct.

            Use (o) to signal the correct answer.

            Question examples:

            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(o)

            Question: What is the capital or Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut

            Question: When was Avatar released?
            Answers: 2007|2001|2009(o)|1998

            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o)|Painter|Actor|Model

            You will generate all quiz as json format

            Additional requirements : {requirements}

            Your turn!

            Context: {context}
            """,
        ),
    ]
)


def get_num_quiz(input):
    return input["num_quiz"]


if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    if file:
        docs = load_file(file)
        response = run_quiz_chain(
            num_quiz=num_quiz,
            difficulty=difficulty,
            user_req=user_req,
            _docs=docs,
            topic=file.name,
        )
        with st.form("questions form"):
            quiz_json = json.loads(
                response.additional_kwargs["function_call"]["arguments"]
            )
            correct_count = 0
            for problem in quiz_json["problems"]:
                st.write(problem["question"])
                value = st.radio(
                    "Select an option",
                    [
                        answer["answer"].replace("(o)", "")
                        for answer in problem["answers"]
                    ],
                    index=None,
                )
                if value is not None:
                    value = value + "(o)"
                if {"answer": value, "correct": True} in problem["answers"]:
                    st.success("Correct!")
                    correct_count += 1
                elif value is not None:
                    st.error("Wrong!")
            button = st.form_submit_button()
            if correct_count == num_quiz:
                st.balloons()

    else:
        st.markdown(
            """
                    Please upload a document to generate a quiz.

                    You can adjust the difficulty level in sidebar"""
        )
else:
    st.markdown(
        """
                Please enter your OpenAI API key first.
        """
    )
