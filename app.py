import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.cache import InMemoryCache

import json

import os
import random

os.environ["OPENAI_API_KEY"] = "dummy_api_key"
cache = InMemoryCache()

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

chat_model = None
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
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"]
        )
        if file:
            temperature = st.slider("Variety", 0.0, 1.0, 0.5, 0.1)
            chat_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
            difficulty = st.selectbox(
                "Difficulty level",
                ["Beginner", "Intermediate", "Advanced", "Random"],
                placeholder="Choose the difficulty level for the quiz.",
            )
            num_quiz = st.number_input("The number of quiz", 0, 20, step=1, value=5)
            language = st.selectbox(
                "Language", ["Korean", "English", "Japanese", "Chinese"]
            )
            user_req = st.text_area(
                "Additional requirements",
                placeholder="""Enter anything you want....
    Ex) Give me Quiz about Winston.
                """,
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


if chat_model:
    llm = ChatOpenAI(
        model=chat_model,
        temperature=temperature,
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler(),
        ],
    ).bind(function_call="auto", functions=[function])


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(
    num_quiz,
    difficulty,
    language,
    user_req,
    _docs,
    topic,
):
    os.environ["OPENAI_API_KEY"] = api_key
    chain = (
        {
            "num_quiz": lambda input: input["num_quiz"],
            "difficulty": lambda input: input["difficulty"],
            "language": lambda input: input["language"],
            "requirements": lambda input: input["requirements"],
            "context": format_docs,
        }
        | prompt
        | llm
    )
    response = chain.invoke(
        {
            "num_quiz": str(num_quiz),
            "difficulty": difficulty,
            "language": language,
            "requirements": user_req,
            "context": _docs,
        }
    )
    quiz_json = json.loads(response.additional_kwargs["function_call"]["arguments"])
    problems = quiz_json["problems"]
    for problem in problems:
        question = problem["question"]
        answers = problem["answers"]
        random.shuffle(answers)
    return problems


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.

            Based ONLY on the following context make {num_quiz} questions to test the user's knowledge about the text.

            The diffculty level of the quiz is {difficulty}

            The quiz should be created in {language}

            Each question should have 4 answers, three of them must be incorrect and one should be correct.

            You will generate all quiz as json format

            Additional requirements : {requirements}

            Your turn! 
            
            IMPORTANT: You do not create problems in the order of the context provided but shuffle them instead.

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
        if st.button("New Quiz"):
            st.cache_data.clear()
        docs = load_file(file)
        problems = run_quiz_chain(
            num_quiz=num_quiz,
            difficulty=difficulty,
            language=language,
            user_req=user_req,
            _docs=docs,
            topic=file.name,
        )
        with st.form("questions form"):
            correct_count = 0
            for problem in problems:
                st.write(problem["question"])
                value = st.radio(
                    "Select an option",
                    [answer["answer"] for answer in problem["answers"]],
                    index=None,
                )
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
