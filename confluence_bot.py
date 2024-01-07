# @see https://github.com/devsentient/examples

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
import os
from dotenv import load_dotenv

# Import the ConfluenceQA class
from confluence_qa import ConfluenceQA

load_dotenv()

base_config = dict(
    db_url = os.getenv("NEO4J_URI"),
    db_username = os.getenv("NEO4J_USERNAME"),
    db_password = os.getenv("NEO4J_PASSWORD"),
    llm_name = os.getenv("LLM"),
    ollama_base_url = os.getenv("OLLAMA_BASE_URL"),
    embedding_model_name = os.getenv("EMBEDDING_MODEL")
)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

st.set_page_config(
    page_title='Q&A Bot for Confluence Page',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='auto',
)


@st.cache_resource
def load_confluence(config):
    # st.write("loading the confluence page")
    confluence_qa = ConfluenceQA(config=config)
    confluence_qa.init_embeddings()
    confluence_qa.init_models()
    confluence_qa.vector_store_confluence_docs(force_reload=True)
    confluence_qa.retreival_qa_chain()
    return confluence_qa

def display_chat():
    # Session state
    if "config" not in st.session_state:
        st.session_state["config"] = {}
        st.session_state["config"].update(base_config)

    if "confluence_qa" not in st.session_state:
        st.session_state["confluence_qa"] = None

    if "generated" not in st.session_state:
        st.session_state[f"generated"] = []

    if "user_input" not in st.session_state:
        st.session_state[f"user_input"] = []

    if st.session_state[f"generated"]:
        size = len(st.session_state[f"generated"])
        # Display only the last three exchanges
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])

            with st.chat_message("assistant"):
                st.write(st.session_state[f"generated"][i])
        with st.container():
            st.write("&nbsp;")

def chat_input():
    user_input = st.chat_input("What question can I help you resolve today?")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            confluence_qa = st.session_state.get("confluence_qa")
            if confluence_qa is not None:
                qa_chain = confluence_qa.retreival_qa_chain()
                result = qa_chain(
                    {"question": user_input, "chat_history": []}, callbacks=[stream_handler]
                )["answer"]
                output = result
                st.session_state[f"user_input"].append(user_input)
                st.session_state[f"generated"].append(output)
            else:
                st.write("Please load Confluence Page content first by [Submit] your configs from left side.")

with st.sidebar.form(key ='ConfigForm'):
        st.markdown('## Add your configs')
        confluence_url = st.text_input("paste the confluence URL", "https://toronto.atlassian.net/wiki")
        username = st.text_input(label="confluence username",
                                help="leave blank if confluence page is public",
                                type="password")
        space_key = st.text_input(label="confluence space",
                                help="Space of Confluence",
                                value="DTSKS")
        api_key = st.text_input(label="confluence api key",
                                help="leave blank if confluence page is public",
                                type="password")
        max_pages = st.text_input(label="maximum pages",
                                value ="100",
                                help="maximum number of pages loaded")
        btSubmitted = st.form_submit_button(label='Submit')

        if btSubmitted and confluence_url and space_key:
            st.session_state["config"].update({
                "confluence_url": confluence_url,
                "username": username if username != "" else None,
                "api_key": api_key if api_key != "" else None,
                "space_key": space_key,
                "max_pages": int(max_pages)
            })
            with st.spinner(text="Ingesting Confluence..."):
                confluence_qa = load_confluence(st.session_state["config"])
                st.session_state["confluence_qa"] = confluence_qa
            st.success("Confluence Space Ingested", icon="✅")
st.title("Confluence Q&A Demo")

display_chat()
chat_input()
