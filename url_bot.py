import os
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from streamlit_mic_recorder import speech_to_text
from streamlit_extras.stylable_container import stylable_container

from url_rag import UrlRAG

load_dotenv()

base_config = dict(
    name = "U-Bot",
    base_url = os.getenv("OLLAMA_BASE_URL"),
    trans_llm = os.getenv("LLM"),
    llm = os.getenv("UB_LLM"),
    embedding = os.getenv('UB_EMBEDDING_MODEL'),
)

languages = dict(
    English="en",
    French="fr",
    Chinese="zh"
)

@st.cache_resource
def load_url_rag():
    with st.spinner(text=f"loading {base_config['name']}..."):
        u_rag = UrlRAG(base_config)
        u_rag.init_models()
        u_rag.init_embeddings()
    return u_rag

st.set_page_config (
    page_title=base_config['name'], 
    page_icon='🎙️',
    layout="wide",
    initial_sidebar_state='auto')

st.title("Website Scraper Bot")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def display_chat():
    # Session state
    if "language" not in st.session_state:
         st.session_state["language"]="English"

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "user_input" not in st.session_state:
        st.session_state["user_input"] = []

    if st.session_state["generated"]:
        size = len(st.session_state["generated"])
        # Display only the last three exchanges
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state["user_input"][i])

            with st.chat_message("assistant"):
                st.write(st.session_state["generated"][i])
        with st.container():
            st.write("&nbsp;")

def lang_code()->str:
    return languages[st.session_state['language']]

def chat_input():
    chat_input_text = st.chat_input(placeholder="What question can I help you resolve today?", disabled=(not st.session_state.get('u_rag')))
    with stylable_container(
        key="bottom_content",
        css_styles="""
            {
                position: fixed;
                bottom: 120px;
            }
            """,
    ):
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            st.empty()
        with col2:
            speech_text = speech_to_text(language=lang_code(), start_prompt="🎙️Listen", stop_prompt="⏹️ Done", just_once=True, key='STT')
        with col3:
            st.selectbox(label='🌐', options=languages.keys(), key="language", index=0)
    user_input = chat_input_text  if chat_input_text else speech_text
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            u_rag = st.session_state.get('u_rag')
            llm_chain, trans_chain = u_rag.rag_chain()
            if lang_code() == "en":
                result = llm_chain.invoke(
                    {"question": user_input}, {"callbacks":[stream_handler]}
                )
            else:
                result = trans_chain.invoke(
                    {"question": user_input, "language": st.session_state['language']}, {"callbacks":[stream_handler]}
                )
            output = result
            st.session_state[f"user_input"].append(user_input)
            st.session_state[f"generated"].append(output)
#see https://python.langchain.com/docs/expression_language/interface

with st.sidebar:
        st.markdown('## URL Config')
        url = st.text_input("Website", "https://www.toronto.ca/home/311-toronto-at-your-service/")
        bt_load = st.button(label='Loading')
        if bt_load:
            u_rag = load_url_rag()
            st.session_state['u_rag']=u_rag
            with st.spinner(text=f"Loading from {url}"):
                u_rag.load_and_retrieve_docs(url)
                del st.session_state["user_input"]
                del st.session_state["generated"]
                st.success("Loaded", icon="✅")

display_chat()
chat_input()
