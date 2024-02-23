from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from utils import BaseLogger

class UrlRAG:
    # Initialization functions
    def __init__(self, config:dict={}):
        self.config = config
        self.name=config['name']
        self._embedding = None
        self._trans_llm=None
        self._llm =None
        self._retriever = None
        self._llm_chain = None
        self._trans_chain = None
        self.logger = BaseLogger()

    def init_models (self) -> None:
        self._llm = ChatOllama(
            temperature=0,
            base_url=self.config['base_url'],
            model=self.config['llm'],
            streaming=True,
            # seed=2,
#            top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
#            top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
#            num_ctx=3072,  # Sets the size of the context window used to generate the next token.
        )
        self._trans_llm = ChatOllama(
            temperature=0,
            base_url=self.config['base_url'],
            model=self.config['trans_llm'],
            streaming=True,
            # seed=2,
#            top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
#            top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
#            num_ctx=3072,  # Sets the size of the context window used to generate the next token.
        )

    def init_embeddings(self) -> None:
        self._embeddings = OllamaEmbeddings( base_url=self.config['base_url'], model=self.config['embedding']) 

    # Function to load, split, and retrieve documents
    def load_and_retrieve_docs(self, url):
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict() 
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=self._embeddings)
        self._retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Function to format documents
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def rag_chain(self):
        if not self._llm_chain:
            formatted_context = RunnableLambda(self.format_docs) 

            self._llm_chain = (
                {"context": self._retriever | formatted_context, "question": RunnablePassthrough()} 
                | ChatPromptTemplate.from_template("""Answer the question based only on the following context:
                    {context}

                    Question: {question}
                    """
                    )
                | self._llm
                | StrOutputParser()
            )

            _chain = (
                self._llm_chain
                | {
                    "context": RunnablePassthrough(),
                    "language": RunnablePassthrough()
                }
            )

            self._trans_chain =  (
                _chain
                | {
                    "context":  itemgetter("context"),
                    "language": itemgetter("language")
                }
                | ChatPromptTemplate.from_template("""Translate the following context in {language}
                {context} 

                """)
                | self._trans_llm
                | StrOutputParser()
            )
        return self._llm_chain , self._trans_chain
