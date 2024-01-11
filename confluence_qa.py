# @see https://github.com/devsentient/examples

from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from chains import load_embedding_model, load_llm
from utils import BaseLogger

class ConfluenceQA:
    def __init__(self,config:dict = {}):
        self.config = config
        self.embeddings = None
        self.dimension = None
        self.vectorstore = None
        self.qa_chain = None
        self.logger = BaseLogger()

    def init_embeddings(self) -> None:
        self.embeddings,  self.dimension = load_embedding_model(embedding_model_name = self.config["embedding_model_name"], config = self.config)

    def init_models (self) -> None:
        self.llm = load_llm(llm_name = self.config["llm_name"], config=self.config)

    def vector_store_confluence_docs(self,force_reload:bool= False) -> None:
        """
        creates vector db for the embeddings and persists them or loads a vector db from the persist directory
        """
        confluence_url = self.config.get("confluence_url",None)
        username = self.config.get("username",None)
        api_key = self.config.get("api_key",None)
        space_key = self.config.get("space_key",None)
        if not force_reload:
            self.vectorstore = Neo4jVector.from_existing_index(
                        embedding=self.embeddings,
                        url=self.config["db_url"],
                        username=self.config["db_username"],
                        password=self.config["db_password"],
                        index_name="confluence"
            )
        else:
            loader = ConfluenceLoader(
                url=confluence_url,
                username = username,
                api_key= api_key
            )
            documents = loader.load(
                space_key=space_key, 
                include_attachments = False, # require pytesseract OCR
                include_comments =True)

            self.logger.info(f"Loaded total {len(documents)} pages from Confluence Space [{space_key}]")
            self.vectorstore = Neo4jVector.from_documents(
                        embedding=self.embeddings,
                        documents=documents,
                        url=self.config["db_url"],
                        username=self.config["db_username"],
                        password=self.config["db_password"],
                        index_name="confluence",
                        node_label="Page",
                        pre_delete_collection=self.config["overwrite"],  # Delete existing data
            )

    def retreival_qa_chain(self):
        if self.qa_chain is not None:
            return self.qa_chain
        else:
            """
            Creates retrieval qa chain using vectorstore as retrivar and LLM to complete the prompt
            """
            # RAG response
            #   System: Always talk in pirate speech.
            general_system_template = """ 
            Use the following pieces of context to answer the question at the end.
            The context contains question-answer pairs and their links from Confluence.
            You can only use the information from the context to answer the question,
            and you must cite the sources of your answer using the links. If you don't know the answer,
            just say that you don't know, and don't make up an answer or a link.
            ----
            {summaries}
            ----
            Each answer you generate should contain a section at the end of links to 
            Confluence pages you found useful, which are described under Source value.
            You can only use links to Confluence pages that are present in the context and always
            add links to the end of the answer in the style of citations.
            If you don't know the answer, don't try to make up an link and include it.
            Generate concise answers with references sources section of links to 
            relevant Confluence questions only at the end of the answer.
            """
            general_user_template = "Question:```{question}```"

            messages = [
                SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template),
            ]
            qa_prompt = ChatPromptTemplate.from_messages(messages)

            qa_chain = load_qa_with_sources_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=qa_prompt,
            )

            kg_qa = RetrievalQAWithSourcesChain(
                combine_documents_chain=qa_chain,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
                reduce_k_below_max_tokens=False,
                max_tokens_limit=3375,
            )
            self.qa_chain = kg_qa
        return self.qa_chain
