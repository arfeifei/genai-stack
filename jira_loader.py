import os
import math
from atlassian import Jira
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
import streamlit as st
from streamlit.logger import get_logger
from chains import load_embedding_model

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

jira_url = os.getenv("JIRA_INSTANCE_URL")
jira_username = os.getenv("JIRA_USERNAME")
jira_password = os.getenv("JIRA_API_TOKEN")

jira = Jira(
    url=jira_url,
    username=jira_username,
    password=jira_password,
    cloud=True)

embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

# if Neo4j is local, you can go to http://localhost:7474/ to browse the database
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)

def create_vector_index(driver, dimension: int) -> None:
    index_query = "CALL db.index.vector.createNodeIndex('jira', 'Issue', 'embedding', $dimension, 'cosine')"
    try:
        driver.query(index_query, {"dimension": dimension})
    except:  # Already exists
        pass

def create_constraints(driver) -> None:
    driver.query(
        "CREATE CONSTRAINT issue_id IF NOT EXISTS FOR (i:Issue) REQUIRE (i.id) IS UNIQUE"
    )

def load_jira_data(projects:[]) -> None:
    #https://atlassian-python-api.readthedocs.io/jira.html#manage-projects
    issues = []
    for project in projects:
        iss = jira.get_all_project_issues(project, fields=['key','description','summary','status','issuelinks','comment'])
        issues = issues + iss
    # dump the issue from Jira to JSON
    # with open("data.json", "w") as f:
    #     json.dump(issues, f, indent=2)
    # with open("data.json", "r") as f:
    #     issues = json.load(f)
    insert_jira_data(issues)

def insert_jira_data(issues:[]) -> None:

    # Calculate embedding values for questions and answers
    for issue in issues:
        text = ''
        description = issue["fields"]["description"]
        summary = issue["fields"]["summary"]
        if description:
            text += description + "\n"
        if summary:
            text += summary + "\n"
        for comment in issue["fields"]["comment"]["comments"]:
            body = comment["body"]
            if body:
                text += body + "\n"
        issue["text"] = text
        issue["embedding"] = embeddings.embed_query(text)
    # Cypher, the query language of Neo4j, is used to import the data
    # https://neo4j.com/docs/getting-started/cypher-intro/
    # https://neo4j.com/docs/cypher-cheat-sheet/5/auradb-enterprise/
    # Define a query to save the issues to Neo4j
    import_query = """
    WITH $data AS issues
    UNWIND issues AS issue
    MERGE (i:Issue {key: issue.key})
    SET i.id = issue.id,
        i.status = issue.fields.status.name,
        i.text = issue.text,
        i.embedding = issue.embedding

    WITH i, issue
    UNWIND issue.fields.issuelinks AS link
    WITH i, issue, link
    WHERE link.outwardIssue.key IS NOT NULL
    MERGE (relatedIssueOutward:Issue {key: link.outwardIssue.key})
    MERGE (i)-[:RELATED {type: link.type.name, outward: link.type.outward}]->(relatedIssueOutward)
    MERGE (relatedIssueOutward)-[:RELATED {type: link.type.name, inward: link.type.inward}]->(i)

    WITH i, issue, link
    WHERE link.inwardIssue.key IS NOT NULL
    MERGE (relatedIssueInward:Issue {key: link.inwardIssue.key})
    MERGE (i)<-[:RELATED {type: link.type.name, inward: link.type.inward}]-(relatedIssueInward)
    MERGE (relatedIssueInward)-[:RELATED {type: link.type.name, outward:link.type.outward}]->(i)
    """
    neo4j_graph.query(import_query, {"data": issues})

# Streamlit
def get_filter() -> str:
    input_text = st.text_input(
        "Enter project name or key filter (Optional)", value=""
    )
    return input_text

def get_project_info(filter_query:str=""):
    data = jira.paginated_projects(url=f"{jira.resource_url('project/search')}?startAt=0&query={filter_query}")
    total_records, number_page = (int(data['total']), int(data['maxResults']))
    # Initialize an empty list to store the records
    records = []
    # total pages
    pages = math.ceil(total_records / number_page)
    # Loop through the pages
    for page in range(pages):
        # Calculate the start and end index of the records for the current page
        start = page * number_page
        # end = min((page + 1) * number_page, total_records)
        records.append(start)
    return (total_records, number_page, records)

def get_paginated_projects(start_at:int, number_page:int, filter_query:str="") -> []:
    projectKeys = []
    data = jira.paginated_projects(url=f"{jira.resource_url('project/search')}?startAt={start_at}&maxResults={number_page}&query={filter_query}")
    projects = data['values']
    for project in projects:
        projectKeys.append(f"{project['name']} | ({project['key']})")
    return projectKeys

def render_page() -> None:
    st.header("Jira Project Issues Loader")
    try:
        filter_query =  get_filter()
        st.session_state['filterQuery'] = filter_query

        total_records, number_page, records = get_project_info(st.session_state['filterQuery'])
        totalheader = f"Total {total_records} projects under {jira_url}"
        if len(filter_query) > 0:
            totalheader += f" with filter ({filter_query}) on"
        st.subheader(totalheader)
        st.caption("Go to http://localhost:7474/ to explore the graph.")
        if total_records > number_page: 
            start_at = st.selectbox("Select project start number", records)
            st.caption(f'List next {number_page} projects start from {start_at}')
            st.session_state['startAt'] = start_at
        else:
            st.session_state['startAt'] = 0
        paged_projects = get_paginated_projects(st.session_state['startAt'], number_page, st.session_state['filterQuery'])
        options = st.multiselect('Which project do you want to import',paged_projects)
        projects = []
        for s in options:
            s = s.split("|")[1].strip().strip("()")
            projects.append(s)
        if projects:
            if st.button("Import", type="primary"):
                with st.spinner("Loading... This might take a minute or two."):
                    load_jira_data(projects)
                    st.success("Import successful", icon="✅")
                    st.caption("Go to http://localhost:7474/ to interact with the database")
    except Exception as e:
        st.error(f"Error: {e}", icon="🚨")

create_constraints(neo4j_graph)
create_vector_index(neo4j_graph, dimension)

render_page()
