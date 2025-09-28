# sql_agent.py

import os
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from sqlalchemy import text, inspect, create_engine
from sqlalchemy.orm import sessionmaker
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# Import your external LLM loader
from my_llm import get_llm   # <-- replace with your actual file path
llm = get_llm("gemini")

# --------------------------------------------------------------------
# MULTI-DB CONNECTION HANDLING
# --------------------------------------------------------------------

DB_CONNECTIONS = {
    "db1": "oracle+cx_oracle://user:password@host1:1521/db1",
    "db2": "oracle+cx_oracle://user:password@host2:1521/db2",
    "db3": "oracle+cx_oracle://user:password@host3:1521/db3",
    "db4": "oracle+cx_oracle://user:password@host4:1521/db4",
}

def get_engine(instance_name: str):
    if instance_name not in DB_CONNECTIONS:
        raise ValueError(f"Unknown DB instance: {instance_name}")
    return create_engine(DB_CONNECTIONS[instance_name])

# --------------------------------------------------------------------
# AGENT STATE
# --------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    sql_query: str
    query_result: str
    query_rows: list
    current_user: str
    attempts: int
    relevance: str
    sql_error: bool
    db_instance: str   # NEW: DB selection

# --------------------------------------------------------------------
# DB INSTANCE SELECTION
# --------------------------------------------------------------------

class GetDBInstance(BaseModel):
    db_instance: str = Field(
        description="The Oracle DB instance to connect to (db1, db2, db3, db4)."
    )

def get_db_instance(state: AgentState, config):
    db_instance = config["configurable"].get("db_instance")
    if not db_instance:
        raise ValueError("No DB instance provided. Please specify one of db1, db2, db3, db4.")
    state["db_instance"] = db_instance
    print(f"Selected DB instance: {db_instance}")
    return state

# --------------------------------------------------------------------
# SCHEMA RETRIEVAL
# --------------------------------------------------------------------

def get_database_schema(instance_name):
    engine = get_engine(instance_name)
    inspector = inspect(engine)
    schema = ""
    for table_name in inspector.get_table_names():
        schema += f"Table: {table_name}\n"
        for column in inspector.get_columns(table_name):
            col_name = column["name"]
            col_type = str(column["type"])
            if column.get("primary_key"):
                col_type += ", Primary Key"
            if column.get("foreign_keys"):
                fk = list(column["foreign_keys"])[0]
                col_type += f", Foreign Key to {fk.column.table.name}.{fk.column.name}"
            schema += f"- {col_name}: {col_type}\n"
        schema += "\n"
    print(f"Retrieved schema from {instance_name}.")
    return schema

# --------------------------------------------------------------------
# CURRENT USER (optional, kept as in your code)
# --------------------------------------------------------------------

class GetCurrentUser(BaseModel):
    current_user: str = Field(description="The name of the current user.")

def get_current_user(state: AgentState, config):
    user_id = config["configurable"].get("current_user_id", None)
    if not user_id:
        state["current_user"] = "Anonymous"
        print("No user ID provided. Defaulting to Anonymous.")
        return state
    # Replace with actual DB lookup if needed
    state["current_user"] = f"user_{user_id}"
    print(f"Current user set to: {state['current_user']}")
    return state

# --------------------------------------------------------------------
# RELEVANCE CHECK
# --------------------------------------------------------------------

class CheckRelevance(BaseModel):
    relevance: str = Field(description="'relevant' or 'not_relevant'.")

def check_relevance(state: AgentState, config):
    question = state["question"]
    schema = get_database_schema(state["db_instance"])
    print(f"Checking relevance of question: {question}")
    system = f"""You are an assistant that determines whether a given question is related to the database schema.

Schema:
{schema}

Respond with only "relevant" or "not_relevant".
"""
    human = f"Question: {question}"
    check_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    structured_llm = llm.with_structured_output(CheckRelevance)
    relevance_checker = check_prompt | structured_llm
    relevance = relevance_checker.invoke({})
    state["relevance"] = relevance.relevance
    print(f"Relevance: {state['relevance']}")
    return state

# --------------------------------------------------------------------
# NATURAL LANGUAGE TO SQL
# --------------------------------------------------------------------

class ConvertToSQL(BaseModel):
    sql_query: str = Field(description="SQL query.")

def convert_nl_to_sql(state: AgentState, config):
    question = state["question"]
    current_user = state["current_user"]
    schema = get_database_schema(state["db_instance"])
    print(f"Converting question to SQL: {question}")
    system = f"""You are an assistant that converts natural language questions into SQL queries 
based on the following schema:

{schema}

The current user is '{current_user}'.
Provide only the SQL query.
"""
    convert_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "Question: {question}")]
    )
    structured_llm = llm.with_structured_output(ConvertToSQL)
    sql_generator = convert_prompt | structured_llm
    result = sql_generator.invoke({"question": question})
    state["sql_query"] = result.sql_query
    print(f"Generated SQL: {state['sql_query']}")
    return state

# --------------------------------------------------------------------
# SQL EXECUTION
# --------------------------------------------------------------------

def execute_sql(state: AgentState):
    sql_query = state["sql_query"].strip()
    engine = get_engine(state["db_instance"])
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    print(f"Executing on {state['db_instance']}: {sql_query}")
    try:
        result = session.execute(text(sql_query))
        if sql_query.lower().startswith("select"):
            rows = result.fetchall()
            columns = result.keys()
            state["query_rows"] = [dict(zip(columns, row)) for row in rows]
            if rows:
                state["query_result"] = f"Found {len(rows)} rows."
            else:
                state["query_result"] = "No results found."
            state["sql_error"] = False
        else:
            session.commit()
            state["query_result"] = "Action executed successfully."
            state["sql_error"] = False
    except Exception as e:
        state["query_result"] = f"Error executing SQL: {str(e)}"
        state["sql_error"] = True
    finally:
        session.close()
    return state

# --------------------------------------------------------------------
# HUMAN ANSWER GENERATION
# --------------------------------------------------------------------

def generate_human_readable_answer(state: AgentState):
    sql = state["sql_query"]
    result = state["query_result"]
    current_user = state["current_user"]
    print("Generating human-readable answer...")
    system = """You are an assistant that converts SQL query results into clear responses."""
    generate_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", f"SQL: {sql}\nResult: {result}")]
    )
    human_response = generate_prompt | llm | StrOutputParser()
    answer = human_response.invoke({})
    state["query_result"] = f"Hello {current_user}, {answer}"
    return state

# --------------------------------------------------------------------
# FALLBACK / RETRIES
# --------------------------------------------------------------------

def regenerate_query(state: AgentState):
    state["attempts"] += 1
    return state

def generate_funny_response(state: AgentState):
    state["query_result"] = "I can't answer that, but maybe try a different DB question?"
    return state

def end_max_iterations(state: AgentState):
    state["query_result"] = "Too many attempts. Please try again."
    return state

# --------------------------------------------------------------------
# ROUTERS
# --------------------------------------------------------------------

def relevance_router(state: AgentState):
    return "convert_to_sql" if state["relevance"] == "relevant" else "generate_funny_response"

def check_attempts_router(state: AgentState):
    return "convert_to_sql" if state["attempts"] < 3 else "end_max_iterations"

def execute_sql_router(state: AgentState):
    return "generate_human_readable_answer" if not state.get("sql_error", False) else "regenerate_query"

# --------------------------------------------------------------------
# WORKFLOW
# --------------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("get_db_instance", get_db_instance)
workflow.add_node("get_current_user", get_current_user)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("convert_to_sql", convert_nl_to_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("generate_human_readable_answer", generate_human_readable_answer)
workflow.add_node("regenerate_query", regenerate_query)
workflow.add_node("generate_funny_response", generate_funny_response)
workflow.add_node("end_max_iterations", end_max_iterations)

workflow.set_entry_point("get_db_instance")
workflow.add_edge("get_db_instance", "get_current_user")
workflow.add_edge("get_current_user", "check_relevance")

workflow.add_conditional_edges(
    "check_relevance", relevance_router,
    {"convert_to_sql": "convert_to_sql", "generate_funny_response": "generate_funny_response"}
)

workflow.add_edge("convert_to_sql", "execute_sql")

workflow.add_conditional_edges(
    "execute_sql", execute_sql_router,
    {"generate_human_readable_answer": "generate_human_readable_answer", "regenerate_query": "regenerate_query"}
)

workflow.add_conditional_edges(
    "regenerate_query", check_attempts_router,
    {"convert_to_sql": "convert_to_sql", "end_max_iterations": "end_max_iterations"}
)

workflow.add_edge("generate_human_readable_answer", END)
workflow.add_edge("generate_funny_response", END)
workflow.add_edge("end_max_iterations", END)

app = workflow.compile()
