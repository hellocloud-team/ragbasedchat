
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy.exc import SQLAlchemyError

# ------------------------------------
# 1. Setup LLM + Oracle DB
# ------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Example Oracle URI:
# "oracle+cx_oracle://username:password@hostname:1521/?service_name=ORCL"
db = SQLDatabase.from_uri(
    "oracle+cx_oracle://scott:tiger@localhost:1521/?service_name=ORCLCDB"
)

# ------------------------------------
# 2. Define workflow functions
# ------------------------------------
def generate_sql(state):
    """Generate SQL from natural language"""
    schema = db.get_table_info()
    question = state["question"]

    prompt = f"""
You are a SQL expert. 
Database schema:
{schema}

User question: {question}

Write a valid **Oracle SQL** query. Return ONLY the SQL.
"""
    sql = llm.invoke(prompt).content.strip()
    return {"sql": sql}


def run_sql(state):
    """Execute SQL safely"""
    sql = state["sql"]
    try:
        result = db.run(sql)
        return {"result": result}
    except SQLAlchemyError as e:
        return {"result": f"SQL ERROR: {e}"}


def summarize(state):
    """Turn SQL result into a natural language answer"""
    question = state["question"]
    result = state["result"]

    prompt = f"""
Question: {question}
SQL Result: {result}

Answer the question in plain English.
"""
    answer = llm.invoke(prompt).content.strip()
    return {"answer": answer}

# ------------------------------------
# 3. Build LangGraph
# ------------------------------------
graph = StateGraph(dict)

graph.add_node("generate_sql", generate_sql)
graph.add_node("run_sql", run_sql)
graph.add_node("summarize", summarize)

graph.add_edge("generate_sql", "run_sql")
graph.add_edge("run_sql", "summarize")
graph.add_edge("summarize", END)

graph.set_entry_point("generate_sql")
app = graph.compile()

# ------------------------------------
# 4. Run it
# ------------------------------------
result = app.invoke({"question": "Show me the top 5 highest paid employees"})
print("Final Answer:", result["answer"])



from langchain.tools import Tool

# Wrap LangGraph app as a function
def text_to_sql_tool_func(question: str) -> str:
    result = app.invoke({"question": question})
    return result["answer"]

# Create tool object
text_to_sql_tool = Tool(
    name="TextToSQL",
    func=text_to_sql_tool_func,
    description="Use this to answer questions about structured data in the Oracle database."
)

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Agent with your SQL tool
agent = initialize_agent(
    tools=[text_to_sql_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # function-calling style
    verbose=True
)

# Ask a question
response = agent.run("What is the average salary of employees in IT?")
print("Agent Answer:", response)

from langchain.tools import Tool

# Wrap LangGraph app as a callable function
def text_to_sql_tool_func(question: str) -> str:
    """Call the LangGraph pipeline with a natural language question"""
    result = app.invoke({"question": question})
    return result["answer"]

# Define the tool
text_to_sql_tool = Tool(
    name="TextToSQL",
    func=text_to_sql_tool_func,
    description="Answer questions about company data by converting natural language to Oracle SQL."
)


from langchain.agents import initialize_agent, AgentType

# Initialize agent with tool
agent = initialize_agent(
    tools=[text_to_sql_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Ask a question (agent decides when to use the tool)
response = agent.run("What is the average salary of IT employees?")
print("Agent Answer:", response)



# Example DB map
DB_CONNECTIONS = {
    "hr_app": SQLDatabase.from_uri("sqlite:///hr.db"),
    "finance_app": SQLDatabase.from_uri("sqlite:///finance.db"),
    "sales_app": SQLDatabase.from_uri("oracle+cx_oracle://scott:tiger@localhost:1521/?service_name=ORCLCDB")
}



# Example DB map
DB_CONNECTIONS = {
    "hr_app": SQLDatabase.from_uri("sqlite:///hr.db"),
    "finance_app": SQLDatabase.from_uri("sqlite:///finance.db"),
    "sales_app": SQLDatabase.from_uri("oracle+cx_oracle://scott:tiger@localhost:1521/?service_name=ORCLCDB")
}


def generate_sql(state):
    """Generate SQL from natural language"""
    app_name = state["app_name"]
    db = DB_CONNECTIONS[app_name]

    schema = db.get_table_info()
    question = state["question"]

    prompt = f"""
You are an SQL expert. 
Database schema for app '{app_name}':
{schema}

User question: {question}

Write a valid SQL query for this schema. Return ONLY the SQL.
"""
    sql = llm.invoke(prompt).content.strip()
    return {"sql": sql}


def run_sql(state):
    """Execute SQL safely"""
    app_name = state["app_name"]
    db = DB_CONNECTIONS[app_name]

    sql = state["sql"]
    try:
        result = db.run(sql)
        return {"result": result}
    except SQLAlchemyError as e:
        return {"result": f"SQL ERROR: {e}"}



        result = app.invoke({
    "app_name": "hr_app",
    "question": "Show me all employees hired after 2020"
})
print("Final Answer:", result["answer"])


def generate_sql(state):
    """Generate SQL just once (schema same across DBs)"""
    # use schema from any DB (say hr_app)
    schema = DB_CONNECTIONS["hr_app"].get_table_info()
    question = state["question"]

    prompt = f"""
You are an SQL expert. 
Database schema:
{schema}

User question: {question}

Write a valid SQL query. Return ONLY the SQL.
"""
    sql = llm.invoke(prompt).content.strip()
    return {"sql": sql}


def run_sql(state):
    """Run SQL against chosen DB"""
    app_name = state["app_name"]
    db = DB_CONNECTIONS[app_name]

    sql = state["sql"]
    try:
        result = db.run(sql)
        return {"result": result}
    except SQLAlchemyError as e:
        return {"result": f"SQL ERROR: {e}"}


    def choose_db(state):
    """Decide which app DB to query based on question"""
    question = state["question"]

    prompt = f"""
You are a routing assistant. 
We have 3 databases: hr_app, finance_app, sales_app.
Decide which one is most relevant for this question.

Question: {question}

Return ONLY the app name (hr_app, finance_app, or sales_app).
"""
    app_name = llm.invoke(prompt).content.strip()
    return {"app_name": app_name}


   graph.add_node("choose_db", choose_db)
graph.add_edge("choose_db", "generate_sql")
graph.set_entry_point("choose_db")

db = SQLDatabase.from_uri(
    "sqlite:///hr.db",
    include_tables=["employees", "departments"]  # ✅ only expose these
)


Chat with PDF using Langchain and Google Gemini

Used: 
1)python
2)Langchain
3)GoogleAI
4)Docker
5)Streamlit


Features

Streamlit Interface: Streamlit provides an easy-to-use interface for building interactive web applications in Python.
PDF Manipulation with PyPDF2: PyPDF2 is a Python library for working with PDF files, allowing users to manipulate PDF documents seamlessly.
Advanced Text Analysis with LangChain: LangChain is a library for advanced text analysis, including capabilities such as RecursiveCharacterTextSplitter for handling complex text structures effectively.
Google Generative AI Embeddings: Leveraging Google Generative AI for intelligent embeddings, which can enhance text analysis and understanding.
Vector Stores with FAISS: FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors, which can be useful for storing and retrieving embeddings efficiently.
Chat-based Generative AI: Integration of Google Generative AI for creating engaging conversational experiences within the application.
Question Answering Chain with LangChain: LangChain's question-answering capabilities enable the application to answer queries based on the content of PDF documents.
Prompt Templates with LangChain: Using predefined templates from the LangChain prompt library can streamline interactions and provide users with prompts for generating responses.
Dotenv Integration: Dotenv integration ensures that sensitive credentials and environment variables are securely managed within the application.
![image](https://github.com/hellocloud-team/ragbasedchat/assets/163302215/1732d9b7-803a-4f67-b0d1-18721100fd84)





Training Takeaways: Secure LLMs, Agentic AI & AI-Powered Java Applications

1. Building Secure LLMs (Large Language Models):

Data Security: Emphasize proper data sanitization and encryption throughout the pipeline—especially during data ingestion and storage.

Access Control: Implement strict access policies for LLM models, APIs, and inference endpoints.

Prompt Injection Defense: Use input validation, context separation, and guardrails to protect against prompt injection attacks.

Auditability: Ensure logging and monitoring are in place to track model usage, anomalies, and abuse patterns.

Ethical Safeguards: Integrate content filters, bias mitigation strategies, and explainability tools.


2. Developing Agentic AI Systems:

Autonomy with Oversight: Build agents that act autonomously but remain aligned with human objectives via continuous feedback loops.

Modular Architecture: Use composable tools like planners, memory modules, and tool interfaces for scalable design.

State Management: Implement persistent and contextual memory systems for long-term agent reasoning.

Security Controls: Prevent agents from taking harmful actions by enforcing constraints and validation at decision points.


3. AI-Powered Application Development with Java:

Framework Integration: Leverage Java ML libraries (e.g., Deep Java Library, Tribuo) or call Python models using JNI or REST APIs.

Microservices Architecture: Deploy AI functionalities as modular, secure microservices for scalability and maintainability.

Data Pipeline Handling: Use Java for robust ETL pipelines to prepare and manage input data for AI models.

Inference Management: Optimize model inference with caching, batching, and concurrency control in Java applications.

Monitoring & Logging: Use tools like Micrometer and Prometheus for observability of AI features in production.




