
import os
from typing import Dict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from sqlalchemy.exc import SQLAlchemyError
import cx_Oracle

# Load environment variables
load_dotenv()

# ------------------------------------
# 1. Oracle + Autosys Database Setup
# ------------------------------------
class AutosysOracleDatabase:
    """Specialized class for Autosys Oracle database operations"""
    
    def __init__(self, connection_string: str):
        """Initialize Oracle connection for Autosys database"""
        try:
            self.db = SQLDatabase.from_uri(connection_string)
            self.llm = ChatOpenAI(
                model="gpt-4o-mini", 
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Test connection
            test_result = self.db.run("SELECT 1 FROM dual")
            print("✅ Connected to Autosys Oracle database successfully")
            
            # Get Autosys schema information
            self._discover_autosys_schema()
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print("Common fixes:")
            print("1. Install cx_Oracle: pip install cx_Oracle")
            print("2. Check Oracle connection string format")
            print("3. Verify Autosys database access permissions")
            raise
    
    def _discover_autosys_schema(self):
        """Discover and cache Autosys table structure"""
        try:
            print("\n🔍 Discovering Autosys database schema...")
            
            # Common Autosys tables (adjust based on your version)
            autosys_tables = [
                'JOB', 'JOB_STATUS', 'JOB_RUNS', 'JOB_DEPENDENCY',
                'CALENDAR', 'MACHINE', 'GLOBAL_VARIABLES',
                'BOX_JOB', 'CMD_JOB', 'FILE_WATCHER_JOB'
            ]
            
            self.available_tables = []
            self.schema_info = {}
            
            for table in autosys_tables:
                try:
                    # Check if table exists by querying
                    self.db.run(f"SELECT COUNT(*) FROM {table} WHERE ROWNUM = 1")
                    self.available_tables.append(table)
                    print(f"✅ Found table: {table}")
                except:
                    print(f"⚠️  Table not found or no access: {table}")
            
            if not self.available_tables:
                print("⚠️  No standard Autosys tables found. Discovering all accessible tables...")
                try:
                    # Fallback: discover all accessible tables
                    all_tables_query = """
                    SELECT table_name FROM user_tables 
                    WHERE table_name LIKE '%JOB%' 
                    OR table_name LIKE '%CALENDAR%'
                    OR table_name LIKE '%SCHEDULE%'
                    ORDER BY table_name
                    """
                    tables_result = self.db.run(all_tables_query)
                    print(f"Available job-related tables: {tables_result}")
                except Exception as e:
                    print(f"Could not discover tables: {e}")
            
            # Get detailed schema information
            self.schema_cache = self.db.get_table_info()
            print(f"\n📊 Available Autosys tables: {self.available_tables}")
            
        except Exception as e:
            print(f"❌ Schema discovery failed: {e}")
            self.available_tables = []
            self.schema_cache = ""

# Initialize database connection
# Update with your actual Autosys Oracle connection details
ORACLE_CONNECTION = "oracle+cx_oracle://username:password@hostname:1521/?service_name=ORCL"
autosys_db = AutosysOracleDatabase(ORACLE_CONNECTION)

# ------------------------------------
# 2. Autosys-Specific SQL Generation
# ------------------------------------
def generate_autosys_sql(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate SQL specifically for Autosys database queries"""
    try:
        question = state["question"]
        schema = autosys_db.schema_cache
        available_tables = autosys_db.available_tables
        
        # Enhanced prompt for Autosys-specific queries
        prompt = f"""
You are an expert in Autosys job scheduling system and Oracle SQL.

AUTOSYS DATABASE CONTEXT:
- Available tables: {available_tables}
- This is an Autosys application database for job scheduling
- Common queries involve job status, schedules, dependencies, calendars

DATABASE SCHEMA:
{schema}

USER QUESTION: {question}

AUTOSYS QUERY PATTERNS:
1. Job Status: SELECT job_name, status, last_start, last_end FROM JOB WHERE job_name LIKE '%job1%'
2. Job Details: SELECT job_name, job_type, command, machine FROM JOB WHERE job_name = 'JOB1'
3. Job Runs History: SELECT job_name, start_time, end_time, status FROM JOB_RUNS WHERE job_name = 'JOB1' ORDER BY start_time DESC
4. Dependencies: SELECT job_name, condition FROM JOB_DEPENDENCY WHERE job_name = 'JOB1'
5. Calendar Info: SELECT calendar_name, date_stamp, run_flag FROM CALENDAR WHERE calendar_name = 'CAL1'
6. Box Jobs: SELECT job_name, box_name FROM BOX_JOB WHERE box_name LIKE '%BOX1%'

ORACLE SQL REQUIREMENTS:
- Use Oracle syntax (ROWNUM instead of LIMIT)
- Use proper date functions (SYSDATE, TO_DATE, etc.)
- Handle NULL values with NVL or NVL2
- Use proper case for Autosys table/column names
- For "top N" queries: WHERE ROWNUM <= N
- Use wildcards (%) for partial job name matches

COMMON AUTOSYS COLUMNS:
- job_name: Job identifier
- status: Current job status (SU=Success, FA=Failure, RU=Running, etc.)
- last_start, last_end: Execution timestamps
- job_type: BOX, CMD, FW (File Watcher), etc.
- machine: Target machine for execution
- command: Command/script to execute

Return ONLY the Oracle SQL query:
"""
        
        response = autosys_db.llm.invoke(prompt)
        sql = response.content.strip()
        
        # Clean up formatting
        sql = sql.replace("```sql", "").replace("```", "").strip()
        if sql.endswith(';'):
            sql = sql[:-1]
        
        print(f"🔍 Generated Autosys SQL: {sql}")
        return {"sql": sql, "question": question}
        
    except Exception as e:
        print(f"❌ Error generating Autosys SQL: {e}")
        return {"sql": "", "error": str(e), "question": state["question"]}

def execute_autosys_sql(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute SQL against Autosys Oracle database"""
    sql = state["sql"]
    question = state["question"]
    
    if not sql or state.get("error"):
        return {**state, "result": "Failed to generate valid SQL for Autosys query"}
    
    try:
        print(f"⚡ Executing Autosys query: {sql}")
        result = autosys_db.db.run(sql)
        
        if not result or str(result).strip() == "":
            result = "No Autosys data found for the specified criteria"
        
        print(f"📊 Autosys query result: {result}")
        return {**state, "result": str(result)}
        
    except SQLAlchemyError as e:
        error_msg = f"Oracle/Autosys query error: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Try to fix common Autosys query issues
        if "invalid identifier" in str(e).lower():
            error_msg += " (Check table/column names - Autosys tables may be case-sensitive)"
        elif "table or view does not exist" in str(e).lower():
            error_msg += f" (Available tables: {autosys_db.available_tables})"
        
        return {**state, "result": error_msg, "needs_retry": True}
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        return {**state, "result": error_msg}

def fix_autosys_sql_error(state: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt to fix common Autosys SQL errors"""
    try:
        if not state.get("needs_retry"):
            return state
            
        original_sql = state["sql"]
        error_result = state["result"]
        question = state["question"]
        
        fix_prompt = f"""
The following Oracle SQL query for Autosys database failed:

Query: {original_sql}
Error: {error_result}

Available Autosys tables: {autosys_db.available_tables}

Common Autosys fixes needed:
1. Table names might be uppercase (JOB instead of job)
2. Column names might be different (job_name vs JOB_NAME)
3. Use Oracle syntax (ROWNUM instead of LIMIT)
4. Autosys status codes: SU, FA, RU, TE, etc.
5. Use proper Oracle date handling

Fix the query for Autosys Oracle database. Return ONLY the corrected SQL:
"""
        
        response = autosys_db.llm.invoke(fix_prompt)
        fixed_sql = response.content.strip().replace("```sql", "").replace("```", "").strip()
        
        print(f"🔧 Attempting fixed Autosys SQL: {fixed_sql}")
        
        try:
            result = autosys_db.db.run(fixed_sql)
            if not result:
                result = "No data found after SQL fix"
            return {"sql": fixed_sql, "question": question, "result": str(result)}
        except:
            return {"sql": original_sql, "question": question, "result": f"Could not fix Autosys query: {error_result}"}
            
    except Exception as e:
        return {**state, "result": f"Error in SQL fix attempt: {str(e)}"}

def summarize_autosys_result(state: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize Autosys query results in business terms"""
    try:
        question = state["question"]
        result = state["result"]
        sql = state.get("sql", "")
        
        if "error" in result.lower() or "failed" in result.lower():
            return {**state, "answer": f"Unable to retrieve Autosys data: {result}"}
        
        prompt = f"""
You are an Autosys expert explaining query results to business users.

Original Question: {question}
Autosys Data Retrieved: {result}

AUTOSYS CONTEXT:
- Job statuses: SU=Success, FA=Failure, RU=Running, TE=Terminated, etc.
- Job types: BOX=Container job, CMD=Command job, FW=File Watcher
- This is job scheduling and workflow automation data

Provide a clear, business-friendly explanation of the Autosys data.
Include relevant context about job scheduling, status meanings, etc.
Format timestamps and status codes in readable terms.

Business-friendly answer:
"""
        
        response = autosys_db.llm.invoke(prompt)
        answer = response.content.strip()
        
        print(f"✅ Autosys answer: {answer}")
        return {**state, "answer": answer}
        
    except Exception as e:
        return {**state, "answer": f"Error summarizing Autosys data: {str(e)}"}

def should_retry_autosys_query(state: Dict[str, Any]) -> str:
    """Decide whether to retry failed Autosys queries"""
    if state.get("needs_retry") and not state.get("retry_attempted"):
        return "fix_sql_error"
    else:
        return "summarize"

# ------------------------------------
# 3. Build Autosys LangGraph Workflow
# ------------------------------------
autosys_workflow = StateGraph(dict)

# Add nodes
autosys_workflow.add_node("generate_sql", generate_autosys_sql)
autosys_workflow.add_node("execute_sql", execute_autosys_sql)
autosys_workflow.add_node("fix_sql_error", fix_autosys_sql_error)
autosys_workflow.add_node("summarize", summarize_autosys_result)

# Add edges with conditional routing
autosys_workflow.add_edge("generate_sql", "execute_sql")
autosys_workflow.add_conditional_edges(
    "execute_sql",
    should_retry_autosys_query,
    {
        "fix_sql_error": "fix_sql_error",
        "summarize": "summarize"
    }
)
autosys_workflow.add_edge("fix_sql_error", "execute_sql")
autosys_workflow.add_edge("summarize", END)

autosys_workflow.set_entry_point("generate_sql")
autosys_app = autosys_workflow.compile()

# ------------------------------------
# 4. Create Autosys Tool
# ------------------------------------
def autosys_tool_func(question: str) -> str:
    """Query Autosys database for job scheduling information"""
    try:
        if not question or question.strip() == "":
            return "Please provide a question about Autosys jobs, calendars, or schedules."
        
        print(f"\n🚀 Processing Autosys query: '{question}'")
        print("=" * 60)
        
        result = autosys_app.invoke({"question": question.strip()})
        answer = result.get("answer", "No answer generated for Autosys query")
        
        print(f"🎯 Autosys response: {answer}")
        return answer
        
    except Exception as e:
        error_msg = f"Autosys tool error: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

# Create the Autosys-specific tool
autosys_sql_tool = Tool(
    name="AutosysQuery",
    func=autosys_tool_func,
    description="""
    Query the Autosys job scheduling database for information about:
    
    - Job status and execution history
    - Job schedules and dependencies  
    - Calendar definitions and dates
    - Box jobs and workflows
    - Machine assignments
    - Error conditions and alerts
    
    Example queries:
    - "What is the status of job1?"
    - "Show me the last 5 runs of BATCH_LOAD_JOB"
    - "Which jobs depend on JOB_EXTRACT?"
    - "What jobs are scheduled to run today?"
    - "Show me all failed jobs from yesterday"
    - "What is the schedule for calendar MONTH_END?"
    
    Input: Natural language question about Autosys jobs/schedules
    Output: Business-friendly explanation of job scheduling data
    """
)

# ------------------------------------
# 5. Test Autosys Scenarios
# ------------------------------------
def test_autosys_queries():
    """Test common Autosys query scenarios"""
    print("\n🧪 Testing Autosys Query Scenarios")
    print("=" * 60)
    
    autosys_test_cases = [
        "What is the status of job1?",
        "Show me the current status of BATCH_PROCESS_JOB",
        "Which jobs failed in the last 24 hours?",
        "What is the schedule for calendar WEEKDAY?",
        "Show me all jobs running on machine PROD01",
        "What are the dependencies for job DATA_LOAD?",
        "List all box jobs and their child jobs",
        "Show me jobs scheduled to run today"
    ]
    
    for i, question in enumerate(autosys_test_cases, 1):
        print(f"\n{i}. Testing: {question}")
        print("-" * 50)
        try:
            answer = autosys_sql_tool.run(question)
            print(f"✅ Result: {answer}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print()

# ------------------------------------
# 6. Integration Example
# ------------------------------------
def create_autosys_agent():
    """Example of using Autosys tool with an agent"""
    try:
        print("\n🤖 Creating Autosys-enabled Agent...")
        
        agent = initialize_agent(
            tools=[autosys_sql_tool],
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        # Test complex Autosys analysis
        complex_queries = [
            "Can you analyze the health of our batch processing jobs?",
            "What jobs should I investigate for performance issues?",
            "Help me understand the dependencies for our daily data pipeline"
        ]
        
        for query in complex_queries:
            print(f"\n🔍 Complex Analysis: {query}")
            try:
                response = agent.run(query)
                print(f"📊 Analysis: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")
            print("-" * 50)
                
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")

if __name__ == "__main__":
    print("🚀 Autosys Oracle Database Query Tool")
    print("=" * 60)
    
    # Update connection string before running
    print("⚠️  IMPORTANT: Update ORACLE_CONNECTION with your Autosys database details")
    print("Format: oracle+cx_oracle://user:pass@host:port/?service_name=SERVICE")
    
    try:
        # Test Autosys queries
        test_autosys_queries()
        
        # Uncomment to test agent
        # create_autosys_agent()
        
        print("\n✅ Autosys tool ready!")
        print("Usage: autosys_sql_tool.run('What is the status of my_job?')")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("Check your Oracle connection and Autosys database access")
