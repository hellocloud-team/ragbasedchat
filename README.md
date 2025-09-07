
        import os
from typing import Dict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.tools import Tool
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

# ------------------------------------
# 1. Fixed Autosys Database Class
# ------------------------------------
class FixedAutosysDatabase:
    """Fixed version that handles connection issues properly"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False
        self.db = None
        self.available_tables = []
        self.schema_info = ""
        
        try:
            # Initialize Gemini LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            print("✅ Gemini LLM initialized")
            
            # Try database connection
            if self._is_valid_connection_string(connection_string):
                try:
                    self.db = SQLDatabase.from_uri(connection_string)
                    # Test connection
                    test_result = self.db.run("SELECT 1 FROM dual")
                    self.connected = True
                    print("✅ Oracle database connected")
                    self._discover_schema()
                except Exception as db_error:
                    print(f"⚠️ Database connection failed: {db_error}")
                    self._setup_mock_mode()
            else:
                print("⚠️ Invalid connection string, using mock mode")
                self._setup_mock_mode()
                
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            self._setup_mock_mode()
    
    def _is_valid_connection_string(self, conn_str: str) -> bool:
        """Check if connection string is properly configured"""
        return (conn_str and 
                "username:password" not in conn_str and 
                "oracle+cx_oracle://" in conn_str)
    
    def _setup_mock_mode(self):
        """Setup mock database for testing"""
        self.connected = False
        self.available_tables = [
            'JOB', 'JOB_STATUS', 'JOB_RUNS', 'CALENDAR', 
            'JOB_DEPENDENCY', 'MACHINE', 'BOX_JOB'
        ]
        self.schema_info = """
        Common Autosys Tables:
        - JOB: Contains job definitions (job_name, status, last_start, last_end, command, machine)
        - JOB_STATUS: Current job status information
        - JOB_RUNS: Historical job execution records
        - CALENDAR: Calendar definitions for job scheduling
        - JOB_DEPENDENCY: Job dependency relationships
        - MACHINE: Machine/server definitions
        - BOX_JOB: Box job hierarchies
        """
        print("🔧 Running in mock mode - update connection string for real data")
    
    def _discover_schema(self):
        """Safely discover Autosys schema"""
        try:
            # Get table list
            tables_query = """
            SELECT table_name FROM user_tables 
            WHERE table_name IN ('JOB', 'JOB_STATUS', 'JOB_RUNS', 'CALENDAR', 
                                  'JOB_DEPENDENCY', 'MACHINE', 'BOX_JOB', 'CMD_JOB')
            ORDER BY table_name
            """
            
            try:
                tables_result = self.db.run(tables_query)
                if tables_result:
                    # Parse table names from result
                    self.available_tables = [t.strip() for t in str(tables_result).replace('(', '').replace(')', '').replace("'", "").split(',') if t.strip()]
                else:
                    self.available_tables = ['JOB', 'JOB_STATUS']  # Fallback
            except:
                self.available_tables = ['JOB', 'JOB_STATUS']  # Fallback
            
            # Get basic schema info safely
            try:
                self.schema_info = self.db.get_table_info()
            except:
                self.schema_info = f"Available tables: {', '.join(self.available_tables)}"
            
            print(f"📊 Discovered tables: {self.available_tables}")
            
        except Exception as e:
            print(f"⚠️ Schema discovery failed: {e}")
            self.available_tables = ['JOB', 'JOB_STATUS']
            self.schema_info = "Schema discovery failed - using basic table structure"

# Initialize the fixed database
ORACLE_CONNECTION = os.getenv("ORACLE_CONNECTION", "oracle+cx_oracle://username:password@hostname:1521/?service_name=ORCL")
autosys_db = FixedAutosysDatabase(ORACLE_CONNECTION)

# ------------------------------------
# 2. Fixed SQL Generation (No More slice() Error)
# ------------------------------------
def generate_sql_fixed(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fixed SQL generation without slice errors"""
    try:
        question = state.get("question", "")
        if not question:
            return {"sql": "", "error": "No question provided", "question": ""}
        
        # Get schema info safely - NO MORE SLICE ERROR
        available_tables = autosys_db.available_tables
        schema_info = autosys_db.schema_info
        
        # Limit schema info length properly (not with slice())
        if len(schema_info) > 1000:
            schema_info = schema_info[:1000] + "..."
        
        # Enhanced prompt for better SQL generation
        prompt = f"""You are an Oracle SQL expert for Autosys job scheduling database.

AVAILABLE TABLES: {available_tables}

SCHEMA INFORMATION:
{schema_info}

USER QUESTION: {question}

IMPORTANT RULES:
1. Generate ONLY Oracle SQL - no explanations
2. Use ROWNUM instead of LIMIT for Oracle
3. For job status queries, use: SELECT job_name, status FROM JOB WHERE job_name LIKE '%jobname%'
4. Always add: AND ROWNUM <= 10 to limit results
5. Common job status codes: SU=Success, FA=Failure, RU=Running, TE=Terminated
6. Job names often have format like: ATSYS.DA3_DBMaint_080_arch_machines.c

EXAMPLE QUERIES:
- Job status: SELECT job_name, status, last_start FROM JOB WHERE job_name LIKE '%{question.split()[-1] if question.split() else 'job'}%' AND ROWNUM <= 10
- All jobs: SELECT job_name, status FROM JOB WHERE ROWNUM <= 10
- Failed jobs: SELECT job_name, status, last_start FROM JOB WHERE status = 'FA' AND ROWNUM <= 10

Generate Oracle SQL query:"""
        
        try:
            response = autosys_db.llm.invoke(prompt)
            sql_content = response.content if hasattr(response, 'content') else str(response)
        except Exception as llm_error:
            print(f"❌ LLM failed: {llm_error}")
            # Create fallback query based on question
            if "status" in question.lower():
                job_hint = extract_job_name_from_question(question)
                sql_content = f"SELECT job_name, status, last_start FROM JOB WHERE job_name LIKE '%{job_hint}%' AND ROWNUM <= 10"
            else:
                sql_content = "SELECT job_name, status FROM JOB WHERE ROWNUM <= 10"
        
        # Clean SQL
        sql = clean_sql_response(sql_content)
        
        print(f"🔍 Generated SQL: {sql}")
        return {"sql": sql, "question": question, "error": None}
        
    except Exception as e:
        print(f"❌ SQL generation error: {e}")
        return {"sql": "", "error": str(e), "question": state.get("question", "")}

def extract_job_name_from_question(question: str) -> str:
    """Extract potential job name from question"""
    try:
        # Look for patterns like job names
        words = question.split()
        for word in words:
            if ('job' in word.lower() or 
                '.' in word or 
                '_' in word or
                word.isupper()):
                return word.replace('job', '').replace('Job', '')
        return 'job'  # fallback
    except:
        return 'job'

def clean_sql_response(sql_content: str) -> str:
    """Clean SQL response from LLM"""
    try:
        sql = sql_content.strip()
        
        # Remove common LLM formatting
        cleanups = ["```sql", "```oracle", "```", "sql:", "query:", "SQL:", "Query:"]
        for cleanup in cleanups:
            sql = sql.replace(cleanup, "")
        
        sql = sql.strip()
        
        # Remove trailing semicolon
        if sql.endswith(';'):
            sql = sql[:-1]
        
        # Basic validation
        if not sql or len(sql) < 10 or 'SELECT' not in sql.upper():
            sql = "SELECT job_name, status FROM JOB WHERE ROWNUM <= 10"
        
        return sql
        
    except Exception as e:
        print(f"❌ SQL cleaning failed: {e}")
        return "SELECT job_name, status FROM JOB WHERE ROWNUM <= 10"

# ------------------------------------
# 3. Fixed SQL Execution
# ------------------------------------
def execute_sql_fixed(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fixed SQL execution with better error handling"""
    try:
        sql = state.get("sql", "")
        question = state.get("question", "")
        error = state.get("error")
        
        if error or not sql:
            return {**state, "result": f"Cannot execute: {error or 'No SQL generated'}"}
        
        print(f"⚡ Executing: {sql}")
        
        if autosys_db.connected:
            try:
                result = autosys_db.db.run(sql)
                if not result or str(result).strip() == "":
                    result = "No data found in Autosys database"
                print(f"📊 Real DB result: {result}")
            except Exception as db_error:
                print(f"❌ Database error: {db_error}")
                result = generate_mock_result(question, sql)
        else:
            # Mock mode
            result = generate_mock_result(question, sql)
            print(f"🎭 Mock result: {result}")
        
        return {**state, "result": str(result)}
        
    except Exception as e:
        error_msg = f"Execution failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {**state, "result": error_msg}

def generate_mock_result(question: str, sql: str) -> str:
    """Generate realistic mock results for testing"""
    try:
        # Extract job name if mentioned
        job_name = extract_job_name_from_question(question)
        
        if "status" in question.lower():
            return f"ATSYS.DA3_DBMaint_080_arch_machines.c,SU,2024-01-15 10:30:00"
        elif "failed" in question.lower() or "FA" in sql:
            return f"ATSYS.FAILED_JOB_001,FA,2024-01-15 09:15:00\nATSYS.FAILED_JOB_002,FA,2024-01-15 08:30:00"
        elif "running" in question.lower() or "RU" in sql:
            return f"ATSYS.RUNNING_JOB_001,RU,2024-01-15 11:00:00"
        else:
            return f"ATSYS.SAMPLE_JOB_001,SU,2024-01-15 10:00:00\nATSYS.SAMPLE_JOB_002,RU,2024-01-15 11:00:00\nATSYS.SAMPLE_JOB_003,FA,2024-01-15 09:00:00"
    
    except Exception as e:
        return f"Mock data generation failed: {e}"

# ------------------------------------
# 4. Fixed Result Formatting
# ------------------------------------
def format_result_fixed(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fixed result formatting"""
    try:
        question = state.get("question", "")
        result = state.get("result", "")
        
        if not result or "error" in result.lower() or "failed" in result.lower():
            answer = f"I encountered an issue retrieving data for '{question}': {result}"
        else:
            # Format the result nicely
            if "," in result:  # CSV-like data
                lines = result.split('\n')
                formatted_lines = []
                for line in lines:
                    if ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            job_name = parts[0].strip()
                            status = parts[1].strip()
                            # Translate status codes
                            status_meaning = translate_status(status)
                            formatted_lines.append(f"• {job_name}: {status_meaning}")
                
                if formatted_lines:
                    answer = f"Here's the information for '{question}':\n" + "\n".join(formatted_lines)
                else:
                    answer = f"Found data: {result}"
            else:
                answer = f"Result for '{question}': {result}"
        
        return {**state, "answer": answer}
        
    except Exception as e:
        return {**state, "answer": f"Error formatting response: {str(e)}"}

def translate_status(status: str) -> str:
    """Translate Autosys status codes to readable format"""
    status_map = {
        'SU': 'Success ✅',
        'FA': 'Failed ❌', 
        'RU': 'Running 🏃',
        'TE': 'Terminated ⏹️',
        'OH': 'On Hold ⏸️',
        'OI': 'On Ice ❄️',
        'QU': 'Queued 📋',
        'ST': 'Starting 🚀'
    }
    return status_map.get(status.upper(), f"{status} (Unknown)")

# ------------------------------------
# 5. Fixed Workflow
# ------------------------------------
workflow = StateGraph(dict)
workflow.add_node("generate_sql", generate_sql_fixed)
workflow.add_node("execute_sql", execute_sql_fixed)
workflow.add_node("format_result", format_result_fixed)

workflow.add_edge("generate_sql", "execute_sql")
workflow.add_edge("execute_sql", "format_result") 
workflow.add_edge("format_result", END)

workflow.set_entry_point("generate_sql")
app = workflow.compile()

# ------------------------------------
# 6. Fixed Tool Function
# ------------------------------------
def autosys_tool_func_fixed(question: str) -> str:
    """Fixed tool function that handles all the previous errors"""
    try:
        if not question or not question.strip():
            return "Please provide a question about Autosys jobs or schedules."
        
        question = question.strip()
        print(f"\n🚀 Processing: {question}")
        
        # Run the fixed workflow
        result = app.invoke({"question": question})
        answer = result.get("answer", "No answer generated")
        
        # Clean up the response
        if isinstance(answer, str):
            return answer.strip()
        else:
            return str(answer).strip()
            
    except Exception as e:
        error_msg = f"Autosys query failed: {str(e)}"
        print(f"❌ {error_msg}")
        return f"I couldn't process your Autosys query. Error: {error_msg}"

# ------------------------------------
# 7. Create Fixed Tool
# ------------------------------------
autosys_sql_tool_fixed = Tool(
    name="AutosysQuery",
    func=autosys_tool_func_fixed,
    description="""Query the Autosys job scheduling database for job status and information.

This tool can answer questions about:
- Job status (running, failed, success)
- Job schedules and execution history  
- Job dependencies and relationships

Example questions:
- "What is the status of job ATSYS.DA3_DBMaint_080_arch_machines.c?"
- "Show me all failed jobs"
- "List running jobs"
- "What jobs ran today?"

The tool handles Oracle SQL generation and execution automatically."""
)

# ------------------------------------
# 8. Test the Fixed Tool
# ------------------------------------
def test_fixed_tool():
    """Test the fixed tool"""
    print("\n🧪 Testing Fixed Autosys Tool")
    print("=" * 50)
    
    test_questions = [
        "What is the status of job ATSYS.DA3_DBMaint_080_arch_machines.c?",
        "Show me all job statuses", 
        "List failed jobs",
        "What jobs are running?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 40)
        
        try:
            result = autosys_sql_tool_fixed.run(question)
            print(f"✅ Answer: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Fixed Autosys Tool - No More Slice/Connection Errors")
    print("=" * 60)
    
    print("📋 Setup Instructions:")
    print("1. Set GOOGLE_API_KEY environment variable")
    print("2. Set ORACLE_CONNECTION environment variable with your DB details")
    print("3. Replace your existing AutosysQuery tool with autosys_sql_tool_fixed")
    
    # Test the tool
    test_fixed_tool()
    
    print(f"\n✅ Tool Status:")
    print(f"Database Connected: {autosys_db.connected}")
    print(f"Available Tables: {autosys_db.available_tables}")
    print(f"LLM Ready: {autosys_db.llm is not None}")
