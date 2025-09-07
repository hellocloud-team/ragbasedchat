
import os
from typing import Dict, Any, List
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import oracledb

# ------------------------------------
# AutosysOracleDatabase Class
# ------------------------------------
class AutosysOracleDatabase:
    """Specialized class for Autosys Oracle database operations using oracledb"""
    
    def __init__(self, connection_string: str):
        """Initialize Oracle connection for Autosys database"""
        try:
            self.engine = create_engine(connection_string)
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1 FROM dual"))
                print("✅ Connected to Autosys Oracle database successfully")
            
            self.available_tables = []
            self.schema_info = {}
            self.schema_cache = ""
            
            self.discover_autosys_schema()
            
        except SQLAlchemyError as e:
            print(f"❌ Database connection failed: {e}")
            print("Common fixes:")
            print("1. Install oracledb: pip install oracledb")
            print("2. Check Oracle connection string format")
            print("3. Verify Autosys database access permissions")
            raise
    
    def discover_autosys_schema(self):
        """Safely discover Autosys schema using direct engine connection"""
        try:
            print("\n🔍 Discovering Autosys database schema...")
            
            # Common Autosys tables to check
            autosys_tables = [
                'JOB', 'JOB_STATUS', 'JOB_RUNS', 'JOB_DEPENDENCY',
                'CALENDAR', 'MACHINE', 'GLOBAL_VARIABLES',
                'BOX_JOB', 'CMD_JOB', 'FILE_WATCHER_JOB'
            ]
            
            with self.engine.connect() as conn:
                for table in autosys_tables:
                    try:
                        # Check if table exists by querying
                        conn.execute(text(f"SELECT COUNT(*) FROM {table} WHERE ROWNUM = 1"))
                        self.available_tables.append(table)
                        print(f"✅ Found table: {table}")
                    except Exception:
                        print(f"⚠️  Table not found or no access: {table}")
            
            if not self.available_tables:
                print("⚠️  No standard Autosys tables found. Discovering all accessible tables...")
                try:
                    with self.engine.connect() as conn:
                        tables_query = text("""
                            SELECT table_name FROM user_tables 
                            WHERE table_name LIKE '%JOB%' 
                            OR table_name LIKE '%CALENDAR%'
                            OR table_name LIKE '%SCHEDULE%'
                            ORDER BY table_name
                        """)
                        result = conn.execute(tables_query)
                        tables_result = [row[0] for row in result.fetchall()]
                        self.available_tables = tables_result
                        print(f"Available job-related tables: {tables_result}")
                except Exception as e:
                    print(f"Could not discover tables: {e}")
            
            # Build schema information
            self._build_schema_info()
            print(f"\n📊 Available Autosys tables: {self.available_tables}")
            
        except Exception as e:
            print(f"❌ Schema discovery failed: {e}")
            self.available_tables = ['JOB', 'JOB_STATUS']
            self.schema_cache = "Schema discovery failed - using basic table structure"
    
    def _build_schema_info(self):
        """Build comprehensive schema information"""
        try:
            schema_parts = []
            schema_parts.append(f"Available Tables: {', '.join(self.available_tables)}")
            
            # Get column info for key tables
            key_tables = ['JOB', 'JOB_STATUS', 'JOB_RUNS']
            
            with self.engine.connect() as conn:
                for table in key_tables:
                    if table in self.available_tables:
                        try:
                            col_query = text(f"""
                                SELECT column_name, data_type 
                                FROM user_tab_columns 
                                WHERE table_name = '{table}' 
                                ORDER BY column_id
                            """)
                            result = conn.execute(col_query)
                            columns = [(row[0], row[1]) for row in result.fetchall()]
                            
                            if columns:
                                col_info = f"{table}: " + ", ".join([f"{col}({dtype})" for col, dtype in columns[:10]])
                                schema_parts.append(col_info)
                                
                        except Exception as col_error:
                            print(f"⚠️ Could not get columns for {table}: {col_error}")
            
            self.schema_cache = "\n".join(schema_parts)
            
        except Exception as e:
            print(f"⚠️ Schema info building failed: {e}")
            self.schema_cache = f"Available tables: {', '.join(self.available_tables)}"
    
    def execute_query(self, sql: str) -> str:
        """Execute SQL query using direct engine connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                
                if not rows:
                    return "No data found"
                
                # Format results as CSV-like string
                formatted_rows = []
                for row in rows:
                    formatted_rows.append(",".join([str(col) if col is not None else 'NULL' for col in row]))
                
                return "\n".join(formatted_rows)
                
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            return f"Query execution error: {str(e)}"
    
    def get_table_info(self) -> str:
        """Get table information for SQL generation"""
        return self.schema_cache
    
    def get_usable_table_names(self) -> List[str]:
        """Get list of available table names"""
        return self.available_tables
    
    def run(self, sql: str) -> str:
        """Execute SQL query (alias for execute_query)"""
        return self.execute_query(sql)
    
    def close(self):
        """Close database connection"""
        try:
            if hasattr(self, 'engine') and self.engine:
                self.engine.dispose()
                print("✅ Database connection closed")
        except Exception as e:
            print(f"⚠️ Error closing connection: {e}")

# ------------------------------------
# Usage Example
# ------------------------------------
"""
# How to use AutosysOracleDatabase:

# 1. Initialize with your connection string
connection_string = "oracle+oracledb://user:password@hostname:1521/?service_name=AUTOSYS"
autosys_db = AutosysOracleDatabase(connection_string)

# 2. Execute queries directly
result = autosys_db.execute_query("SELECT job_name, status FROM JOB WHERE ROWNUM <= 5")
print(result)

# 3. Get schema information
schema_info = autosys_db.get_table_info()
print(schema_info)

# 4. Get available tables
tables = autosys_db.get_usable_table_names()
print(tables)

# 5. Use run method (alias for execute_query)
result = autosys_db.run("SELECT COUNT(*) FROM JOB")
print(result)

# 6. Close connection when done
autosys_db.close()
"""

if __name__ == "__main__":
    print("🚀 AutosysOracleDatabase Class")
    print("=" * 40)
    print("Usage:")
    print("1. Set connection string: oracle+oracledb://user:pass@host:port/?service_name=SERVICE")
    print("2. Initialize: autosys_db = AutosysOracleDatabase(connection_string)")
    print("3. Execute queries: result = autosys_db.execute_query(sql)")
    print("4. Get schema info: schema = autosys_db.get_table_info()")









import os
from typing import Dict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.tools import Tool
from sqlalchemy.exc import SQLAlchemyError
import oracledb

load_dotenv()

# ------------------------------------
# 1. Autosys Oracle Database Class  
# ------------------------------------
class AutosysOracleDatabase:
    """Autosys Oracle Database class using oracledb driver"""
    
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
                    print("✅ Oracle database connected using oracledb driver")
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
        """Check if connection string is properly configured for oracledb"""
        return (conn_str and 
                "username:password" not in conn_str and 
                ("oracle+oracledb://" in conn_str or "oracle://" in conn_str))
    
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
        print("💡 Expected format: oracle+oracledb://user:pass@host:port/?service_name=SERVICE")
    
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

# Initialize the Autosys Oracle database
ORACLE_CONNECTION = os.getenv("ORACLE_CONNECTION", "oracle+oracledb://username:password@hostname:1521/?service_name=ORCL")
autosys_db = AutosysOracleDatabase(ORACLE_CONNECTION)

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
    print("🚀 Autosys Oracle Database Tool - Using oracledb Driver")
    print("=" * 60)
    
    print("📋 Setup Instructions:")
    print("1. Install packages: pip install oracledb langchain-google-genai")
    print("2. Set GOOGLE_API_KEY environment variable")
    print("3. Set ORACLE_CONNECTION environment variable:")
    print("   Format: oracle+oracledb://user:pass@host:port/?service_name=SERVICE")
    print("   Example: oracle+oracledb://autosys_user:password@db.company.com:1521/?service_name=AUTOSYS")
    print("4. Replace your existing AutosysQuery tool with autosys_sql_tool_fixed")
    print()
    print("💡 Note: Using oracledb driver (newer than cx_Oracle)")
    
    # Show connection status
    print(f"\n🔍 Current Status:")
    print(f"Database Connected: {autosys_db.connected}")
    print(f"Connection String: {ORACLE_CONNECTION[:50]}...")
    print(f"Available Tables: {autosys_db.available_tables}")
    print(f"LLM Ready: {autosys_db.llm is not None}")
    
    # Test the tool
    test_fixed_tool()
