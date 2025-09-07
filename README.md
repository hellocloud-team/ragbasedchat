
   # ------------------------------------
# 1. Fixed extract_last_ai_message function (REPLACE YOUR EXISTING ONE)
# ------------------------------------
def extract_last_ai_message(result):
    """Extract the last AI message from agent response - handles multiple response types"""
    try:
        # Handle different types of agent responses
        
        # Case 1: Direct string response
        if isinstance(result, str):
            return result.strip()
        
        # Case 2: Dictionary with 'output' key (most common with agents)
        if isinstance(result, dict):
            # Try common output keys
            for key in ['output', 'result', 'answer', 'content', 'text']:
                if key in result and result[key]:
                    return str(result[key]).strip()
            
            # If no standard keys, try to get any meaningful string value
            for key, value in result.items():
                if isinstance(value, str) and len(value.strip()) > 0 and key != 'input':
                    return value.strip()
        
        # Case 3: AgentFinish object (common with LangChain agents)
        if hasattr(result, 'return_values'):
            if isinstance(result.return_values, dict):
                output = result.return_values.get('output', '')
                if output:
                    return str(output).strip()
            return str(result.return_values).strip()
        
        # Case 4: List of messages
        if isinstance(result, list) and len(result) > 0:
            last_item = result[-1]
            if hasattr(last_item, 'content'):
                return last_item.content.strip()
            elif isinstance(last_item, dict) and 'content' in last_item:
                return last_item['content'].strip()
            return str(last_item).strip()
        
        # Case 5: Object with content attribute
        if hasattr(result, 'content'):
            return result.content.strip()
        
        # Case 6: Try to convert to string
        result_str = str(result).strip()
        if result_str and result_str not in ['None', '', 'null']:
            # Clean up common JSON artifacts
            if result_str.startswith("{'output':") or result_str.startswith('{"output":'):
                try:
                    import re
                    match = re.search(r"'output':\s*'([^']+)'", result_str)
                    if not match:
                        match = re.search(r'"output":\s*"([^"]+)"', result_str)
                    if match:
                        return match.group(1)
                except:
                    pass
            return result_str
        
        # Fallback
        return "I couldn't generate a proper response. Please try again."
        
    except Exception as e:
        print(f"❌ Error extracting AI message: {e}")
        return f"Error processing response: {str(e)}"

# ------------------------------------
# 2. Your FIXED get_chat_response function (REPLACE YOUR EXISTING ONE)
# ------------------------------------
def get_chat_response(message: str, session_id: str) -> str:
    """Fixed version of your get_chat_response function"""
    try:
        if not message or not message.strip():
            return "Please provide a question about the database."
        
        message = message.strip()
        print(f"🔍 Processing: {message}")
        
        # Your existing config structure
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            # Try the main agent invoke (your existing approach)
            response = sql_agent.invoke(
                {"input": message},
                config=config
            )
            
            print(f"🔍 Agent response type: {type(response)}")
            print(f"🔍 Agent response content: {response}")
            
            # Use the enhanced extraction function
            output = extract_last_ai_message(response)
            
            if output and len(output.strip()) > 0:
                return output.rstrip('\n')  # Remove trailing newlines
            else:
                return handle_empty_agent_response(response, message)
                
        except Exception as agent_error:
            print(f"⚠️ Agent invoke failed: {agent_error}")
            return handle_agent_error(message, str(agent_error))
        
    except Exception as e:
        error_msg = f"Chat response error: {str(e)}"
        print(f"❌ {error_msg}")
        return f"I encountered an error: {str(e)}"

# ------------------------------------
# 3. Helper functions for your chat response
# ------------------------------------
def handle_empty_agent_response(response, message: str) -> str:
    """Handle when agent returns empty or None response"""
    try:
        print(f"🔄 Handling empty response for: {message}")
        
        # Try to extract from response attributes
        if hasattr(response, '__dict__'):
            for attr_name, attr_value in response.__dict__.items():
                if attr_value and isinstance(attr_value, str) and len(attr_value.strip()) > 0:
                    return attr_value.strip()
        
        # Try direct tool access if available
        if hasattr(sql_agent, 'tools') and len(sql_agent.tools) > 0:
            try:
                first_tool = sql_agent.tools[0]
                if hasattr(first_tool, 'run'):
                    tool_response = first_tool.run(message)
                    return f"Tool response: {tool_response}"
            except Exception as tool_error:
                print(f"⚠️ Direct tool access failed: {tool_error}")
        
        return f"No response generated for: '{message}'. Please try rephrasing your question."
        
    except Exception as e:
        return f"Error handling empty response: {str(e)}"

def handle_agent_error(message: str, error_str: str) -> str:
    """Handle agent execution errors"""
    try:
        # Check for specific error types and provide helpful messages
        if "parsing" in error_str.lower():
            return "I had trouble understanding the response format. Please try a simpler question."
        elif "timeout" in error_str.lower():
            return "The query took too long. Please try a more specific question."
        elif "connection" in error_str.lower():
            return "Database connection issue. Please try again in a moment."
        elif "permission" in error_str.lower():
            return "Database permission error. Please check your access rights."
        else:
            return f"I encountered an issue processing '{message}'. Please try again or rephrase your question."
    except:
        return "I'm experiencing technical difficulties. Please try again."

# ------------------------------------
# 4. Alternative simple version (if the above still has issues)
# ------------------------------------
def get_chat_response_simple(message: str, session_id: str) -> str:
    """Simple version that bypasses complex parsing"""
    try:
        print(f"📝 Simple processing: {message}")
        
        config = {"configurable": {"thread_id": session_id}}
        
        # Try multiple approaches
        approaches = [
            # Approach 1: Standard invoke
            lambda: sql_agent.invoke({"input": message}, config=config),
            # Approach 2: Direct run (if available)
            lambda: sql_agent.run(message) if hasattr(sql_agent, 'run') else None,
            # Approach 3: Call directly (if callable)
            lambda: sql_agent({"input": message}) if callable(sql_agent) else None
        ]
        
        for i, approach in enumerate(approaches, 1):
            try:
                print(f"🔍 Trying approach {i}")
                result = approach()
                
                if result is not None:
                    extracted = extract_last_ai_message(result)
                    if extracted and len(extracted.strip()) > 5:
                        return extracted
                        
            except Exception as approach_error:
                print(f"⚠️ Approach {i} failed: {approach_error}")
                continue
        
        return "I couldn't generate a proper response. Please check the agent configuration."
        
    except Exception as e:
        return f"Simple chat error: {str(e)}"

# ------------------------------------
# 5. Debug function to understand your agent better
# ------------------------------------
def debug_your_agent(test_message: str = "What is the status of job1?"):
    """Debug function to see what your agent actually returns"""
    print(f"\n🔍 DEBUGGING YOUR AGENT")
    print("=" * 50)
    print(f"Test message: {test_message}")
    
    try:
        config = {"configurable": {"thread_id": "debug_session"}}
        response = sql_agent.invoke({"input": test_message}, config=config)
        
        print(f"\n📋 RESPONSE ANALYSIS:")
        print(f"Type: {type(response)}")
        print(f"Content: {repr(response)}")
        
        if hasattr(response, '__dict__'):
            print(f"\nAttributes: {list(response.__dict__.keys())}")
            for key, value in response.__dict__.items():
                print(f"  {key}: {type(value)} = {repr(value)}")
        
        if isinstance(response, dict):
            print(f"\nDictionary keys: {list(response.keys())}")
            for key, value in response.items():
                print(f"  {key}: {type(value)} = {repr(value)}")
        
        # Test extraction
        extracted = extract_last_ai_message(response)
        print(f"\n✅ EXTRACTED: '{extracted}'")
        
        return extracted
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        return None

# ------------------------------------
# 6. Quick test function
# ------------------------------------
def test_fixed_chat():
    """Test the fixed chat function"""
    print("\n🧪 TESTING FIXED CHAT FUNCTION")
    print("=" * 50)
    
    test_messages = [
        "What is the status of job1?",
        "Show me all jobs",
        "List failed jobs"
    ]
    
    for msg in test_messages:
        print(f"\n📝 Testing: '{msg}'")
        try:
            # Test the main fixed function
            result = get_chat_response(msg, "test_session")
            print(f"✅ Main function result: {result}")
        except Exception as main_error:
            print(f"❌ Main function failed: {main_error}")
            
            # Try simple version
            try:
                simple_result = get_chat_response_simple(msg, "test_session") 
                print(f"🔄 Simple function result: {simple_result}")
            except Exception as simple_error:
                print(f"❌ Simple function also failed: {simple_error}")

# ------------------------------------
# 7. Instructions for replacing your code
# ------------------------------------
"""
INSTRUCTIONS TO FIX YOUR CODE:

1. REPLACE your existing extract_last_ai_message function with the one above

2. REPLACE your existing get_chat_response function with the fixed version above

3. If you still get errors, try the simple version:
   - Rename your current get_chat_response to get_chat_response_old
   - Use get_chat_response_simple instead

4. For debugging, call:
   debug_your_agent("test message")

5. Test with:
   test_fixed_chat()
"""

if __name__ == "__main__":
    print("🚀 Fixed Chat Functions for Your Code Structure")
    print("Copy the functions above to replace your existing ones.")
    
    # Uncomment these lines to test (after you have sql_agent defined):
    # debug_your_agent()
    # test_fixed_chat()             
                
                
                
                
                
                
                return str(result['result']).strip()
            elif 'answer' in result:
                return str(result['answer']).strip()
            # Try to get any string value from the dict
            for key, value in result.items():
                if isinstance(value, str) and len(value.strip()) > 0:
                    return value.strip()
        
        # Case 3: AgentFinish object
        if hasattr(result, 'return_values'):
            if isinstance(result.return_values, dict):
                return str(result.return_values.get('output', '')).strip()
            return str(result.return_values).strip()
        
        # Case 4: List of messages (from some chat models)
        if isinstance(result, list) and len(result) > 0:
            # Get the last message
            last_item = result[-1]
            if hasattr(last_item, 'content'):
                return last_item.content.strip()
            elif isinstance(last_item, dict) and 'content' in last_item:
                return last_item['content'].strip()
            return str(last_item).strip()
        
        # Case 5: Object with content attribute
        if hasattr(result, 'content'):
            return result.content.strip()
        
        # Case 6: Convert whatever we got to string
        result_str = str(result).strip()
        if result_str and result_str != 'None':
            return result_str
        
        # Fallback
        return "I couldn't generate a proper response. Please try again."
        
    except Exception as e:
        print(f"Error extracting AI message: {e}")
        return f"Error processing response: {str(e)}"

# ------------------------------------
# 2. Enhanced get_chat_response function
# ------------------------------------
def get_chat_response(message: str, session_id: str, sql_agent) -> str:
    """Enhanced chat response function with better error handling"""
    try:
        if not message or not message.strip():
            return "Please provide a question about the database."
        
        message = message.strip()
        print(f"🔍 Processing message: {message}")
        
        # Configuration for thread-safe execution
        config = {
            "configurable": {"thread_id": session_id},
            "max_iterations": 3,
            "handle_parsing_errors": True
        }
        
        try:
            # Method 1: Try the standard invoke
            response = sql_agent.invoke(
                {"input": message},
                config=config
            )
            
            # Extract the actual response content
            output = extract_last_ai_message(response)
            
            if output and len(output.strip()) > 0:
                return output
            else:
                # If we got empty output, try alternative extraction
                return handle_empty_response(response, message, sql_agent)
                
        except Exception as invoke_error:
            print(f"⚠️ Agent invoke failed: {invoke_error}")
            
            # Method 2: Try direct run method (some agents use this)
            try:
                if hasattr(sql_agent, 'run'):
                    direct_response = sql_agent.run(message)
                    return extract_last_ai_message(direct_response)
                else:
                    return fallback_response(message, sql_agent)
                    
            except Exception as run_error:
                print(f"⚠️ Agent run failed: {run_error}")
                return fallback_response(message, sql_agent)
        
    except Exception as e:
        error_msg = f"Chat response error: {str(e)}"
        print(f"❌ {error_msg}")
        return f"I encountered an error processing your request: {str(e)}"

# ------------------------------------
# 3. Helper functions for error handling
# ------------------------------------
def handle_empty_response(response, message: str, sql_agent) -> str:
    """Handle cases where agent returns empty responses"""
    try:
        print(f"🔄 Handling empty response: {type(response)}")
        
        # Try to extract from different response formats
        if hasattr(response, '__dict__'):
            response_dict = response.__dict__
            print(f"Response attributes: {response_dict.keys()}")
            
            # Look for common output fields
            for field in ['output', 'result', 'answer', 'content', 'text']:
                if field in response_dict and response_dict[field]:
                    return str(response_dict[field]).strip()
        
        # Try string conversion
        response_str = str(response)
        if response_str and response_str != 'None' and len(response_str.strip()) > 0:
            return response_str.strip()
        
        return fallback_response(message, sql_agent)
        
    except Exception as e:
        print(f"❌ Empty response handling failed: {e}")
        return fallback_response(message, sql_agent)

def fallback_response(message: str, sql_agent) -> str:
    """Fallback response when agent fails"""
    try:
        # Try to access the tool directly if possible
        if hasattr(sql_agent, 'tools') and len(sql_agent.tools) > 0:
            first_tool = sql_agent.tools[0]
            if hasattr(first_tool, 'run'):
                tool_response = first_tool.run(message)
                return f"Direct tool response: {tool_response}"
        
        # Generic fallback
        return f"I couldn't process your query '{message}' properly. Please try rephrasing your question about the database."
        
    except Exception as e:
        return f"I'm experiencing technical difficulties. Error: {str(e)}"

# ------------------------------------
# 4. Alternative simplified chat function
# ------------------------------------
def simple_chat_response(message: str, sql_agent) -> str:
    """Simplified chat function that avoids complex parsing"""
    try:
        print(f"📝 Simple chat processing: {message}")
        
        # Direct approach - try multiple methods
        methods_to_try = [
            lambda: sql_agent.invoke({"input": message}),
            lambda: sql_agent.run(message) if hasattr(sql_agent, 'run') else None,
            lambda: sql_agent(message) if callable(sql_agent) else None
        ]
        
        for method in methods_to_try:
            try:
                result = method()
                if result is not None:
                    extracted = extract_last_ai_message(result)
                    if extracted and len(extracted.strip()) > 5:
                        return extracted
            except Exception as method_error:
                print(f"⚠️ Method failed: {method_error}")
                continue
        
        return "I couldn't generate a response. Please check your agent configuration."
        
    except Exception as e:
        return f"Chat error: {str(e)}"

# ------------------------------------
# 5. Debug function to understand your agent's response format
# ------------------------------------
def debug_agent_response(message: str, sql_agent):
    """Debug function to understand what your agent returns"""
    print(f"\n🔍 DEBUGGING AGENT RESPONSE FOR: '{message}'")
    print("=" * 60)
    
    try:
        # Try invoke method
        response = sql_agent.invoke({"input": message})
        
        print(f"Response type: {type(response)}")
        print(f"Response content: {response}")
        
        if hasattr(response, '__dict__'):
            print(f"Response attributes: {list(response.__dict__.keys())}")
            for key, value in response.__dict__.items():
                print(f"  {key}: {type(value)} = {value}")
        
        if isinstance(response, dict):
            print("Response is a dictionary:")
            for key, value in response.items():
                print(f"  {key}: {type(value)} = {value}")
        
        # Try extraction
        extracted = extract_last_ai_message(response)
        print(f"\nExtracted message: '{extracted}'")
        
    except Exception as e:
        print(f"Debug failed: {e}")

# ------------------------------------
# 6. Your updated chat functions with error handling
# ------------------------------------
def get_chat_response_robust(message: str, session_id: str, sql_agent) -> str:
    """Robust version of your chat function"""
    
    # First, let's debug the response to understand the format
    if os.getenv('DEBUG_MODE', '').lower() == 'true':
        debug_agent_response(message, sql_agent)
    
    # Use the enhanced chat response
    try:
        response = get_chat_response(message, session_id, sql_agent)
        
        # Additional cleanup for common issues
        response = response.replace('\n\n', '\n').strip()
        
        # Remove any JSON formatting artifacts
        if response.startswith('{') and response.endswith('}'):
            try:
                import json
                parsed = json.loads(response)
                if 'output' in parsed:
                    response = parsed['output']
            except:
                pass  # Keep original response if JSON parsing fails
        
        return response
        
    except Exception as e:
        return simple_chat_response(message, sql_agent)

# ------------------------------------
# 7. Usage example for your chatbot
# ------------------------------------
"""
USAGE IN YOUR CHATBOT:

# Replace your original get_chat_response function with:
def get_chat_response(message: str, session_id: str) -> str:
    return get_chat_response_robust(message, session_id, sql_agent)

# Or use the simple version:
def get_chat_response(message: str, session_id: str) -> str:
    return simple_chat_response(message, sql_agent)

# For debugging, set environment variable:
os.environ['DEBUG_MODE'] = 'true'
"""

# ------------------------------------
# 8. Test your functions
# ------------------------------------
def test_chat_functions(sql_agent):
    """Test the chat functions with your agent"""
    print("\n🧪 Testing Chat Functions")
    print("=" * 50)
    
    test_messages = [
        "What is the status of job1?",
        "Show me all jobs",
        "List failed jobs"
    ]
    
    for i, msg in enumerate(test_messages, 1):
        print(f"\n{i}. Testing: '{msg}'")
        print("-" * 30)
        
        try:
            # Test the robust function
            result = get_chat_response_robust(msg, "test_session", sql_agent)
            print(f"✅ Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
            
            # Try simple function as fallback
            try:
                simple_result = simple_chat_response(msg, sql_agent)
                print(f"🔄 Simple fallback: {simple_result}")
            except Exception as e2:
                print(f"❌ Simple fallback also failed: {e2}")

if __name__ == "__main__":
    print("🚀 Enhanced Chat Functions for SQL Agent")
    print("Replace your existing functions with these enhanced versions.")












≠=======================
format_instructions": """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be AutosysQuery
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
            }
        



import os
from typing import Dict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
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
            
            # Initialize Gemini LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY")  # Make sure this is set
            )
            
            # Initialize attributes first
            self.available_tables = []
            self.schema_info = {}
            self.schema_cache = ""
            
            # Test connection
            test_result = self.db.run("SELECT 1 FROM dual")
            print("✅ Connected to Autosys Oracle database successfully")
            
            # Get Autosys schema information
            self.discover_autosys_schema()
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print("Common fixes:")
            print("1. Install cx_Oracle: pip install cx_Oracle")
            print("2. Install Google AI: pip install langchain-google-genai")
            print("3. Set GOOGLE_API_KEY environment variable")
            print("4. Check Oracle connection string format")
            print("5. Verify Autosys database access permissions")
            raise
    
    def discover_autosys_schema(self):
        """Discover and cache Autosys table structure"""
        try:
            print("\n🔍 Discovering Autosys database schema...")
            
            # Common Autosys tables (adjust based on your version)
            autosys_tables = [
                'JOB', 'JOB_STATUS', 'JOB_RUNS', 'JOB_DEPENDENCY',
                'CALENDAR', 'MACHINE', 'GLOBAL_VARIABLES',
                'BOX_JOB', 'CMD_JOB', 'FILE_WATCHER_JOB'
            ]
            
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

# Initialize database connection with better error handling
# Update with your actual Autosys Oracle connection details
ORACLE_CONNECTION = "oracle+cx_oracle://username:password@hostname:1521/?service_name=ORCL"

# Global variables to handle initialization
autosys_db = None
initialization_error = None

try:
    print("🔄 Initializing Autosys Oracle connection...")
    autosys_db = AutosysOracleDatabase(ORACLE_CONNECTION)
except Exception as e:
    initialization_error = str(e)
    print(f"⚠️  Database initialization failed: {e}")
    print("The tool will work in limited mode. Update ORACLE_CONNECTION and restart.")
    
    # Create a mock database object for testing
    class MockAutosysDB:
        def __init__(self):
            self.available_tables = ['JOB', 'JOB_STATUS', 'CALENDAR']
            self.schema_cache = "Mock schema - update connection string"
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
    
    autosys_db = MockAutosysDB()

# ------------------------------------
# 2. Autosys-Specific SQL Generation
# ------------------------------------
def generate_autosys_sql(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate SQL specifically for Autosys database queries using Gemini"""
    try:
        question = state.get("question", "")
        if not question:
            return {"sql": "", "error": "No question provided", "question": ""}
            
        schema = getattr(autosys_db, 'schema_cache', '')
        available_tables = getattr(autosys_db, 'available_tables', [])
        
        # Gemini works well with structured, clear prompts
        prompt = f"""You are an Oracle SQL expert for Autosys job scheduling database.

TASK: Generate Oracle SQL query for the user question.

CONTEXT:
- Database: Autosys job scheduling system on Oracle
- Available tables: {available_tables}
- Schema info: {schema[:800] if schema else 'Schema not available'}

USER QUESTION: {question}

RULES:
1. Use Oracle SQL syntax only
2. Use ROWNUM instead of LIMIT
3. For job status queries, use JOB table
4. For partial job names, use LIKE with wildcards (%)
5. Common columns: job_name, status, last_start, last_end
6. Return ONLY the SQL query - no explanations

EXAMPLES:
- Job status: SELECT job_name, status FROM JOB WHERE job_name LIKE '%job1%'
- Recent runs: SELECT * FROM (SELECT job_name, status, last_start FROM JOB ORDER BY last_start DESC) WHERE ROWNUM <= 5

SQL Query:"""
        
        try:
            response = autosys_db.llm.invoke(prompt)
            # Gemini response handling
            sql_content = response.content if hasattr(response, 'content') else str(response)
        except Exception as llm_error:
            print(f"❌ Gemini LLM call failed: {llm_error}")
            # Create a basic fallback query
            job_hint = ""
            for word in question.lower().split():
                if 'job' in word or word.isalnum():
                    job_hint = word.replace('job', '')
                    break
            sql_content = f"SELECT job_name, status, last_start, last_end FROM JOB WHERE job_name LIKE '%{job_hint}%' AND ROWNUM <= 10"
        
        # Clean SQL output - Gemini sometimes adds formatting
        sql = sql_content.strip()
        
        # Remove common formatting that Gemini might add
        cleanups = ["```sql", "```oracle", "```", "sql:", "query:", "SQL:", "Query:"]
        for cleanup in cleanups:
            sql = sql.replace(cleanup, "")
        
        sql = sql.strip()
        
        # Remove trailing semicolon for SQLAlchemy
        if sql.endswith(';'):
            sql = sql[:-1]
            
        # Validate basic SQL structure
        if not sql or len(sql.strip()) < 10 or not any(keyword in sql.upper() for keyword in ['SELECT', 'FROM']):
            print("⚠️ Generated SQL seems invalid, using fallback")
            sql = "SELECT job_name, status FROM JOB WHERE ROWNUM <= 10"
        
        print(f"🔍 Gemini generated SQL: {sql}")
        
        return {
            "sql": sql,
            "question": question,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"SQL generation error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "sql": "",
            "error": error_msg,
            "question": state.get("question", "")
        }

def execute_autosys_sql(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute SQL against Autosys Oracle database"""
    try:
        sql = state.get("sql", "")
        question = state.get("question", "")
        error = state.get("error")
        
        if error or not sql:
            return {
                **state,
                "result": f"Cannot execute query: {error or 'No SQL generated'}",
                "needs_retry": False
            }
        
        print(f"⚡ Executing: {sql}")
        
        try:
            result = autosys_db.db.run(sql)
            if not result or str(result).strip() == "":
                result = "No data found"
            
            print(f"📊 Result: {result}")
            return {
                **state,
                "result": str(result),
                "needs_retry": False
            }
            
        except Exception as db_error:
            error_msg = f"Database error: {str(db_error)}"
            print(f"❌ {error_msg}")
            
            return {
                **state,
                "result": error_msg,
                "needs_retry": True,
                "retry_attempted": False
            }
        
    except Exception as e:
        error_msg = f"Execution error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            **state,
            "result": error_msg,
            "needs_retry": False
        }

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
    """Summarize Autosys query results using Gemini"""
    try:
        question = state.get("question", "")
        result = state.get("result", "")
        sql = state.get("sql", "")
        
        if not result or "error" in result.lower():
            return {
                **state, 
                "answer": f"Unable to retrieve data: {result}"
            }
        
        # Handle no data case
        if "no data found" in result.lower():
            answer = f"No Autosys jobs found matching your query: '{question}'"
        else:
            # Try Gemini summarization with simple prompt
            try:
                # Gemini-optimized prompt - more conversational
                prompt = f"""Help explain this Autosys database query result to a user.

User asked: "{question}"
Database returned: {result}

Context: This is from an Autosys job scheduling system where:
- Jobs have statuses like SU (Success), FA (Failure), RU (Running)
- job_name identifies the scheduled job
- Timestamps show when jobs ran

Please give a clear, helpful answer in plain English. Be concise but informative.

Answer:"""
                
                response = autosys_db.llm.invoke(prompt)
                gemini_answer = response.content if hasattr(response, 'content') else str(response)
                
                if gemini_answer and len(gemini_answer.strip()) > 10:
                    answer = gemini_answer.strip()
                else:
                    # Fallback to basic formatting
                    answer = f"Found the following Autosys data for '{question}':\n{result}"
                    
            except Exception as llm_error:
                print(f"⚠️ Gemini summarization failed: {llm_error}")
                # Basic fallback without LLM
                answer = f"Here's what I found for '{question}':\n{result}"
        
        print(f"✅ Final answer: {answer}")
        return {
            **state,
            "answer": answer
        }
        
    except Exception as e:
        error_msg = f"Summarization error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            **state,
            "answer": f"Retrieved data but couldn't format it: {state.get('result', '')}"
        }

def should_retry_autosys_query(state: Dict[str, Any]) -> str:
    """Decide whether to retry failed Autosys queries"""
    needs_retry = state.get("needs_retry", False)
    retry_attempted = state.get("retry_attempted", False)
    
    if needs_retry and not retry_attempted:
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
        if not question or not question.strip():
            return "Please provide a question about Autosys jobs, calendars, or schedules."
        
        question = question.strip()
        print(f"\n🚀 Processing: '{question}'")
        
        # Initialize state with proper structure
        initial_state = {
            "question": question,
            "sql": "",
            "result": "",
            "answer": "",
            "error": None,
            "needs_retry": False,
            "retry_attempted": False
        }
        
        try:
            # Run the workflow
            result = autosys_app.invoke(initial_state)
            
            # Extract answer with fallbacks
            if isinstance(result, dict):
                answer = result.get("answer", "")
                if not answer:
                    answer = result.get("result", "No response generated")
            else:
                answer = str(result)
            
            if not answer or answer.strip() == "":
                answer = "No answer could be generated for your query."
            
            print(f"🎯 Response: {answer}")
            return answer
            
        except Exception as workflow_error:
            error_msg = f"Workflow execution failed: {str(workflow_error)}"
            print(f"❌ {error_msg}")
            return f"Sorry, I couldn't process your query: {error_msg}"
        
    except Exception as e:
        error_msg = f"Tool error: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

# Create the Autosys-specific tool with improved description
autosys_sql_tool = Tool(
    name="AutosysQuery",
    func=autosys_tool_func,
    description="""
    Use this tool to get information from the Autosys job scheduling database.
    
    This tool handles all database connectivity and SQL execution internally.
    Do NOT try to execute SQL queries directly.
    
    Input: A natural language question about Autosys jobs, schedules, or status
    Output: A formatted answer with the requested information
    
    Example inputs:
    - "What is the status of job1?"
    - "Show me failed jobs from yesterday"
    - "List all jobs in the DAILY_BATCH box"
    - "What jobs depend on DATA_EXTRACT_JOB?"
    
    The tool will automatically:
    1. Generate the appropriate Oracle SQL query
    2. Execute it against the Autosys database
    3. Format the results in a readable way
    
    Always use this tool for any Autosys database questions.
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
    """Example of using Autosys tool with an agent - with better configuration"""
    try:
        print("\n🤖 Creating Autosys-enabled Agent...")
        
        # Use a more restrictive agent type to avoid SQL execution attempts
        agent = initialize_agent(
            tools=[autosys_sql_tool],
            llm=ChatGoogleGenerativeAI(
                model="gemini-1.5-pro", 
                temperature=0,
                # Add system instruction to prevent direct SQL execution
                system_instruction="You are an Autosys expert assistant. Always use the AutosysQuery tool for database questions. Never attempt to execute SQL queries directly."
            ),
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,  # This helps with parsing issues
            max_iterations=3,  # Limit iterations to prevent loops
            early_stopping_method="generate"
        )
        
        # Test with simpler, more direct queries first
        simple_queries = [
            "Use the AutosysQuery tool to check the status of job1",
            "What is the current status of any job named job1?",
        ]
        
        for query in simple_queries:
            print(f"\n🔍 Testing: {query}")
            try:
                response = agent.run(query)
                print(f"📊 Response: {response}")
                break  # Success, no need to try more
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")


# ------------------------------------
# 6. Direct Tool Usage (Recommended)
# ------------------------------------
def test_direct_tool_usage():
    """Test the tool directly without agent - more reliable"""
    print("\n🧪 Testing Direct Tool Usage (Recommended)")
    print("=" * 60)
    
    test_queries = [
        "What is the status of job1?",
        "Show me all job statuses",
        "List jobs that failed recently",
        "What jobs are currently running?",
        "Show me job information for any job with 'batch' in the name"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Direct Query: {query}")
        print("-" * 50)
        try:
            # Call the tool directly - this avoids agent parsing issues
            result = autosys_sql_tool.run(query)
            print(f"✅ Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print()

# ------------------------------------
# 7. Alternative Simple Agent
# ------------------------------------
def create_simple_autosys_interface():
    """Create a simple interface that avoids complex agent behavior"""
    
    def simple_autosys_chat(question: str) -> str:
        """Simple chat interface that always uses the tool correctly"""
        try:
            print(f"🔍 Processing: {question}")
            
            # Always route through our tool
            result = autosys_sql_tool.run(question)
            return result
            
        except Exception as e:
            return f"Error processing your question: {str(e)}"
    
    return simple_autosys_chat

# Create the simple interface
simple_autosys_chat = create_simple_autosys_interface()

if __name__ == "__main__":
    print("🚀 Autosys Oracle Database Query Tool (Powered by Gemini)")
    print("=" * 70)
    
    # Setup instructions
    print("📋 SETUP REQUIREMENTS:")
    print("1. Install packages: pip install langchain-google-genai cx_Oracle")
    print("2. Set environment variable: GOOGLE_API_KEY=your_gemini_api_key")
    print("3. Update ORACLE_CONNECTION with your Autosys database details")
    print("   Format: oracle+cx_oracle://user:pass@host:port/?service_name=SERVICE")
    print()
    
    if initialization_error:
        print(f"⚠️  Database not connected: {initialization_error}")
        print("Tool running in mock mode - update connection to use with real data")
        print()
    
    try:
        # Test direct tool usage (recommended)
        test_direct_tool_usage()
        
        print("\n" + "="*60)
        print("💡 USAGE RECOMMENDATIONS:")
        print("="*60)
        print("1. DIRECT USAGE (Most Reliable):")
        print("   result = autosys_sql_tool.run('What is the status of job1?')")
        print()
        print("2. SIMPLE INTERFACE:")
        print("   answer = simple_autosys_chat('Show me failed jobs')")
        print()
        print("3. Agent Usage (if needed):")
        print("   # Use create_autosys_agent() but direct usage is more reliable")
        
        # Test simple interface
        print("\n🧪 Testing Simple Interface:")
        test_question = "What is the status of job1?"
        simple_result = simple_autosys_chat(test_question)
        print(f"Simple interface result: {simple_result}")
        
        # Uncomment to test agent (may have parsing issues)
        # create_autosys_agent()
        
        print("\n✅ Autosys tool ready!")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("Make sure GOOGLE_API_KEY is set and try again")
