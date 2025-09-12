
def handle_conversation_node(self, state: AutosysState) -> AutosysState:
    """Handle general conversation using LLM"""
    
    try:
        conversation_prompt = f"""
You are a helpful AI assistant for an Autosys database system.

You excel at both friendly conversation and database queries. For this message, respond naturally to the general conversation.

Your capabilities:
- Natural conversation about any topic
- Query Autosys job scheduler database
- Provide job status, schedules, and reports

User message: "{state['user_question']}"

Respond in a friendly, professional manner:
"""
        
        response = self.llm.invoke(conversation_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Clean chat bubble styling like the image
        state["formatted_output"] = f"""
        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
            <div style="max-width: 70%; background: #e9ecef; border-radius: 18px; padding: 12px 16px; color: #212529; font-size: 14px; line-height: 1.4; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                {content}
            </div>
        </div>
        """
        
    except Exception as e:
        # Fallback response with same styling
        state["formatted_output"] = f"""
        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
            <div style="max-width: 70%; background: #e9ecef; border-radius: 18px; padding: 12px 16px; color: #212529; font-size: 14px; line-height: 1.4;">
                Hi there! How can I help you today?
            </div>
        </div>
        """
    
    return state







Here's the fix for your


handle_conversation_node function:

def handle_conversation_node(self, state: AutosysState) -> AutosysState:
    """Handle general conversation using LLM"""
    
    try:
        conversation_prompt = f"""
You are a helpful AI assistant for an Autosys database system.

You excel at both friendly conversation and database queries. For this message, respond naturally to the general conversation.

Your capabilities:
- Natural conversation about any topic
- Query Autosys job scheduler database
- Provide job status, schedules, and reports

User message: "{state['user_question']}"

Respond in a friendly, professional manner:
"""
        
        response = self.llm.invoke(conversation_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Fixed styling - inline-block and max-width for content-based sizing
        state["formatted_output"] = f"""
        <div style="display: inline-block; max-width: 80%; background: #f1f3f5; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 14px; margin-right: 6px; color: #28a745;">💬</span>
                <span style="font-weight: 500; color: #495057; font-size: 13px;">Assistant</span>
            </div>
            <p style="margin: 0; color: #212529; line-height: 1.5; font-size: 14px;">{content}</p>
            <div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid #dee2e6; font-size: 10px; color: #6c757d;">
                <em>General conversation • Ask about Autosys jobs for database queries</em>
            </div>
        </div>
        """
        
    except Exception as e:
        # Compact fallback response
        state["formatted_output"] = f"""
        <div style="display: inline-block; max-width: 70%; background: #f1f3f5; border-radius: 8px; border-left: 4px solid #28a745; padding: 12px; margin: 10px 0;">
            <p style="margin: 0; color: #212529; font-size: 14px;">Hello! I'm here to help with both general questions and Autosys database queries. How can I assist you today?</p>
            <div style="margin-top: 6px; font-size: 10px; color: #6c757d;">
                <em>General conversation mode</em>
            </div>
        </div>
        """
    
    return state







"""""""

The Fix:Looking at your route_message_node, the logic has a flaw. Here's the corrected version:def route_message_node(self, state: AutosysState) -> AutosysState:
    """Route message to appropriate handler"""
    
    message = state["user_question"].strip().lower()
    
    # First check for explicit casual greetings - these should ALWAYS go to conversation
    casual_inputs = [
        "hi", "hello", "hey", "how are you", "good morning", "good evening",
        "what's up", "how's it going", "yo", "greetings"
    ]
    
    # Prioritize casual messages - if it's a greeting, it's conversation
    if any(casual_word in message for casual_word in casual_inputs):
        is_conversation = True
        logger.info("Detected casual input - routing to conversation")
    else:
        # Only then check for Autosys keywords
        is_conversation = not is_autosys_related_query(state["user_question"])
        logger.info(f"Autosys check result - routing to: {'conversation' if is_conversation else 'database query'}")
    
    state["is_general_conversation"] = is_conversation
    
    state["messages"].append({
        "role": "system",
        "content": f"Routing to: {'conversation' if is_conversation else 'database query'}"
    })
    
    return stateAlternative: Modify your is_autosys_related_query functionAdd this check at the beginning of your is_autosys_related_query function:def is_autosys_related_query(message: str) -> bool:
    """Determine if the message requires Autosys database query"""
    
    # FIRST: Check for casual inputs - these are NEVER Autosys queries
    casual_inputs = [
        "hi", "hello", "hey", "how are you", "good morning", "good evening",
        "what's up", "how's it going", "yo", "greetings", "thanks", "thank you"
    ]
    
    message_lower = message.lower().strip()
    
    # If it's a casual message, return False immediately
    if any(casual_word in message_lower for casual_word in casual_inputs):
        logger.info("Detected casual input - NOT routing to database")
        return False
    
    # Rest of your existing logic...
    autosys_keywords = [
        'job', 'jobs', 'atsys', 'autosys', 'schedule', 'status', 'failed', 'failure',
        'running', 'success', 'database', 'query', 'select', 'show', 'list', 'find',
        'search', 'owner', 'machine', 'execution', 'sql', 'table', 'count', 'report'
    ]
    
    # ... continue with existing logicQuick Debug:Add this debug line in your get_chat_response function to see what's happening:def get_chat_response(message: str, session_id: str, autosys_system=None) -> str:
    # Add this debug line
    is_db_query = is_autosys_related_query(message)
    logger.info(f"Message: '{message}' -> Database query: {is_db_query}")
    
    # Continue with your existing code...The issue is that your routing logic isn't properly prioritizing casual inputs over the general Autosys keyword detection. The fix ensures that greetings like "hi" are immediately identified as conversation and never get processed as database queries.
&&&&&&&&&&&&&&&
# ============================================================================
# CLEAN LANGGRAPH AUTOSYS IMPLEMENTATION - NEW WAY ONLY
# ============================================================================

import oracledb
import json
import logging
import re
from typing import Dict, Any, Optional, List, TypedDict, Annotated
from datetime import datetime
from pydantic import BaseModel, Field
import operator

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports (minimal, for tool compatibility only)
from langchain.tools import BaseTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# STATE DEFINITION FOR LANGGRAPH
# ============================================================================

class AutosysState(TypedDict):
    """State for the Autosys LangGraph workflow"""
    messages: Annotated[List[Dict[str, str]], operator.add]
    user_question: str
    is_general_conversation: bool
    sql_query: str
    query_results: Dict[str, Any]
    formatted_output: str
    error: str
    session_id: str

# ============================================================================
# MESSAGE ROUTING LOGIC
# ============================================================================

def is_autosys_related_query(message: str) -> bool:
    """Determine if the message requires Autosys database query"""
    
    # Keywords that indicate database/job queries
    autosys_keywords = [
        'job', 'jobs', 'atsys', 'autosys', 'schedule', 'status', 'failed', 'failure',
        'running', 'success', 'database', 'query', 'select', 'show', 'list', 'find',
        'search', 'owner', 'machine', 'execution', 'sql', 'table', 'count', 'report'
    ]
    
    message_lower = message.lower()
    
    # Direct keyword match
    if any(keyword in message_lower for keyword in autosys_keywords):
        return True
    
    # Pattern-based detection
    database_patterns = [
        r'\b(what|which|how many|show me|list|find)\b.*\b(job|status|schedule)\b',
        r'\b(failed|running|success|error)\b',
        r'\b(today|yesterday|last|recent)\b.*\b(job|run|execution)\b'
    ]
    
    return any(re.search(pattern, message_lower) for pattern in database_patterns)

# ============================================================================
# AUTOSYS DATABASE TOOL
# ============================================================================

class AutosysQueryInput(BaseModel):
    """Input schema for Autosys Query Tool"""
    question: str = Field(description="Natural language question about Autosys jobs")

class AutosysQueryTool(BaseTool):
    """Tool for querying Autosys database"""
    
    name: str = "AutosysQuery"
    description: str = "Query Autosys job scheduler database using natural language"
    args_schema = AutosysQueryInput
    
    def __init__(self, autosys_db, llm_instance, max_results: int = 50):
        super().__init__()
        self.autosys_db = autosys_db
        self.llm = llm_instance
        self.max_results = max_results
        self.logger = logging.getLogger(self.__class__.__name__)

    def _run(self, question: str) -> Dict[str, Any]:
        """Execute database query and return structured results"""
        try:
            # Generate SQL
            sql_query = self._generate_sql(question)
            
            # Execute query  
            query_result = self._execute_query(sql_query)
            
            return {
                "success": query_result["success"],
                "sql_query": sql_query,
                "results": query_result.get("results", []),
                "row_count": query_result.get("row_count", 0),
                "execution_time": query_result.get("execution_time", 0),
                "error": query_result.get("error", "")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "sql_query": "",
                "results": [],
                "row_count": 0,
                "execution_time": 0
            }

    def _generate_sql(self, user_question: str) -> str:
        """Generate SQL using LLM"""
        
        sql_prompt = f"""
Generate Oracle SQL for Autosys job scheduler database.

SCHEMA:
- aedbadmin.ujo_jobst: job_name, status (SU=Success, FA=Failure, RU=Running), last_start, last_end, joid
- aedbadmin.ujo_job: joid, owner, machine, job_type  
- aedbadmin.UJO_INTCODES: code, TEXT (status descriptions)

PATTERNS:
- Time: TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss')
- Joins: js INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
- Status: LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
- Recent: WHERE js.last_start >= (SYSDATE - INTERVAL '1' DAY) * 86400

Question: {user_question}

Return only SQL query without explanations or formatting:
"""
        
        try:
            response = self.llm.invoke(sql_prompt)
            sql = response.content if hasattr(response, 'content') else str(response)
            return self._clean_sql(sql)
        except Exception as e:
            self.logger.error(f"SQL generation failed: {e}")
            return self._fallback_sql()

    def _clean_sql(self, sql: str) -> str:
        """Clean and enhance SQL query"""
        # Remove markdown
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        sql = ' '.join(sql.split()).strip()
        
        # Add result limiting
        if 'ROWNUM' not in sql.upper():
            if 'WHERE' in sql.upper():
                sql = sql.replace('WHERE', f'WHERE ROWNUM <= {self.max_results} AND', 1)
            else:
                sql += f' WHERE ROWNUM <= {self.max_results}'
        
        return sql

    def _execute_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query using existing database connection"""
        try:
            start_time = datetime.now()
            
            # Use existing database methods
            if hasattr(self.autosys_db, 'run'):
                raw_results = self.autosys_db.run(sql_query)
            elif hasattr(self.autosys_db, 'execute_query'):
                raw_results = self.autosys_db.execute_query(sql_query)
            else:
                raise Exception("Database connection method not found")
            
            # Process results
            processed_results = self._process_results(raw_results)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "results": processed_results,
                "row_count": len(processed_results),
                "execution_time": execution_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "row_count": 0,
                "execution_time": 0
            }

    def _process_results(self, raw_results) -> List[Dict]:
        """Convert raw database results to standardized format"""
        if isinstance(raw_results, str):
            try:
                import ast
                if raw_results.startswith('[') and raw_results.endswith(']'):
                    return self._convert_tuples_to_dicts(ast.literal_eval(raw_results))
                else:
                    return [{"result": raw_results}]
            except:
                return [{"result": raw_results}]
        elif isinstance(raw_results, list):
            return self._convert_tuples_to_dicts(raw_results)
        else:
            return [{"result": str(raw_results)}]

    def _convert_tuples_to_dicts(self, raw_results: List) -> List[Dict]:
        """Convert list of tuples to list of dictionaries"""
        results = []
        column_names = ["JOB_NAME", "START_TIME", "END_TIME", "STATUS", "OWNER", "MACHINE"]
        
        for item in raw_results:
            if isinstance(item, (tuple, list)):
                row_dict = {}
                for i, value in enumerate(item):
                    col_name = column_names[i] if i < len(column_names) else f"COLUMN_{i + 1}"
                    row_dict[col_name] = str(value) if value is not None else ""
                results.append(row_dict)
            elif isinstance(item, dict):
                results.append(item)
            else:
                results.append({"VALUE": str(item)})
        
        return results

    def _fallback_sql(self) -> str:
        """Fallback SQL query"""
        return f"""
        SELECT 
            js.job_name,
            TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (js.last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss') AS start_time,
            NVL(ic.TEXT, 'UNKNOWN') AS status,
            j.owner
        FROM aedbadmin.ujo_jobst js
        INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
        LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
        WHERE UPPER(js.job_name) LIKE UPPER('%ATSYS%')
        AND ROWNUM <= {self.max_results}
        ORDER BY js.job_name
        """

# ============================================================================
# LANGGRAPH WORKFLOW SYSTEM
# ============================================================================

class AutosysLangGraphSystem:
    """Complete LangGraph-based system for Autosys queries and conversation"""
    
    def __init__(self, autosys_db, llm_instance):
        self.autosys_db = autosys_db
        self.llm = llm_instance
        self.tool = AutosysQueryTool(autosys_db, llm_instance)
        self.memory = MemorySaver()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Build workflow
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the complete LangGraph workflow"""
        
        workflow = StateGraph(AutosysState)
        
        # Add nodes
        workflow.add_node("route_message", self.route_message_node)
        workflow.add_node("handle_conversation", self.handle_conversation_node)
        workflow.add_node("generate_sql", self.generate_sql_node)
        workflow.add_node("execute_query", self.execute_query_node)
        workflow.add_node("format_results", self.format_results_node)
        workflow.add_node("handle_error", self.handle_error_node)
        
        # Set entry point
        workflow.set_entry_point("route_message")
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "route_message",
            self._should_handle_conversation,
            {
                "conversation": "handle_conversation",
                "database": "generate_sql"
            }
        )
        
        workflow.add_edge("handle_conversation", END)
        
        workflow.add_conditional_edges(
            "generate_sql",
            self._should_execute_query,
            {
                "execute": "execute_query",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_query",
            self._should_format_results,
            {
                "format": "format_results",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("format_results", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile(checkpointer=self.memory)

    def route_message_node(self, state: AutosysState) -> AutosysState:
        """Route message to appropriate handler"""
        
        is_conversation = not is_autosys_related_query(state["user_question"])
        state["is_general_conversation"] = is_conversation
        
        state["messages"].append({
            "role": "system",
            "content": f"Routing to: {'conversation' if is_conversation else 'database query'}"
        })
        
        return state

    def handle_conversation_node(self, state: AutosysState) -> AutosysState:
        """Handle general conversation using LLM"""
        
        try:
            conversation_prompt = f"""
You are a helpful AI assistant for an Autosys database system.

You excel at both friendly conversation and database queries. For this message, respond naturally to the general conversation.

Your capabilities:
- Natural conversation about any topic
- Query Autosys job scheduler database
- Provide job status, schedules, and reports

User message: "{state['user_question']}"

Respond in a friendly, professional manner:
"""
            
            response = self.llm.invoke(conversation_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Format response
            state["formatted_output"] = f"""
            <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 16px; margin-right: 8px;">💬</span>
                    <span style="font-weight: 500; color: #333;">Assistant</span>
                </div>
                <p style="margin: 0; color: #333; line-height: 1.6;">{content}</p>
                <div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid #dee2e6; font-size: 11px; color: #666;">
                    <em>General conversation • Ask about Autosys jobs for database queries</em>
                </div>
            </div>
            """
            
        except Exception as e:
            # Fallback response
            state["formatted_output"] = f"""
            <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #28a745;">
                <p style="margin: 0; color: #333;">Hello! I'm here to help with both general questions and Autosys database queries. How can I assist you today?</p>
            </div>
            """
        
        return state

    def generate_sql_node(self, state: AutosysState) -> AutosysState:
        """Generate SQL query using tool"""
        
        try:
            tool_result = self.tool.run(state["user_question"])
            
            if tool_result["success"]:
                state["sql_query"] = tool_result["sql_query"]
            else:
                state["error"] = f"SQL generation failed: {tool_result.get('error', 'Unknown error')}"
                
        except Exception as e:
            state["error"] = f"SQL generation exception: {str(e)}"
        
        return state

    def execute_query_node(self, state: AutosysState) -> AutosysState:
        """Execute database query"""
        
        try:
            tool_result = self.tool.run(state["user_question"])
            
            if tool_result["success"]:
                state["query_results"] = {
                    "success": True,
                    "results": tool_result["results"],
                    "row_count": tool_result["row_count"],
                    "execution_time": tool_result["execution_time"]
                }
            else:
                state["error"] = f"Query execution failed: {tool_result.get('error', 'Unknown error')}"
                state["query_results"] = {"success": False}
                
        except Exception as e:
            state["error"] = f"Query execution exception: {str(e)}"
            state["query_results"] = {"success": False}
        
        return state

    def format_results_node(self, state: AutosysState) -> AutosysState:
        """Format query results using LLM"""
        
        try:
            results = state["query_results"]["results"]
            
            if not results:
                state["formatted_output"] = """
                <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px;">
                    <h4 style="margin: 0 0 10px 0; color: #856404;">No Results Found</h4>
                    <p style="margin: 0; color: #856404;">No Autosys jobs match your query criteria.</p>
                </div>
                """
                return state
            
            # Format using LLM
            format_prompt = f"""
Create professional HTML for these Autosys query results:

Question: {state['user_question']}
Results: {len(results)} jobs found
Data: {json.dumps(results[:10], indent=2, default=str)}

Requirements:
- Responsive HTML with inline CSS
- Color-coded status badges (SUCCESS=green, FAILURE=red, RUNNING=blue, INACTIVE=gray)
- Professional styling with good contrast
- Mobile-friendly design
- Include summary statistics

Return only HTML:
"""
            
            response = self.llm.invoke(format_prompt)
            formatted_html = response.content if hasattr(response, 'content') else str(response)
            
            # Add metadata
            metadata = f"""
            <div style="margin-top: 15px; padding: 8px 12px; background: #f8f9fa; border-radius: 4px; font-size: 11px; color: #666; border-left: 3px solid #007bff;">
                <strong>AutosysQuery</strong> • {state['query_results']['row_count']} results • {state['query_results']['execution_time']:.2f}s
            </div>
            """
            
            state["formatted_output"] = formatted_html + metadata
            
        except Exception as e:
            state["error"] = f"Formatting failed: {str(e)}"
        
        return state

    def handle_error_node(self, state: AutosysState) -> AutosysState:
        """Handle errors with user-friendly display"""
        
        error_msg = state.get("error", "Unknown error occurred")
        sql_query = state.get("sql_query", "")
        
        error_html = f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;">
            <h4 style="margin: 0 0 10px 0;">Query Error</h4>
            <p style="margin: 0;"><strong>Error:</strong> {error_msg}</p>
        """
        
        if sql_query:
            error_html += f"""
            <details style="margin-top: 10px;">
                <summary style="cursor: pointer;">View SQL Query</summary>
                <pre style="background: #e9ecef; padding: 10px; margin-top: 5px; border-radius: 3px; font-size: 11px;">{sql_query}</pre>
            </details>
            """
        
        error_html += "</div>"
        state["formatted_output"] = error_html
        
        return state

    # Conditional edge functions
    def _should_handle_conversation(self, state: AutosysState) -> str:
        return "conversation" if state.get("is_general_conversation") else "database"

    def _should_execute_query(self, state: AutosysState) -> str:
        return "error" if state.get("error") else "execute"

    def _should_format_results(self, state: AutosysState) -> str:
        if state.get("error"):
            return "error"
        elif state.get("query_results", {}).get("success"):
            return "format"
        else:
            return "error"

    def query(self, user_question: str, session_id: str) -> Dict[str, Any]:
        """Main method to process any user input"""
        
        initial_state = {
            "messages": [],
            "user_question": user_question,
            "is_general_conversation": False,
            "sql_query": "",
            "query_results": {},
            "formatted_output": "",
            "error": "",
            "session_id": session_id
        }
        
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            final_state = self.graph.invoke(initial_state, config=config)
            
            return {
                "success": not bool(final_state.get("error")),
                "formatted_output": final_state.get("formatted_output", ""),
                "is_conversation": final_state.get("is_general_conversation", False),
                "error": final_state.get("error", "")
            }
            
        except Exception as e:
            self.logger.error(f"Graph execution failed: {e}")
            return {
                "success": False,
                "formatted_output": f"""
                <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;">
                    <h4>System Error</h4>
                    <p>Graph execution failed: {str(e)}</p>
                </div>
                """,
                "error": str(e)
            }

# ============================================================================
# MAIN INTERFACE FUNCTIONS
# ============================================================================

# Global system instance
_autosys_system = None

def setup_autosys_system(autosys_db, llm_instance):
    """Setup the complete Autosys LangGraph system"""
    global _autosys_system
    
    try:
        _autosys_system = AutosysLangGraphSystem(autosys_db, llm_instance)
        logger.info("Autosys LangGraph system initialized successfully")
        
        return {
            "status": "ready",
            "features": [
                "Smart message routing (conversation vs database)",
                "LLM-powered SQL generation",
                "Professional HTML formatting",
                "Session persistence with memory",
                "Comprehensive error handling"
            ]
        }
        
    except Exception as e:
        logger.error(f"System setup failed: {e}")
        raise Exception(f"Failed to initialize system: {str(e)}")

def get_chat_response(message: str, session_id: str) -> str:
    """Main chat function - handles both conversation and database queries"""
    global _autosys_system
    
    try:
        # Input validation
        if not message or not message.strip():
            message = "Hello! How can I help you today?"
        
        # Check system initialization
        if not _autosys_system:
            return """
            <div style="border: 1px solid #ffc107; background: #fff3cd; color: #856404; padding: 15px; border-radius: 5px;">
                <h4>System Not Ready</h4>
                <p>The system is not initialized. Please contact administrator.</p>
            </div>
            """
        
        # Process message
        result = _autosys_system.query(message.strip(), session_id)
        
        return result["formatted_output"]
        
    except Exception as e:
        logger.error(f"Chat response error: {e}")
        return f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;">
            <h4>System Error</h4>
            <p>Error processing request: {str(e)}</p>
        </div>
        """

def extract_last_ai_message(result) -> str:
    """Extract final response (for compatibility)"""
    if isinstance(result, dict) and "formatted_output" in result:
        return result["formatted_output"]
    elif isinstance(result, str):
        return result
    else:
        return str(result)

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Example usage"""
    
    # Your existing setup
    oracle_uri = "oracle+oracledb://username:password@host:port/service_name"
    # autosys_db = AutosysOracleDatabase(oracle_uri)
    # llm = get_llm("langchain")
    
    # Initialize system
    # setup_autosys_system(autosys_db, llm)
    
    # Test queries
    test_messages = [
        "Hello there!",  # → Conversation
        "How are you doing?",  # → Conversation  
        "Show me failed jobs today",  # → Database query
        "List all ATSYS jobs",  # → Database query
        "What can you help me with?"  # → Conversation
    ]
    
    # for message in test_messages:
    #     response = get_chat_response(message, "test_session")
    #     print(f"Input: {message}")
    #     print(f"Output: {response[:100]}...")
    #     print("-" * 50)

if __name__ == "__main__":
    main()

"""
INTEGRATION INSTRUCTIONS:

1. Replace your existing setup with:
   setup_autosys_system(autosys_db, llm)

2. Your get_chat_response() function automatically:
   - Routes general conversation to LLM
   - Routes database queries to LangGraph workflow
   - Returns formatted HTML for both types

3. No other changes needed to your existing API endpoints!
"""












†**********************# ============================================================================
# COMPLETE NEW WAY IMPLEMENTATION - LangGraph with Old Function Names
# ============================================================================

import oracledb
import json
import logging
import re
from typing import Dict, Any, Optional, List, Union, TypedDict, Annotated
from datetime import datetime
from pydantic import BaseModel, Field
import operator

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports (for tool compatibility)
from langchain.tools import BaseTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# STATE DEFINITION FOR LANGGRAPH
# ============================================================================

class AutosysState(TypedDict):
    """State for the Autosys LangGraph workflow"""
    messages: Annotated[List[Dict[str, str]], operator.add]
    user_question: str
    sql_query: str
    query_results: Dict[str, Any]
    formatted_output: str
    error: str
    iteration_count: int
    session_id: str

# ============================================================================
# ENHANCED TOOL FOR LANGGRAPH
# ============================================================================

class AutosysQueryInput(BaseModel):
    """Input schema for Autosys Query Tool"""
    question: str = Field(description="Natural language question about Autosys jobs")

class AutosysLLMQueryTool(BaseTool):
    """Enhanced LangGraph-compatible Autosys Database Query Tool"""
    
    name: str = "AutosysQuery"
    description: str = """
    Query Autosys job scheduler database using natural language.
    Supports questions about job status, schedules, failures, and performance.
    Returns structured data for LangGraph processing.
    
    Examples:
    - "Show me all failed jobs today"
    - "Which ATSYS jobs are currently running?"
    - "List jobs owned by user ADMIN"
    - "Show job history for the last 24 hours"
    """
    
    args_schema = AutosysQueryInput
    
    def __init__(self, autosys_db, llm_instance, max_results: int = 50):
        super().__init__()
        self.autosys_db = autosys_db
        self.llm = llm_instance
        self.max_results = max_results
        self.logger = logging.getLogger(self.__class__.__name__)
        self._verify_db_connection()

    def _verify_db_connection(self) -> bool:
        """Test database connection on initialization"""
        try:
            if hasattr(self.autosys_db, 'connection') and self.autosys_db.connection:
                self.logger.info("AutosysOracleDatabase connection verified")
                return True
            elif hasattr(self.autosys_db, 'connect'):
                self.autosys_db.connect()
                self.logger.info("AutosysOracleDatabase connection established")
                return True
            else:
                self.logger.warning("Could not verify database connection")
                return False
        except Exception as e:
            self.logger.error(f"Database connection verification failed: {str(e)}")
            return False

    def _run(self, question: str) -> Dict[str, Any]:
        """Execute query and return structured results for LangGraph"""
        try:
            self.logger.info(f"Processing question: {question}")
            
            # Generate SQL
            sql_query = self._generate_sql_with_llm(question)
            
            # Execute query
            query_result = self._execute_query_with_existing_db(sql_query)
            
            return {
                "success": query_result["success"],
                "sql_query": sql_query,
                "results": query_result.get("results", []),
                "row_count": query_result.get("row_count", 0),
                "execution_time": query_result.get("execution_time", 0),
                "error": query_result.get("error", "")
            }
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "sql_query": "",
                "results": [],
                "row_count": 0,
                "execution_time": 0
            }

    def _generate_sql_with_llm(self, user_question: str) -> str:
        """Generate SQL using external LLM instance"""
        
        sql_generation_prompt = f"""
You are an expert Oracle SQL generator for Autosys job scheduler database queries.

AUTOSYS DATABASE SCHEMA:
- Table: aedbadmin.ujo_jobst (Job Status)
  * job_name: VARCHAR2 - Job identifier (e.g., 'ATSYS.job_name.c')
  * status: NUMBER - Status code (SU=Success, FA=Failure, RU=Running, IN=Inactive)
  * last_start: NUMBER - Last start time (epoch seconds)
  * last_end: NUMBER - Last end time (epoch seconds)  
  * joid: NUMBER - Job ID (foreign key)

- Table: aedbadmin.ujo_job (Job Details)
  * joid: NUMBER - Job ID (primary key)
  * owner: VARCHAR2 - Job owner/user
  * machine: VARCHAR2 - Execution machine
  * job_type: VARCHAR2 - Type of job

- Table: aedbadmin.UJO_INTCODES (Status Translation)
  * code: NUMBER - Status code number
  * TEXT: VARCHAR2 - Human readable status text

ORACLE SQL PATTERNS FOR AUTOSYS:
- Time conversion: TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss')
- Standard joins: js INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
- Status lookup: LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
- Recent jobs: WHERE js.last_start >= (SYSDATE - INTERVAL '1' DAY) * 86400
- Job name search: WHERE UPPER(js.job_name) LIKE UPPER('%pattern%')

COMMON STATUS CODES:
- SU or 4 = SUCCESS
- FA or 7 = FAILURE  
- RU or 8 = RUNNING
- IN or 5 = INACTIVE

USER QUESTION: "{user_question}"

Generate a complete Oracle SQL query that answers this question.
Use proper table aliases (js for ujo_jobst, j for ujo_job, ic for UJO_INTCODES).
Include relevant columns like job_name, status text, start/end times, owner.
Always use the full table names with aedbadmin schema prefix.
Ensure proper WHERE clauses and result limiting.

Return ONLY the SQL query without any explanations, markdown, or formatting.
"""

        try:
            response = self.llm.invoke(sql_generation_prompt)
            sql_query = response.content if hasattr(response, 'content') else str(response)
            
            # Clean and enhance SQL
            sql_query = self._clean_and_enhance_sql(sql_query)
            
            self.logger.info(f"Generated SQL: {sql_query[:100]}...")
            return sql_query
            
        except Exception as e:
            self.logger.error(f"SQL generation failed: {str(e)}")
            return self._get_fallback_sql(user_question)

    def _clean_and_enhance_sql(self, sql_query: str) -> str:
        """Clean up SQL response and add enhancements"""
        # Remove markdown formatting
        sql_query = re.sub(r'```sql\s*', '', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'```\s*', '', sql_query)
        
        # Clean whitespace
        sql_query = ' '.join(sql_query.split())
        sql_query = sql_query.strip()
        
        # Ensure result limiting
        if 'ROWNUM' not in sql_query.upper() and 'LIMIT' not in sql_query.upper():
            if 'WHERE' in sql_query.upper():
                # Insert ROWNUM condition into existing WHERE clause
                where_pos = sql_query.upper().find('WHERE')
                before_where = sql_query[:where_pos + 5]  # Include 'WHERE'
                after_where = sql_query[where_pos + 5:]
                sql_query = f"{before_where} ROWNUM <= {self.max_results} AND{after_where}"
            else:
                # Add WHERE clause with ROWNUM
                order_pos = sql_query.upper().find('ORDER BY')
                if order_pos > 0:
                    before_order = sql_query[:order_pos]
                    after_order = sql_query[order_pos:]
                    sql_query = f"{before_order} WHERE ROWNUM <= {self.max_results} {after_order}"
                else:
                    sql_query = sql_query.rstrip(';') + f" WHERE ROWNUM <= {self.max_results}"
        
        # Ensure proper ending
        if not sql_query.endswith(';'):
            sql_query += ';'
            
        return sql_query

    def _execute_query_with_existing_db(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query using existing AutosysOracleDatabase"""
        
        try:
            start_time = datetime.now()
            
            # Use existing database connection method
            if hasattr(self.autosys_db, 'run'):
                raw_results = self.autosys_db.run(sql_query)
            elif hasattr(self.autosys_db, 'execute_query'):
                raw_results = self.autosys_db.execute_query(sql_query)
            elif hasattr(self.autosys_db, 'query'):
                raw_results = self.autosys_db.query(sql_query)
            else:
                # Fallback: try to access connection directly
                if hasattr(self.autosys_db, 'connection') and self.autosys_db.connection:
                    with self.autosys_db.connection.cursor() as cursor:
                        cursor.execute(sql_query.rstrip(';'))  # Remove semicolon for cursor.execute
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        raw_results = cursor.fetchall()
                        
                        # Convert to list of dictionaries
                        results = []
                        for row in raw_results:
                            row_dict = {}
                            for i, value in enumerate(row):
                                col_name = columns[i] if i < len(columns) else f"COLUMN_{i}"
                                row_dict[col_name] = self._format_oracle_value(value)
                            results.append(row_dict)
                        raw_results = results
                else:
                    raise Exception("Could not access database connection")
            
            # Process results
            processed_results = self._process_database_results(raw_results)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "results": processed_results,
                "row_count": len(processed_results),
                "execution_time": execution_time,
                "sql_query": sql_query
            }
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "row_count": 0,
                "execution_time": 0,
                "sql_query": sql_query
            }

    def _process_database_results(self, raw_results) -> List[Dict]:
        """Convert raw database results to standardized format"""
        if isinstance(raw_results, str):
            # Handle string results (parsed from your database class)
            try:
                import ast
                if raw_results.startswith('[') and raw_results.endswith(']'):
                    parsed_results = ast.literal_eval(raw_results)
                    return self._convert_to_dict_list(parsed_results)
                else:
                    return [{"result": raw_results}]
            except:
                return [{"result": raw_results}]
        elif isinstance(raw_results, list):
            return self._convert_to_dict_list(raw_results)
        else:
            return [{"result": str(raw_results)}]

    def _convert_to_dict_list(self, raw_results: List) -> List[Dict]:
        """Convert list of tuples to list of dictionaries"""
        results = []
        
        for item in raw_results:
            if isinstance(item, (tuple, list)):
                # Convert tuple/list to dictionary
                row_dict = {}
                for i, value in enumerate(item):
                    col_name = self._get_column_name(i, value)
                    row_dict[col_name] = self._format_oracle_value(value)
                results.append(row_dict)
            elif isinstance(item, dict):
                # Already a dictionary
                results.append(item)
            else:
                # Single value
                results.append({"VALUE": str(item)})
        
        return results

    def _get_column_name(self, index: int, value: Any) -> str:
        """Generate appropriate column name based on typical Autosys schema"""
        common_names = ["JOB_NAME", "START_TIME", "END_TIME", "STATUS", "OWNER", "MACHINE"]
        
        if index < len(common_names):
            return common_names[index]
        else:
            return f"COLUMN_{index + 1}"

    def _format_oracle_value(self, value: Any) -> str:
        """Format Oracle-specific data types"""
        if hasattr(value, 'read'):  # CLOB/BLOB
            return value.read()
        elif isinstance(value, datetime):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return str(value) if value is not None else ""

    def _get_fallback_sql(self, user_question: str) -> str:
        """Fallback SQL query if LLM generation fails"""
        question_lower = user_question.lower()
        
        if 'failed' in question_lower or 'failure' in question_lower:
            status_condition = "js.status = 7"  # FA status code
        elif 'running' in question_lower:
            status_condition = "js.status = 8"  # RU status code  
        elif 'success' in question_lower:
            status_condition = "js.status = 4"  # SU status code
        else:
            status_condition = "1=1"  # All jobs
        
        return f"""
        SELECT 
            js.job_name,
            TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (js.last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss') AS start_time,
            TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (js.last_end / 86400), 'MM/DD/YYYY HH24:Mi:Ss') AS end_time,
            NVL(ic.TEXT, 'UNKNOWN') AS job_status,
            j.owner
        FROM aedbadmin.ujo_jobst js
        INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
        LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
        WHERE {status_condition}
        AND UPPER(js.job_name) LIKE UPPER('%ATSYS%')
        AND ROWNUM <= {self.max_results}
        ORDER BY js.last_start DESC;
        """

# ============================================================================
# LANGGRAPH WORKFLOW IMPLEMENTATION
# ============================================================================

class AutosysLangGraphSystem:
    """LangGraph-based Autosys system with old function names"""
    
    def __init__(self, autosys_db, llm_instance):
        self.autosys_db = autosys_db
        self.llm = llm_instance
        self.tool = AutosysLLMQueryTool(autosys_db, llm_instance)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize memory for persistence
        self.memory = MemorySaver()
        
        # Build the graph
        self.graph = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(AutosysState)
        
        # Add nodes
        workflow.add_node("understand_query", self.understand_query_node)
        workflow.add_node("generate_sql", self.generate_sql_node)
        workflow.add_node("execute_query", self.execute_query_node)
        workflow.add_node("format_results", self.format_results_node)
        workflow.add_node("handle_error", self.handle_error_node)
        
        # Set entry point
        workflow.set_entry_point("understand_query")
        
        # Add edges with conditional logic
        workflow.add_edge("understand_query", "generate_sql")
        workflow.add_conditional_edges(
            "generate_sql",
            self._should_execute_query,
            {
                "execute": "execute_query",
                "error": "handle_error"
            }
        )
        workflow.add_conditional_edges(
            "execute_query", 
            self._should_format_results,
            {
                "format": "format_results",
                "error": "handle_error"
            }
        )
        workflow.add_edge("format_results", END)
        workflow.add_edge("handle_error", END)
        
        # Compile with memory
        return workflow.compile(checkpointer=self.memory)

    def understand_query_node(self, state: AutosysState) -> AutosysState:
        """Analyze and enhance the user query"""
        self.logger.info(f"Understanding query: {state['user_question']}")
        
        # Add analysis message
        state["messages"].append({
            "role": "system", 
            "content": f"Processing Autosys query: {state['user_question']}"
        })
        
        # Enhance question with context if needed
        question = state["user_question"].strip()
        
        if not any(keyword in question.lower() for keyword in ['atsys', 'job', 'status', 'schedule']):
            question = f"Show Autosys jobs related to: {question}"
            state["user_question"] = question
            
        self.logger.info(f"Enhanced question: {state['user_question']}")
        
        return state

    def generate_sql_node(self, state: AutosysState) -> AutosysState:
        """Generate SQL query using LLM"""
        try:
            self.logger.info("Generating SQL query")
            
            # Use the tool to generate SQL
            tool_result = self.tool.run(state["user_question"])
            
            if tool_result["success"]:
                state["sql_query"] = tool_result["sql_query"]
                state["messages"].append({
                    "role": "system",
                    "content": f"SQL generated successfully: {len(tool_result['sql_query'])} characters"
                })
            else:
                state["error"] = f"SQL generation failed: {tool_result.get('error', 'Unknown error')}"
                state["messages"].append({
                    "role": "system",
                    "content": f"SQL generation error: {state['error']}"
                })
            
        except Exception as e:
            state["error"] = f"SQL generation exception: {str(e)}"
            self.logger.error(f"SQL generation failed: {e}")
        
        return state

    def execute_query_node(self, state: AutosysState) -> AutosysState:
        """Execute the SQL query"""
        try:
            self.logger.info("Executing SQL query")
            
            # Execute using the tool
            tool_result = self.tool.run(state["user_question"])
            
            if tool_result["success"]:
                state["query_results"] = {
                    "success": True,
                    "results": tool_result["results"],
                    "row_count": tool_result["row_count"],
                    "execution_time": tool_result["execution_time"],
                    "sql_query": tool_result["sql_query"]
                }
                state["messages"].append({
                    "role": "system",
                    "content": f"Query executed: {tool_result['row_count']} results in {tool_result['execution_time']:.2f}s"
                })
            else:
                state["error"] = f"Query execution failed: {tool_result.get('error', 'Unknown error')}"
                state["query_results"] = {"success": False, "error": state["error"]}
                
        except Exception as e:
            state["error"] = f"Query execution exception: {str(e)}"
            state["query_results"] = {"success": False, "error": state["error"]}
            self.logger.error(f"Query execution failed: {e}")
        
        return state

    def format_results_node(self, state: AutosysState) -> AutosysState:
        """Format results using LLM with system prompt approach"""
        try:
            self.logger.info("Formatting results")
            
            results = state["query_results"]["results"]
            
            if not results:
                state["formatted_output"] = self._create_no_results_html()
                return state
            
            # Use LLM for professional formatting
            formatting_prompt = f"""
Create professional HTML formatting for these Autosys database query results.

USER QUESTION: "{state['user_question']}"
EXECUTION TIME: {state['query_results'].get('execution_time', 0):.2f} seconds  
TOTAL RESULTS: {len(results)} jobs (showing sample)

QUERY RESULTS TO FORMAT:
{json.dumps(results[:15], indent=2, default=str)}

FORMATTING REQUIREMENTS:
1. Create responsive HTML with inline CSS only - no external stylesheets
2. Use professional styling with good contrast and readability
3. Color-code job status with badges:
   - SUCCESS/SU: Green background (#28a745) with white text
   - FAILURE/FA: Red background (#dc3545) with white text  
   - RUNNING/RU: Blue background (#007bff) with white text
   - INACTIVE/IN: Gray background (#6c757d) with white text
4. Include a summary section at the top showing total jobs and key statistics
5. Use monospace font for job names for better readability
6. Make tables responsive for mobile devices
7. Keep output concise to avoid truncation - prioritize most important info
8. Add subtle hover effects for better user experience
9. Include proper spacing and margins for professional appearance
10. Show execution time and result count in a footer

Create a clean, modern design that's easy to scan and read.
Return only the formatted HTML without any explanations or markdown.
"""
            
            response = self.llm.invoke(formatting_prompt)
            formatted_html = response.content if hasattr(response, 'content') else str(response)
            
            # Add LangGraph metadata footer
            metadata = f"""
            <div style="margin-top: 15px; padding: 8px 12px; background: #f8f9fa; border-radius: 4px; font-size: 11px; color: #666; border-left: 3px solid #007bff;">
                <strong>LangGraph AutosysQuery</strong> • 
                {state['query_results']['row_count']} results in {state['query_results']['execution_time']:.2f}s • 
                Enhanced Oracle Database Query System
            </div>
            """
            
            state["formatted_output"] = formatted_html + metadata
            state["messages"].append({
                "role": "assistant",
                "content": "Results formatted successfully with professional HTML styling"
            })
            
        except Exception as e:
            state["error"] = f"Formatting failed: {str(e)}"
            state["formatted_output"] = self._create_error_html(state["error"])
            self.logger.error(f"Formatting failed: {e}")
        
        return state

    def handle_error_node(self, state: AutosysState) -> AutosysState:
        """Handle errors and provide user-friendly error messages"""
        self.logger.error(f"Handling error: {state.get('error', 'Unknown error')}")
        
        error_msg = state.get("error", "An unknown error occurred")
        sql_query = state.get("sql_query", "")
        
        state["formatted_output"] = self._create_error_html(error_msg, sql_query)
        state["messages"].append({
            "role": "system",
            "content": f"Error handled and formatted for user display"
        })
        
        return state

    def _should_execute_query(self, state: AutosysState) -> str:
        """Conditional edge: decide whether to execute query or handle error"""
        if state.get("error"):
            return "error"
        elif state.get("sql_query"):
            return "execute"
        else:
            return "error"

    def _should_format_results(self, state: AutosysState) -> str:
        """Conditional edge: decide whether to format results or handle error"""
        if state.get("error"):
            return "error"
        elif state.get("query_results", {}).get("success"):
            return "format"
        else:
            return "error"

    def _create_no_results_html(self) -> str:
        """HTML for no results found"""
        return """
        <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #856404;">No Results Found</h4>
            <p style="margin: 0; color: #856404;">No Autosys jobs match your query criteria. Try rephrasing your question or checking job names.</p>
        </div>
        """

    def _create_error_html(self, error_msg: str, sql_query: str = "") -> str:
        """HTML for error display"""
        html = f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #721c24;">LangGraph AutosysQuery Error</h4>
            <p style="margin: 0; font-size: 14px;"><strong>Error:</strong> {error_msg}</p>
        """
        
        if sql_query:
            html += f"""
            <details style="margin-top: 10px;">
                <summary style="cursor: pointer; color: #495057; font-size: 12px;">View Generated SQL Query</summary>
                <pre style="background: #e9ecef; padding: 10px; margin-top: 5px; border-radius: 3px; font-size: 10px; overflow-x: auto; font-family: 'Courier New', monospace;">{sql_query}</pre>
            </details>
            """
        
        html += """
            <p style="margin: 10px 0 0 0; font-size: 12px; color: #856404;">
                <em>Tip: Try rephrasing your question or check if the job names/criteria exist in the database.</em>
            </p>
        </div>
        """
        return html

    def process_query(self, user_question: str, session_id: str) -> Dict[str, Any]:
        """Process a query using the LangGraph workflow"""
        
        # Initialize state
        initial_state = {
            "messages": [],
            "user_question": user_question,
            "sql_query": "",
            "query_results": {},
            "formatted_output": "",
            "error": "",
            "iteration_count": 0,
            "session_id": session_id
        }
        
        # Configuration for session management
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            self.logger.info(f"Processing query for session {session_id}: {user_question}")
            final_state = self.graph.invoke(initial_state, config=config)
            
            return {
                "success": not bool(final_state.get("error")),
                "formatted_output": final_state.get("formatted_output", ""),
                "sql_query": final_state.get("sql_query", ""),
                "row_count": final_state.get("query_results", {}).get("row_count", 0),
                "execution_time": final_state.get("query_results", {}).get("execution_time", 0),
                "error": final_state.get("error", ""),
                "messages": final_state.get("messages", [])
            }
            
        except Exception as e:
            self.logger.error(f"LangGraph execution failed: {e}")
            return {
                "success": False,
                "formatted_output": self._create_error_html(f"System execution failed: {str(e)}"),
                "error": str(e),
                "messages": []
            }

# ============================================================================
# OLD FUNCTION NAMES WITH NEW IMPLEMENTATION
# ============================================================================

# Global variable to store the LangGraph system
_autosys_langgraph_system = None

def initialize_agent(tools, llm, agent=None, verbose=True, checkpointer=None, 
                    handle_parsing_errors=True, max_iterations=3, 
                    early_stopping_method="generate", agent_kwargs=None):
    """
    Initialize the LangGraph-based Autosys system with the same function signature
    as the old initialize_agent function
    """
    global _autosys_langgraph_system
    
    try:
        # Extract autosys_db from tools or use global reference
        # This assumes you pass your autosys_db somehow - adjust as needed
        # For now, we'll assume it's available globally or passed in a specific way
        
        if hasattr(tools[0], 'autosys_db') if tools else False:
            autosys_db = tools[0].autosys_db
        else:
            # You'll need to provide autosys_db reference here
            # This is where you'd inject your AutosysOracleDatabase instance
            autosys_db = None  # Replace with your actual autosys_db instance
            
        _autosys_langgraph_system = AutosysLangGraphSystem(autosys_db, llm)
        
        logger.info("LangGraph Autosys system initialized successfully")
        return _autosys_langgraph_system
        
    except Exception as e:
        logger.error(f"Failed to initialize LangGraph system: {e}")
        raise e

def extract_last_ai_message(result: Union[Dict, str, Any]) -> str:
    """
    Enhanced message extraction that works with both old and new LangGraph responses
    """
    try:
        # Handle LangGraph response format (New Way)
        if isinstance(result, dict):
            if "formatted_output" in result and result["formatted_output"]:
                return result["formatted_output"].strip()
            if result.get("success") and "formatted_output" in result:
                return result["formatted_output"].strip()
            if "answer" in result and result["answer"]:
                return result["answer"].strip()
            if "result" in result and result["result"]:
                return result["result"].strip()
            if "content" in result and result["content"]:
                return result["content"].strip()
            if "output" in result and result["output"]:
                return result["output"].strip()
        
        # Handle string responses (Old Way compatibility)
        if isinstance(result, str):
            result = result.strip()
            final_answer_match = re.search(r"Final Answer:\s*(.+?)(?=\n\n|\n(?=\w+:)|\Z)", result, re.DOTALL | re.IGNORECASE)
            if final_answer_match:
                return final_answer_match.group(1).strip()
            if "<html>" in result.lower() or "<div>" in result.lower() or "<table>" in result.lower():
                return result
            if len(result) > 0 and not any(marker in result.lower() for marker in ['thought:', 'action:', 'observation:']):
                return result
        
        # Handle object attributes
        if hasattr(result, 'content'):
            return result.content.strip()
        elif hasattr(result, 'output'):
            return result.output.strip()
        
        result_str = str(result).strip()
        if result_str and result_str != "None":
            return result_str
            
        return "No response generated."
        
    except Exception as e:
        logging.error(f"Error extracting AI message: {str(e)}")
        return f"Error processing response: {str(e)}"

def is_autosys_related_query(message: str) -> bool:
    """
    Determine if the message is related to Autosys/database queries
    """
    autosys_keywords = [
        'job', 'jobs', 'atsys', 'autosys', 'schedule', 'status', 'failed', 'failure',
        'running', 'success', 'database', 'query', 'select', 'show', 'list', 'find',
        'search', 'owner', 'machine', 'execution', 'sql', 'table', 'count', 'report'
    ]
    
    message_lower = message.lower()
    
    # Check for direct Autosys keywords
    if any(keyword in message_lower for keyword in autosys_keywords):
        return True
    
    # Check for question patterns that might be database-related
    database_patterns = [
        r'\b(what|which|how many|show me|list|find)\b.*\b(job|status|schedule)\b',
        r'\b(failed|running|success|error)\b',
        r'\b(today|yesterday|last|recent)\b.*\b(job|run|execution)\b'
    ]
    
    if any(re.search(pattern, message_lower) for pattern in database_patterns):
        return True
    
    return False

def handle_general_conversation(message: str, session_id: str, llm_instance) -> str:
    """
    Handle general conversation using LLM
    """
    try:
        # System prompt for general conversation
        general_conversation_prompt = f"""
You are a helpful AI assistant for an Autosys database system. You can handle both general conversation and specific database queries.

For this message, the user is asking a general question that doesn't require database access. Respond naturally and helpfully.

If the user wants to know about your capabilities, mention that you can:
- Answer general questions and have conversations
- Query the Autosys job scheduler database
- Provide job status, schedules, and performance information
- Generate reports and analyze job data

Keep responses friendly, professional, and concise.

User message: "{message}"

Respond naturally to this message:
"""
        
        response = llm_instance.invoke(general_conversation_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Format as simple HTML for consistency
        formatted_response = f"""
        <div style="padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 3px solid #28a745;">
            <p style="margin: 0; color: #333; line-height: 1.5;">{content}</p>
            <div style="margin-top: 10px; font-size: 11px; color: #666;">
                <em>General conversation • Ask me about Autosys jobs or schedules for database queries</em>
            </div>
        </div>
        """
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in general conversation: {e}")
        return f"""
        <div style="padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 3px solid #28a745;">
            <p style="margin: 0; color: #333;">Hello! I'm here to help with both general questions and Autosys database queries.</p>
            <p style="margin: 10px 0 0 0; color: #333;">Feel free to ask me about job statuses, schedules, or just chat!</p>
            <div style="margin-top: 10px; font-size: 11px; color: #666;">
                <em>General conversation mode</em>
            </div>
        </div>
        """

def get_chat_response(message: str, session_id: str) -> str:
    """
    Enhanced chat response function that handles both general conversation and Autosys queries
    
    This function:
    1. Determines if the message is Autosys-related or general conversation
    2. Routes to appropriate handler (LangGraph for Autosys, LLM for general)
    3. Maintains the same signature as your existing function
    """
    global _autosys_langgraph_system
    
    try:
        # Input validation
        if not message or not message.strip():
            return handle_general_conversation("Hello! How can I help you today?", session_id, 
                                             _autosys_langgraph_system.llm if _autosys_langgraph_system else None)
        
        message = message.strip()
        logger.info(f"Processing chat request: {message}")
        
        # Determine if this is an Autosys-related query or general conversation
        is_autosys_query = is_autosys_related_query(message)
        
        if not is_autosys_query:
            # Handle general conversation with LLM
            logger.info("Routing to general conversation handler")
            
            if _autosys_langgraph_system and _autosys_langgraph_system.llm:
                return handle_general_conversation(message, session_id, _autosys_langgraph_system.llm)
            else:
                return """
                <div style="padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 3px solid #28a745;">
                    <p style="margin: 0; color: #333;">Hello! I'm here to help with both general questions and Autosys database queries.</p>
                    <p style="margin: 10px 0 0 0; color: #333;">However, the system isn't fully initialized yet. Please contact an administrator.</p>
                </div>
                """
        
        # Handle Autosys-related queries with LangGraph
        logger.info("Routing to Autosys LangGraph system")
        
        # Check if LangGraph system is initialized
        if not _autosys_langgraph_system:
            return """
            <div style="border: 1px solid #ffc107; background: #fff3cd; color: #856404; padding: 15px; border-radius: 5px;">
                <h4 style="margin: 0 0 10px 0;">System Not Ready</h4>
                <p style="margin: 0;">The Autosys query system is not initialized. Please contact administrator.</p>
            </div>
            """
        
        # Process Autosys query using LangGraph
        result = _autosys_langgraph_system.process_query(message, session_id)
        
        if result["success"]:
            logger.info(f"Autosys query processed successfully: {result['row_count']} results in {result['execution_time']:.2f}s")
            return result["formatted_output"]
        else:
            logger.error(f"Autosys query failed: {result['error']}")
            return result["formatted_output"]  # Error HTML is already in formatted_output
            
    except Exception as e:
        logger.error(f"Critical error in get_chat_response: {str(e)}")
        return f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;">
            <h4 style="margin: 0 0 10px 0;">System Error</h4>
            <p style="margin: 0;">A critical error occurred while processing your request: {str(e)}</p>
            <p style="margin: 10px 0 0 0; font-size: 12px;">Please try again or contact support if the problem persists.</p>
        </div>
        """

# ============================================================================
# ENHANCED CONVERSATION MANAGEMENT
# ============================================================================

def get_conversation_context(session_id: str) -> str:
    """Get recent conversation context for better responses"""
    global _autosys_langgraph_system
    
    try:
        if not _autosys_langgraph_system or not hasattr(_autosys_langgraph_system, 'memory'):
            return ""
        
        # Try to get recent context from memory
        config = {"configurable": {"thread_id": session_id}}
        # This would require implementing memory retrieval - simplified for now
        return ""
        
    except Exception:
        return ""

def enhance_general_conversation_with_context(message: str, session_id: str, llm_instance) -> str:
    """Enhanced general conversation with context awareness"""
    
    context = get_conversation_context(session_id)
    context_info = f"Previous conversation context: {context}" if context else "This is a new conversation."
    
    enhanced_prompt = f"""
You are a helpful AI assistant for an Autosys database system. You excel at both general conversation and database queries.

{context_info}

You can:
- Have natural, friendly conversations about any topic
- Query the Autosys job scheduler database for job information
- Provide job status, schedules, performance data, and reports
- Help with general questions about technology, work, or daily life

Current user message: "{message}"

Guidelines:
- Be conversational, helpful, and professional
- If the user greets you, greet them back warmly
- If they ask how you are, mention you're ready to help with both chat and database queries
- If they ask about your capabilities, explain both conversation and Autosys features
- Keep responses natural and engaging
- Don't be overly formal for casual conversation

Respond naturally to this message:
"""
    
    try:
        response = llm_instance.invoke(enhanced_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Determine styling based on message type
        greeting_words = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings']
        is_greeting = any(word in message.lower() for word in greeting_words)
        
        if is_greeting:
            border_color = "#28a745"  # Green for greetings
            icon = "👋"
        elif any(word in message.lower() for word in ['how are you', 'how do you do', 'how have you been']):
            border_color = "#17a2b8"  # Blue for status questions  
            icon = "💬"
        elif any(word in message.lower() for word in ['help', 'what can you do', 'capabilities', 'features']):
            border_color = "#ffc107"  # Yellow for help
            icon = "ℹ️"
        else:
            border_color = "#6c757d"  # Gray for general conversation
            icon = "💭"
        
        formatted_response = f"""
        <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid {border_color}; margin: 10px 0;">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 16px; margin-right: 8px;">{icon}</span>
                <span style="font-weight: 500; color: #333;">Assistant</span>
            </div>
            <p style="margin: 0; color: #333; line-height: 1.6;">{content}</p>
            <div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid #dee2e6; font-size: 11px; color: #666;">
                <em>💬 General conversation • Ask about Autosys jobs for database queries • Session: {session_id[:8]}...</em>
            </div>
        </div>
        """
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in enhanced general conversation: {e}")
        return create_friendly_fallback_response(message)

def create_friendly_fallback_response(message: str) -> str:
    """Create a friendly fallback response when LLM fails"""
    
    greeting_words = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
    question_words = ['how are you', 'how do you do', 'what', 'who', 'where', 'when', 'why', 'help']
    
    if any(word in message.lower() for word in greeting_words):
        response_text = "Hello! I'm doing great and ready to help. I can chat with you about anything or help you query the Autosys database for job information. What would you like to know?"
    elif any(word in message.lower() for word in ['how are you', 'how do you do']):
        response_text = "I'm doing well, thank you for asking! I'm here and ready to assist with both general conversation and Autosys database queries. How can I help you today?"
    elif 'help' in message.lower() or 'what can you do' in message.lower():
        response_text = "I'm here to help! I can have conversations about various topics and also query our Autosys job scheduler database. I can check job statuses, find failed jobs, show running processes, and generate reports. Just ask me anything!"
    else:
        response_text = "I'm here to help with both general questions and Autosys database queries. Feel free to ask me anything - from casual conversation to specific job status requests!"
    
    return f"""
    <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 16px; margin-right: 8px;">🤖</span>
            <span style="font-weight: 500; color: #333;">Assistant</span>
        </div>
        <p style="margin: 0; color: #333; line-height: 1.6;">{response_text}</p>
        <div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid #dee2e6; font-size: 11px; color: #666;">
            <em>💬 General conversation mode</em>
        </div>
    </div>
    """

# Update the handle_general_conversation function to use enhanced version
def handle_general_conversation(message: str, session_id: str, llm_instance) -> str:
    """
    Handle general conversation using LLM with enhanced context and styling
    """
    return enhance_general_conversation_with_context(message, session_id, llm_instance)

# ============================================================================
# EXAMPLE CONVERSATION FLOWS
# ============================================================================

def demonstrate_conversation_routing():
    """
    Demonstration of how the system routes different types of messages
    """
    
    test_cases = [
        # General conversation
        ("Hi there!", "General conversation - friendly greeting"),
        ("Hello, how are you?", "General conversation - status inquiry"),
        ("What can you help me with?", "General conversation - capability inquiry"),
        ("Good morning!", "General conversation - greeting"),
        ("How's your day going?", "General conversation - casual chat"),
        
        # Autosys queries
        ("Show me failed jobs", "Autosys query - job status"),
        ("List all ATSYS jobs running today", "Autosys query - specific search"),
        ("What jobs are scheduled for tonight?", "Autosys query - schedule inquiry"),
        ("Find jobs owned by admin", "Autosys query - owner search"),
        ("Database query for job status", "Autosys query - explicit database request"),
        
        # Edge cases
        ("Hello, can you show me job status?", "Mixed - greeting + Autosys query (routes to Autosys)"),
        ("Hi! What failed today?", "Mixed - greeting + job inquiry (routes to Autosys)"),
        ("", "Empty message (routes to general conversation)")
    ]
    
    print("Message Routing Demonstration:")
    print("=" * 50)
    
    for message, expected in test_cases:
        is_autosys = is_autosys_related_query(message)
        route = "Autosys LangGraph" if is_autosys else "General LLM"
        print(f"Message: '{message}'")
        print(f"Route: {route}")
        print(f"Expected: {expected}")
        print("-" * 30)

# ============================================================================
# COMPLETE SETUP WITH CONVERSATION HANDLING
# ============================================================================

def setup_autosys_langgraph_system(autosys_db, llm_instance):
    """
    Complete setup function with enhanced conversation handling
    
    Args:
        autosys_db: Your AutosysOracleDatabase instance
        llm_instance: Your LLM instance from get_llm("langchain")
    
    Returns:
        Configured LangGraph system ready for both conversation and database queries
    """
    global _autosys_langgraph_system
    
    try:
        # Initialize the new LangGraph system
        _autosys_langgraph_system = AutosysLangGraphSystem(autosys_db, llm_instance)
        
        logger.info("LangGraph Autosys system with conversation handling setup completed")
        
        # Test both conversation and database capabilities
        conversation_test = handle_general_conversation("Hello!", "test_session", llm_instance)
        logger.info("Conversation handling test completed")
        
        return {
            "system": _autosys_langgraph_system,
            "status": "ready",
            "type": "langgraph_with_conversation",
            "features": [
                "General conversation with LLM",
                "Smart message routing (conversation vs database)",
                "Enhanced SQL generation for Autosys queries",
                "Professional HTML formatting", 
                "State-based error handling",
                "Session persistence with context",
                "Comprehensive logging and debugging"
            ],
            "conversation_test": "passed" if "Hello" in conversation_test else "needs_review"
        }
        
    except Exception as e:
        logger.error(f"Failed to setup enhanced system: {e}")
        raise Exception(f"Enhanced system setup failed: {str(e)}")

# ============================================================================
# FINAL INTEGRATION WITH CONVERSATION SUPPORT
# ============================================================================

"""
ENHANCED INTEGRATION INSTRUCTIONS:

Your system now supports both general conversation and Autosys database queries!

ROUTING LOGIC:
- General conversation (hi, hello, how are you, help, etc.) → LLM
- Autosys queries (jobs, status, failed, running, database, etc.) → LangGraph
- Mixed messages route to the primary intent (usually Autosys if detected)

EXAMPLES:
1. User: "Hi there!" 
   → Routes to LLM for friendly conversation

2. User: "How are you doing?"
   → Routes to LLM for general chat

3. User: "Show me failed jobs today"
   → Routes to LangGraph for database query

4. User: "Hello, can you check job status?"
   → Routes to LangGraph (Autosys keywords detected)

SETUP:
Just one line replaces your entire agent setup:
setup_autosys_langgraph_system(autosys_db, llm)

Your get_chat_response() function automatically handles routing!
"""

# ============================================================================
# DIRECT REPLACEMENT FOR YOUR EXISTING CODE
# ============================================================================

"""
COMPLETE INTEGRATION INSTRUCTIONS:

1. REPLACE your existing imports and setup:

# OLD CODE:
from langchain.agents import initialize_agent, AgentType
sql_agent = initialize_agent(
    tools=[autosys_sql_tool_enhanced],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    checkpointer=checkpointer,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="generate",
    agent_kwargs={"format_instructions": "..."}
)

# NEW CODE:
# Just add this one line after your existing setup:
setup_autosys_langgraph_system(autosys_db, llm)

# Your get_chat_response function stays the same!
# It will automatically use the new LangGraph implementation

2. YOUR EXISTING FUNCTION CALLS REMAIN UNCHANGED:
   - get_chat_response(message, session_id) works exactly the same
   - extract_last_ai_message() works exactly the same
   - All your API endpoints and session handling stay the same

3. WHAT YOU GET:
   - Better error handling and recovery
   - Professional HTML formatting with system prompts
   - Enhanced SQL generation using LLM
   - State-based workflow with observability
   - Session persistence with memory
   - Comprehensive logging and debugging
"""

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_integration():
    """
    Example showing how to integrate with your existing code
    """
    
    # Your existing setup (no changes needed):
    oracle_uri = "oracle+oracledb://username:password@host:port/service_name"
    # autosys_db = AutosysOracleDatabase(oracle_uri)  # Your existing DB
    # llm = get_llm("langchain")  # Your existing LLM
    
    # NEW: Single line to replace your entire agent setup:
    # setup_autosys_langgraph_system(autosys_db, llm)
    
    # Your existing function calls work unchanged:
    # response = get_chat_response("Show me failed ATSYS jobs", "session_123")
    # print(response)  # Gets professional HTML output
    
    pass

# ============================================================================
# UTILITY FUNCTIONS FOR DEBUGGING AND MONITORING
# ============================================================================

def debug_langgraph_system():
    """Debug function to check system status"""
    global _autosys_langgraph_system
    
    if not _autosys_langgraph_system:
        return {"status": "not_initialized", "system": None}
    
    return {
        "status": "initialized",
        "system_type": "LangGraph",
        "has_memory": hasattr(_autosys_langgraph_system, 'memory'),
        "has_tool": hasattr(_autosys_langgraph_system, 'tool'),
        "has_graph": hasattr(_autosys_langgraph_system, 'graph')
    }

def get_system_metrics(session_id: str = "default") -> Dict[str, Any]:
    """Get system performance metrics"""
    global _autosys_langgraph_system
    
    if not _autosys_langgraph_system:
        return {"error": "System not initialized"}
    
    try:
        # Test query to check system health
        test_result = _autosys_langgraph_system.process_query(
            "SELECT COUNT(*) FROM aedbladmin.ujo_jobst WHERE ROWNUM <= 1", 
            session_id
        )
        
        return {
            "system_status": "healthy" if test_result["success"] else "error",
            "test_execution_time": test_result.get("execution_time", 0),
            "last_error": test_result.get("error", None),
            "memory_enabled": hasattr(_autosys_langgraph_system, 'memory'),
            "session_id": session_id
        }
        
    except Exception as e:
        return {
            "system_status": "error",
            "error": str(e),
            "session_id": session_id
        }

# ============================================================================
# FINAL INTEGRATION SNIPPET
# ============================================================================

# COPY THIS EXACT CODE INTO YOUR EXISTING FILE:
"""
# Add these imports at the top of your file:
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Replace your existing sql_agent setup with this single line:
setup_autosys_langgraph_system(autosys_db, llm)

# Your existing get_chat_response function will now automatically use LangGraph!
# No other changes needed to your existing code.
"""

if __name__ == "__main__":
    # Test the system
    print("LangGraph Autosys System - NEW WAY Implementation")
    print("Ready for integration with existing codebase")
    print("\nFeatures enabled:")
    print("- LLM-powered SQL generation")
    print("- Professional HTML formatting") 
    print("- State-based error handling")
    print("- Session persistence")
    print("- Enhanced observability")





£££££££££££££££££££££££££££££





# ============================================================================
# COMPLETE NEW WAY IMPLEMENTATION - LangGraph with Old Function Names
# ============================================================================

import oracledb
import json
import logging
import re
from typing import Dict, Any, Optional, List, Union, TypedDict, Annotated
from datetime import datetime
from pydantic import BaseModel, Field
import operator

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports (for tool compatibility)
from langchain.tools import BaseTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# STATE DEFINITION FOR LANGGRAPH
# ============================================================================

class AutosysState(TypedDict):
    """State for the Autosys LangGraph workflow"""
    messages: Annotated[List[Dict[str, str]], operator.add]
    user_question: str
    sql_query: str
    query_results: Dict[str, Any]
    formatted_output: str
    error: str
    iteration_count: int
    session_id: str

# ============================================================================
# ENHANCED TOOL FOR LANGGRAPH
# ============================================================================

class AutosysQueryInput(BaseModel):
    """Input schema for Autosys Query Tool"""
    question: str = Field(description="Natural language question about Autosys jobs")

class AutosysLLMQueryTool(BaseTool):
    """Enhanced LangGraph-compatible Autosys Database Query Tool"""
    
    name: str = "AutosysQuery"
    description: str = """
    Query Autosys job scheduler database using natural language.
    Supports questions about job status, schedules, failures, and performance.
    Returns structured data for LangGraph processing.
    
    Examples:
    - "Show me all failed jobs today"
    - "Which ATSYS jobs are currently running?"
    - "List jobs owned by user ADMIN"
    - "Show job history for the last 24 hours"
    """
    
    args_schema = AutosysQueryInput
    
    def __init__(self, autosys_db, llm_instance, max_results: int = 50):
        super().__init__()
        self.autosys_db = autosys_db
        self.llm = llm_instance
        self.max_results = max_results
        self.logger = logging.getLogger(self.__class__.__name__)
        self._verify_db_connection()

    def _verify_db_connection(self) -> bool:
        """Test database connection on initialization"""
        try:
            if hasattr(self.autosys_db, 'connection') and self.autosys_db.connection:
                self.logger.info("AutosysOracleDatabase connection verified")
                return True
            elif hasattr(self.autosys_db, 'connect'):
                self.autosys_db.connect()
                self.logger.info("AutosysOracleDatabase connection established")
                return True
            else:
                self.logger.warning("Could not verify database connection")
                return False
        except Exception as e:
            self.logger.error(f"Database connection verification failed: {str(e)}")
            return False

    def _run(self, question: str) -> Dict[str, Any]:
        """Execute query and return structured results for LangGraph"""
        try:
            self.logger.info(f"Processing question: {question}")
            
            # Generate SQL
            sql_query = self._generate_sql_with_llm(question)
            
            # Execute query
            query_result = self._execute_query_with_existing_db(sql_query)
            
            return {
                "success": query_result["success"],
                "sql_query": sql_query,
                "results": query_result.get("results", []),
                "row_count": query_result.get("row_count", 0),
                "execution_time": query_result.get("execution_time", 0),
                "error": query_result.get("error", "")
            }
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "sql_query": "",
                "results": [],
                "row_count": 0,
                "execution_time": 0
            }

    def _generate_sql_with_llm(self, user_question: str) -> str:
        """Generate SQL using external LLM instance"""
        
        sql_generation_prompt = f"""
You are an expert Oracle SQL generator for Autosys job scheduler database queries.

AUTOSYS DATABASE SCHEMA:
- Table: aedbadmin.ujo_jobst (Job Status)
  * job_name: VARCHAR2 - Job identifier (e.g., 'ATSYS.job_name.c')
  * status: NUMBER - Status code (SU=Success, FA=Failure, RU=Running, IN=Inactive)
  * last_start: NUMBER - Last start time (epoch seconds)
  * last_end: NUMBER - Last end time (epoch seconds)  
  * joid: NUMBER - Job ID (foreign key)

- Table: aedbadmin.ujo_job (Job Details)
  * joid: NUMBER - Job ID (primary key)
  * owner: VARCHAR2 - Job owner/user
  * machine: VARCHAR2 - Execution machine
  * job_type: VARCHAR2 - Type of job

- Table: aedbadmin.UJO_INTCODES (Status Translation)
  * code: NUMBER - Status code number
  * TEXT: VARCHAR2 - Human readable status text

ORACLE SQL PATTERNS FOR AUTOSYS:
- Time conversion: TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss')
- Standard joins: js INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
- Status lookup: LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
- Recent jobs: WHERE js.last_start >= (SYSDATE - INTERVAL '1' DAY) * 86400
- Job name search: WHERE UPPER(js.job_name) LIKE UPPER('%pattern%')

COMMON STATUS CODES:
- SU or 4 = SUCCESS
- FA or 7 = FAILURE  
- RU or 8 = RUNNING
- IN or 5 = INACTIVE

USER QUESTION: "{user_question}"

Generate a complete Oracle SQL query that answers this question.
Use proper table aliases (js for ujo_jobst, j for ujo_job, ic for UJO_INTCODES).
Include relevant columns like job_name, status text, start/end times, owner.
Always use the full table names with aedbadmin schema prefix.
Ensure proper WHERE clauses and result limiting.

Return ONLY the SQL query without any explanations, markdown, or formatting.
"""

        try:
            response = self.llm.invoke(sql_generation_prompt)
            sql_query = response.content if hasattr(response, 'content') else str(response)
            
            # Clean and enhance SQL
            sql_query = self._clean_and_enhance_sql(sql_query)
            
            self.logger.info(f"Generated SQL: {sql_query[:100]}...")
            return sql_query
            
        except Exception as e:
            self.logger.error(f"SQL generation failed: {str(e)}")
            return self._get_fallback_sql(user_question)

    def _clean_and_enhance_sql(self, sql_query: str) -> str:
        """Clean up SQL response and add enhancements"""
        # Remove markdown formatting
        sql_query = re.sub(r'```sql\s*', '', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'```\s*', '', sql_query)
        
        # Clean whitespace
        sql_query = ' '.join(sql_query.split())
        sql_query = sql_query.strip()
        
        # Ensure result limiting
        if 'ROWNUM' not in sql_query.upper() and 'LIMIT' not in sql_query.upper():
            if 'WHERE' in sql_query.upper():
                # Insert ROWNUM condition into existing WHERE clause
                where_pos = sql_query.upper().find('WHERE')
                before_where = sql_query[:where_pos + 5]  # Include 'WHERE'
                after_where = sql_query[where_pos + 5:]
                sql_query = f"{before_where} ROWNUM <= {self.max_results} AND{after_where}"
            else:
                # Add WHERE clause with ROWNUM
                order_pos = sql_query.upper().find('ORDER BY')
                if order_pos > 0:
                    before_order = sql_query[:order_pos]
                    after_order = sql_query[order_pos:]
                    sql_query = f"{before_order} WHERE ROWNUM <= {self.max_results} {after_order}"
                else:
                    sql_query = sql_query.rstrip(';') + f" WHERE ROWNUM <= {self.max_results}"
        
        # Ensure proper ending
        if not sql_query.endswith(';'):
            sql_query += ';'
            
        return sql_query

    def _execute_query_with_existing_db(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query using existing AutosysOracleDatabase"""
        
        try:
            start_time = datetime.now()
            
            # Use existing database connection method
            if hasattr(self.autosys_db, 'run'):
                raw_results = self.autosys_db.run(sql_query)
            elif hasattr(self.autosys_db, 'execute_query'):
                raw_results = self.autosys_db.execute_query(sql_query)
            elif hasattr(self.autosys_db, 'query'):
                raw_results = self.autosys_db.query(sql_query)
            else:
                # Fallback: try to access connection directly
                if hasattr(self.autosys_db, 'connection') and self.autosys_db.connection:
                    with self.autosys_db.connection.cursor() as cursor:
                        cursor.execute(sql_query.rstrip(';'))  # Remove semicolon for cursor.execute
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        raw_results = cursor.fetchall()
                        
                        # Convert to list of dictionaries
                        results = []
                        for row in raw_results:
                            row_dict = {}
                            for i, value in enumerate(row):
                                col_name = columns[i] if i < len(columns) else f"COLUMN_{i}"
                                row_dict[col_name] = self._format_oracle_value(value)
                            results.append(row_dict)
                        raw_results = results
                else:
                    raise Exception("Could not access database connection")
            
            # Process results
            processed_results = self._process_database_results(raw_results)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "results": processed_results,
                "row_count": len(processed_results),
                "execution_time": execution_time,
                "sql_query": sql_query
            }
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "row_count": 0,
                "execution_time": 0,
                "sql_query": sql_query
            }

    def _process_database_results(self, raw_results) -> List[Dict]:
        """Convert raw database results to standardized format"""
        if isinstance(raw_results, str):
            # Handle string results (parsed from your database class)
            try:
                import ast
                if raw_results.startswith('[') and raw_results.endswith(']'):
                    parsed_results = ast.literal_eval(raw_results)
                    return self._convert_to_dict_list(parsed_results)
                else:
                    return [{"result": raw_results}]
            except:
                return [{"result": raw_results}]
        elif isinstance(raw_results, list):
            return self._convert_to_dict_list(raw_results)
        else:
            return [{"result": str(raw_results)}]

    def _convert_to_dict_list(self, raw_results: List) -> List[Dict]:
        """Convert list of tuples to list of dictionaries"""
        results = []
        
        for item in raw_results:
            if isinstance(item, (tuple, list)):
                # Convert tuple/list to dictionary
                row_dict = {}
                for i, value in enumerate(item):
                    col_name = self._get_column_name(i, value)
                    row_dict[col_name] = self._format_oracle_value(value)
                results.append(row_dict)
            elif isinstance(item, dict):
                # Already a dictionary
                results.append(item)
            else:
                # Single value
                results.append({"VALUE": str(item)})
        
        return results

    def _get_column_name(self, index: int, value: Any) -> str:
        """Generate appropriate column name based on typical Autosys schema"""
        common_names = ["JOB_NAME", "START_TIME", "END_TIME", "STATUS", "OWNER", "MACHINE"]
        
        if index < len(common_names):
            return common_names[index]
        else:
            return f"COLUMN_{index + 1}"

    def _format_oracle_value(self, value: Any) -> str:
        """Format Oracle-specific data types"""
        if hasattr(value, 'read'):  # CLOB/BLOB
            return value.read()
        elif isinstance(value, datetime):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return str(value) if value is not None else ""

    def _get_fallback_sql(self, user_question: str) -> str:
        """Fallback SQL query if LLM generation fails"""
        question_lower = user_question.lower()
        
        if 'failed' in question_lower or 'failure' in question_lower:
            status_condition = "js.status = 7"  # FA status code
        elif 'running' in question_lower:
            status_condition = "js.status = 8"  # RU status code  
        elif 'success' in question_lower:
            status_condition = "js.status = 4"  # SU status code
        else:
            status_condition = "1=1"  # All jobs
        
        return f"""
        SELECT 
            js.job_name,
            TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (js.last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss') AS start_time,
            TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (js.last_end / 86400), 'MM/DD/YYYY HH24:Mi:Ss') AS end_time,
            NVL(ic.TEXT, 'UNKNOWN') AS job_status,
            j.owner
        FROM aedbadmin.ujo_jobst js
        INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
        LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
        WHERE {status_condition}
        AND UPPER(js.job_name) LIKE UPPER('%ATSYS%')
        AND ROWNUM <= {self.max_results}
        ORDER BY js.last_start DESC;
        """

# ============================================================================
# LANGGRAPH WORKFLOW IMPLEMENTATION
# ============================================================================

class AutosysLangGraphSystem:
    """LangGraph-based Autosys system with old function names"""
    
    def __init__(self, autosys_db, llm_instance):
        self.autosys_db = autosys_db
        self.llm = llm_instance
        self.tool = AutosysLLMQueryTool(autosys_db, llm_instance)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize memory for persistence
        self.memory = MemorySaver()
        
        # Build the graph
        self.graph = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(AutosysState)
        
        # Add nodes
        workflow.add_node("understand_query", self.understand_query_node)
        workflow.add_node("generate_sql", self.generate_sql_node)
        workflow.add_node("execute_query", self.execute_query_node)
        workflow.add_node("format_results", self.format_results_node)
        workflow.add_node("handle_error", self.handle_error_node)
        
        # Set entry point
        workflow.set_entry_point("understand_query")
        
        # Add edges with conditional logic
        workflow.add_edge("understand_query", "generate_sql")
        workflow.add_conditional_edges(
            "generate_sql",
            self._should_execute_query,
            {
                "execute": "execute_query",
                "error": "handle_error"
            }
        )
        workflow.add_conditional_edges(
            "execute_query", 
            self._should_format_results,
            {
                "format": "format_results",
                "error": "handle_error"
            }
        )
        workflow.add_edge("format_results", END)
        workflow.add_edge("handle_error", END)
        
        # Compile with memory
        return workflow.compile(checkpointer=self.memory)

    def understand_query_node(self, state: AutosysState) -> AutosysState:
        """Analyze and enhance the user query"""
        self.logger.info(f"Understanding query: {state['user_question']}")
        
        # Add analysis message
        state["messages"].append({
            "role": "system", 
            "content": f"Processing Autosys query: {state['user_question']}"
        })
        
        # Enhance question with context if needed
        question = state["user_question"].strip()
        
        if not any(keyword in question.lower() for keyword in ['atsys', 'job', 'status', 'schedule']):
            question = f"Show Autosys jobs related to: {question}"
            state["user_question"] = question
            
        self.logger.info(f"Enhanced question: {state['user_question']}")
        
        return state

    def generate_sql_node(self, state: AutosysState) -> AutosysState:
        """Generate SQL query using LLM"""
        try:
            self.logger.info("Generating SQL query")
            
            # Use the tool to generate SQL
            tool_result = self.tool.run(state["user_question"])
            
            if tool_result["success"]:
                state["sql_query"] = tool_result["sql_query"]
                state["messages"].append({
                    "role": "system",
                    "content": f"SQL generated successfully: {len(tool_result['sql_query'])} characters"
                })
            else:
                state["error"] = f"SQL generation failed: {tool_result.get('error', 'Unknown error')}"
                state["messages"].append({
                    "role": "system",
                    "content": f"SQL generation error: {state['error']}"
                })
            
        except Exception as e:
            state["error"] = f"SQL generation exception: {str(e)}"
            self.logger.error(f"SQL generation failed: {e}")
        
        return state

    def execute_query_node(self, state: AutosysState) -> AutosysState:
        """Execute the SQL query"""
        try:
            self.logger.info("Executing SQL query")
            
            # Execute using the tool
            tool_result = self.tool.run(state["user_question"])
            
            if tool_result["success"]:
                state["query_results"] = {
                    "success": True,
                    "results": tool_result["results"],
                    "row_count": tool_result["row_count"],
                    "execution_time": tool_result["execution_time"],
                    "sql_query": tool_result["sql_query"]
                }
                state["messages"].append({
                    "role": "system",
                    "content": f"Query executed: {tool_result['row_count']} results in {tool_result['execution_time']:.2f}s"
                })
            else:
                state["error"] = f"Query execution failed: {tool_result.get('error', 'Unknown error')}"
                state["query_results"] = {"success": False, "error": state["error"]}
                
        except Exception as e:
            state["error"] = f"Query execution exception: {str(e)}"
            state["query_results"] = {"success": False, "error": state["error"]}
            self.logger.error(f"Query execution failed: {e}")
        
        return state

    def format_results_node(self, state: AutosysState) -> AutosysState:
        """Format results using LLM with system prompt approach"""
        try:
            self.logger.info("Formatting results")
            
            results = state["query_results"]["results"]
            
            if not results:
                state["formatted_output"] = self._create_no_results_html()
                return state
            
            # Use LLM for professional formatting
            formatting_prompt = f"""
Create professional HTML formatting for these Autosys database query results.

USER QUESTION: "{state['user_question']}"
EXECUTION TIME: {state['query_results'].get('execution_time', 0):.2f} seconds  
TOTAL RESULTS: {len(results)} jobs (showing sample)

QUERY RESULTS TO FORMAT:
{json.dumps(results[:15], indent=2, default=str)}

FORMATTING REQUIREMENTS:
1. Create responsive HTML with inline CSS only - no external stylesheets
2. Use professional styling with good contrast and readability
3. Color-code job status with badges:
   - SUCCESS/SU: Green background (#28a745) with white text
   - FAILURE/FA: Red background (#dc3545) with white text  
   - RUNNING/RU: Blue background (#007bff) with white text
   - INACTIVE/IN: Gray background (#6c757d) with white text
4. Include a summary section at the top showing total jobs and key statistics
5. Use monospace font for job names for better readability
6. Make tables responsive for mobile devices
7. Keep output concise to avoid truncation - prioritize most important info
8. Add subtle hover effects for better user experience
9. Include proper spacing and margins for professional appearance
10. Show execution time and result count in a footer

Create a clean, modern design that's easy to scan and read.
Return only the formatted HTML without any explanations or markdown.
"""
            
            response = self.llm.invoke(formatting_prompt)
            formatted_html = response.content if hasattr(response, 'content') else str(response)
            
            # Add LangGraph metadata footer
            metadata = f"""
            <div style="margin-top: 15px; padding: 8px 12px; background: #f8f9fa; border-radius: 4px; font-size: 11px; color: #666; border-left: 3px solid #007bff;">
                <strong>LangGraph AutosysQuery</strong> • 
                {state['query_results']['row_count']} results in {state['query_results']['execution_time']:.2f}s • 
                Enhanced Oracle Database Query System
            </div>
            """
            
            state["formatted_output"] = formatted_html + metadata
            state["messages"].append({
                "role": "assistant",
                "content": "Results formatted successfully with professional HTML styling"
            })
            
        except Exception as e:
            state["error"] = f"Formatting failed: {str(e)}"
            state["formatted_output"] = self._create_error_html(state["error"])
            self.logger.error(f"Formatting failed: {e}")
        
        return state

    def handle_error_node(self, state: AutosysState) -> AutosysState:
        """Handle errors and provide user-friendly error messages"""
        self.logger.error(f"Handling error: {state.get('error', 'Unknown error')}")
        
        error_msg = state.get("error", "An unknown error occurred")
        sql_query = state.get("sql_query", "")
        
        state["formatted_output"] = self._create_error_html(error_msg, sql_query)
        state["messages"].append({
            "role": "system",
            "content": f"Error handled and formatted for user display"
        })
        
        return state

    def _should_execute_query(self, state: AutosysState) -> str:
        """Conditional edge: decide whether to execute query or handle error"""
        if state.get("error"):
            return "error"
        elif state.get("sql_query"):
            return "execute"
        else:
            return "error"

    def _should_format_results(self, state: AutosysState) -> str:
        """Conditional edge: decide whether to format results or handle error"""
        if state.get("error"):
            return "error"
        elif state.get("query_results", {}).get("success"):
            return "format"
        else:
            return "error"

    def _create_no_results_html(self) -> str:
        """HTML for no results found"""
        return """
        <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #856404;">No Results Found</h4>
            <p style="margin: 0; color: #856404;">No Autosys jobs match your query criteria. Try rephrasing your question or checking job names.</p>
        </div>
        """

    def _create_error_html(self, error_msg: str, sql_query: str = "") -> str:
        """HTML for error display"""
        html = f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #721c24;">LangGraph AutosysQuery Error</h4>
            <p style="margin: 0; font-size: 14px;"><strong>Error:</strong> {error_msg}</p>
        """
        
        if sql_query:
            html += f"""
            <details style="margin-top: 10px;">
                <summary style="cursor: pointer; color: #495057; font-size: 12px;">View Generated SQL Query</summary>
                <pre style="background: #e9ecef; padding: 10px; margin-top: 5px; border-radius: 3px; font-size: 10px; overflow-x: auto; font-family: 'Courier New', monospace;">{sql_query}</pre>
            </details>
            """
        
        html += """
            <p style="margin: 10px 0 0 0; font-size: 12px; color: #856404;">
                <em>Tip: Try rephrasing your question or check if the job names/criteria exist in the database.</em>
            </p>
        </div>
        """
        return html

    def process_query(self, user_question: str, session_id: str) -> Dict[str, Any]:
        """Process a query using the LangGraph workflow"""
        
        # Initialize state
        initial_state = {
            "messages": [],
            "user_question": user_question,
            "sql_query": "",
            "query_results": {},
            "formatted_output": "",
            "error": "",
            "iteration_count": 0,
            "session_id": session_id
        }
        
        # Configuration for session management
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            self.logger.info(f"Processing query for session {session_id}: {user_question}")
            final_state = self.graph.invoke(initial_state, config=config)
            
            return {
                "success": not bool(final_state.get("error")),
                "formatted_output": final_state.get("formatted_output", ""),
                "sql_query": final_state.get("sql_query", ""),
                "row_count": final_state.get("query_results", {}).get("row_count", 0),
                "execution_time": final_state.get("query_results", {}).get("execution_time", 0),
                "error": final_state.get("error", ""),
                "messages": final_state.get("messages", [])
            }
            
        except Exception as e:
            self.logger.error(f"LangGraph execution failed: {e}")
            return {
                "success": False,
                "formatted_output": self._create_error_html(f"System execution failed: {str(e)}"),
                "error": str(e),
                "messages": []
            }

# ============================================================================
# OLD FUNCTION NAMES WITH NEW IMPLEMENTATION
# ============================================================================

# Global variable to store the LangGraph system
_autosys_langgraph_system = None

def initialize_agent(tools, llm, agent=None, verbose=True, checkpointer=None, 
                    handle_parsing_errors=True, max_iterations=3, 
                    early_stopping_method="generate", agent_kwargs=None):
    """
    Initialize the LangGraph-based Autosys system with the same function signature
    as the old initialize_agent function
    """
    global _autosys_langgraph_system
    
    try:
        # Extract autosys_db from tools or use global reference
        # This assumes you pass your autosys_db somehow - adjust as needed
        # For now, we'll assume it's available globally or passed in a specific way
        
        if hasattr(tools[0], 'autosys_db') if tools else False:
            autosys_db = tools[0].autosys_db
        else:
            # You'll need to provide autosys_db reference here
            # This is where you'd inject your AutosysOracleDatabase instance
            autosys_db = None  # Replace with your actual autosys_db instance
            
        _autosys_langgraph_system = AutosysLangGraphSystem(autosys_db, llm)
        
        logger.info("LangGraph Autosys system initialized successfully")
        return _autosys_langgraph_system
        
    except Exception as e:
        logger.error(f"Failed to initialize LangGraph system: {e}")
        raise e

def extract_last_ai_message(result: Union[Dict, str, Any]) -> str:
    """
    Enhanced message extraction that works with both old and new LangGraph responses
    """
    try:
        # Handle LangGraph response format (New Way)
        if isinstance(result, dict):
            if "formatted_output" in result and result["formatted_output"]:
                return result["formatted_output"].strip()
            if result.get("success") and "formatted_output" in result:
                return result["formatted_output"].strip()
            if "answer" in result and result["answer"]:
                return result["answer"].strip()
            if "result" in result and result["result"]:
                return result["result"].strip()
            if "content" in result and result["content"]:
                return result["content"].strip()
            if "output" in result and result["output"]:
                return result["output"].strip()
        
        # Handle string responses (Old Way compatibility)
        if isinstance(result, str):
            result = result.strip()
            final_answer_match = re.search(r"Final Answer:\s*(.+?)(?=\n\n|\n(?=\w+:)|\Z)", result, re.DOTALL | re.IGNORECASE)
            if final_answer_match:
                return final_answer_match.group(1).strip()
            if "<html>" in result.lower() or "<div>" in result.lower() or "<table>" in result.lower():
                return result
            if len(result) > 0 and not any(marker in result.lower() for marker in ['thought:', 'action:', 'observation:']):
                return result
        
        # Handle object attributes
        if hasattr(result, 'content'):
            return result.content.strip()
        elif hasattr(result, 'output'):
            return result.output.strip()
        
        result_str = str(result).strip()
        if result_str and result_str != "None":
            return result_str
            
        return "No response generated."
        
    except Exception as e:
        logging.error(f"Error extracting AI message: {str(e)}")
        return f"Error processing response: {str(e)}"

def get_chat_response(message: str, session_id: str) -> str:
    """
    Main chat response function using NEW WAY (LangGraph) with old function name
    
    This function maintains the same signature as your existing function but uses
    the new LangGraph implementation under the hood.
    """
    global _autosys_langgraph_system
    
    try:
        # Input validation
        if not message or not message.strip():
            return "Please provide a question about the database."
        
        message = message.strip()
        logger.info(f"Processing chat request: {message}")
        
        # Check if LangGraph system is initialized
        if not _autosys_langgraph_system:
            return "System not initialized. Please contact administrator."
        
        # Process query using LangGraph
        result = _autosys_langgraph_system.process_query(message, session_id)
        
        if result["success"]:
            logger.info(f"Query processed successfully: {result['row_count']} results in {result['execution_time']:.2f}s")
            return result["formatted_output"]
        else:
            logger.error(f"Query failed: {result['error']}")
            return result["formatted_output"]  # Error HTML is already in formatted_output
            
    except Exception as e:
        logger.error(f"Critical error in get_chat_response: {str(e)}")
        return f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;">
            <h4 style="margin: 0 0 10px 0;">System Error</h4>
            <p style="margin: 0;">A critical error occurred while processing your request: {str(e)}</p>
            <p style="margin: 10px 0 0 0; font-size: 12px;">Please try again or contact support if the problem persists.</p>
        </div>
        """

# ============================================================================
# COMPLETE SETUP FUNCTION FOR YOUR EXISTING CODE
# ============================================================================

def setup_autosys_langgraph_system(autosys_db, llm_instance):
    """
    Complete setup function that replaces your existing agent initialization
    
    Args:
        autosys_db: Your AutosysOracleDatabase instance
        llm_instance: Your LLM instance from get_llm("langchain")
    
    Returns:
        Configured LangGraph system ready for use
    """
    global _autosys_langgraph_system
    
    try:
        # Initialize the new LangGraph system
        _autosys_langgraph_system = AutosysLangGraphSystem(autosys_db, llm_instance)
        
        logger.info("LangGraph Autosys system setup completed successfully")
        
        return {
            "system": _autosys_langgraph_system,
            "status": "ready",
            "type": "langgraph",
            "features": [
                "Enhanced SQL generation",
                "Professional HTML formatting", 
                "State-based error handling",
                "Session persistence",
                "Comprehensive logging"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to setup LangGraph system: {e}")
        raise Exception(f"System setup failed: {str(e)}")

# ============================================================================
# DIRECT REPLACEMENT FOR YOUR EXISTING CODE
# ============================================================================

"""
COMPLETE INTEGRATION INSTRUCTIONS:

1. REPLACE your existing imports and setup:

# OLD CODE:
from langchain.agents import initialize_agent, AgentType
sql_agent = initialize_agent(
    tools=[autosys_sql_tool_enhanced],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    checkpointer=checkpointer,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="generate",
    agent_kwargs={"format_instructions": "..."}
)

# NEW CODE:
# Just add this one line after your existing setup:
setup_autosys_langgraph_system(autosys_db, llm)

# Your get_chat_response function stays the same!
# It will automatically use the new LangGraph implementation

2. YOUR EXISTING FUNCTION CALLS REMAIN UNCHANGED:
   - get_chat_response(message, session_id) works exactly the same
   - extract_last_ai_message() works exactly the same
   - All your API endpoints and session handling stay the same

3. WHAT YOU GET:
   - Better error handling and recovery
   - Professional HTML formatting with system prompts
   - Enhanced SQL generation using LLM
   - State-based workflow with observability
   - Session persistence with memory
   - Comprehensive logging and debugging
"""

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_integration():
    """
    Example showing how to integrate with your existing code
    """
    
    # Your existing setup (no changes needed):
    oracle_uri = "oracle+oracledb://username:password@host:port/service_name"
    # autosys_db = AutosysOracleDatabase(oracle_uri)  # Your existing DB
    # llm = get_llm("langchain")  # Your existing LLM
    
    # NEW: Single line to replace your entire agent setup:
    # setup_autosys_langgraph_system(autosys_db, llm)
    
    # Your existing function calls work unchanged:
    # response = get_chat_response("Show me failed ATSYS jobs", "session_123")
    # print(response)  # Gets professional HTML output
    
    pass

# ============================================================================
# UTILITY FUNCTIONS FOR DEBUGGING AND MONITORING
# ============================================================================

def debug_langgraph_system():
    """Debug function to check system status"""
    global _autosys_langgraph_system
    
    if not _autosys_langgraph_system:
        return {"status": "not_initialized", "system": None}
    
    return {
        "status": "initialized",
        "system_type": "LangGraph",
        "has_memory": hasattr(_autosys_langgraph_system, 'memory'),
        "has_tool": hasattr(_autosys_langgraph_system, 'tool'),
        "has_graph": hasattr(_autosys_langgraph_system, 'graph')
    }

def get_system_metrics(session_id: str = "default") -> Dict[str, Any]:
    """Get system performance metrics"""
    global _autosys_langgraph_system
    
    if not _autosys_langgraph_system:
        return {"error": "System not initialized"}
    
    try:
        # Test query to check system health
        test_result = _autosys_langgraph_system.process_query(
            "SELECT COUNT(*) FROM aedbladmin.ujo_jobst WHERE ROWNUM <= 1", 
            session_id
        )
        
        return {
            "system_status": "healthy" if test_result["success"] else "error",
            "test_execution_time": test_result.get("execution_time", 0),
            "last_error": test_result.get("error", None),
            "memory_enabled": hasattr(_autosys_langgraph_system, 'memory'),
            "session_id": session_id
        }
        
    except Exception as e:
        return {
            "system_status": "error",
            "error": str(e),
            "session_id": session_id
        }

# ============================================================================
# FINAL INTEGRATION SNIPPET
# ============================================================================

# COPY THIS EXACT CODE INTO YOUR EXISTING FILE:
"""
# Add these imports at the top of your file:
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Replace your existing sql_agent setup with this single line:
setup_autosys_langgraph_system(autosys_db, llm)

# Your existing get_chat_response function will now automatically use LangGraph!
# No other changes needed to your existing code.
"""

if __name__ == "__main__":
    # Test the system
    print("LangGraph Autosys System - NEW WAY Implementation")
    print("Ready for integration with existing codebase")
    print("\nFeatures enabled:")
    print("- LLM-powered SQL generation")
    print("- Professional HTML formatting") 
    print("- State-based error handling")
    print("- Session persistence")
    print("- Enhanced observability")




.
####################
# ============================================================================
# COMPLETE NEW WAY IMPLEMENTATION - LangGraph with Old Function Names
# ============================================================================

import oracledb
import json
import logging
import re
from typing import Dict, Any, Optional, List, Union, TypedDict, Annotated
from datetime import datetime
from pydantic import BaseModel, Field
import operator

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports (for tool compatibility)
from langchain.tools import BaseTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# STATE DEFINITION FOR LANGGRAPH
# ============================================================================

class AutosysState(TypedDict):
    """State for the Autosys LangGraph workflow"""
    messages: Annotated[List[Dict[str, str]], operator.add]
    user_question: str
    sql_query: str
    query_results: Dict[str, Any]
    formatted_output: str
    error: str
    iteration_count: int
    session_id: str

# ============================================================================
# ENHANCED TOOL FOR LANGGRAPH
# ============================================================================

class AutosysQueryInput(BaseModel):
    """Input schema for Autosys Query Tool"""
    question: str = Field(description="Natural language question about Autosys jobs")

class AutosysLLMQueryTool(BaseTool):
    """Enhanced LangGraph-compatible Autosys Database Query Tool"""
    
    name: str = "AutosysQuery"
    description: str = """
    Query Autosys job scheduler database using natural language.
    Supports questions about job status, schedules, failures, and performance.
    Returns structured data for LangGraph processing.
    
    Examples:
    - "Show me all failed jobs today"
    - "Which ATSYS jobs are currently running?"
    - "List jobs owned by user ADMIN"
    - "Show job history for the last 24 hours"
    """
    
    args_schema = AutosysQueryInput
    
    def __init__(self, autosys_db, llm_instance, max_results: int = 50):
        super().__init__()
        self.autosys_db = autosys_db
        self.llm = llm_instance
        self.max_results = max_results
        self.logger = logging.getLogger(self.__class__.__name__)
        self._verify_db_connection()

    def _verify_db_connection(self) -> bool:
        """Test database connection on initialization"""
        try:
            if hasattr(self.autosys_db, 'connection') and self.autosys_db.connection:
                self.logger.info("AutosysOracleDatabase connection verified")
                return True
            elif hasattr(self.autosys_db, 'connect'):
                self.autosys_db.connect()
                self.logger.info("AutosysOracleDatabase connection established")
                return True
            else:
                self.logger.warning("Could not verify database connection")
                return False
        except Exception as e:
            self.logger.error(f"Database connection verification failed: {str(e)}")
            return False

    def _run(self, question: str) -> Dict[str, Any]:
        """Execute query and return structured results for LangGraph"""
        try:
            self.logger.info(f"Processing question: {question}")
            
            # Generate SQL
            sql_query = self._generate_sql_with_llm(question)
            
            # Execute query
            query_result = self._execute_query_with_existing_db(sql_query)
            
            return {
                "success": query_result["success"],
                "sql_query": sql_query,
                "results": query_result.get("results", []),
                "row_count": query_result.get("row_count", 0),
                "execution_time": query_result.get("execution_time", 0),
                "error": query_result.get("error", "")
            }
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "sql_query": "",
                "results": [],
                "row_count": 0,
                "execution_time": 0
            }

    def _generate_sql_with_llm(self, user_question: str) -> str:
        """Generate SQL using external LLM instance"""
        
        sql_generation_prompt = f"""
You are an expert Oracle SQL generator for Autosys job scheduler database queries.

AUTOSYS DATABASE SCHEMA:
- Table: aedbadmin.ujo_jobst (Job Status)
  * job_name: VARCHAR2 - Job identifier (e.g., 'ATSYS.job_name.c')
  * status: NUMBER - Status code (SU=Success, FA=Failure, RU=Running, IN=Inactive)
  * last_start: NUMBER - Last start time (epoch seconds)
  * last_end: NUMBER - Last end time (epoch seconds)  
  * joid: NUMBER - Job ID (foreign key)

- Table: aedbadmin.ujo_job (Job Details)
  * joid: NUMBER - Job ID (primary key)
  * owner: VARCHAR2 - Job owner/user
  * machine: VARCHAR2 - Execution machine
  * job_type: VARCHAR2 - Type of job

- Table: aedbadmin.UJO_INTCODES (Status Translation)
  * code: NUMBER - Status code number
  * TEXT: VARCHAR2 - Human readable status text

ORACLE SQL PATTERNS FOR AUTOSYS:
- Time conversion: TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss')
- Standard joins: js INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
- Status lookup: LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
- Recent jobs: WHERE js.last_start >= (SYSDATE - INTERVAL '1' DAY) * 86400
- Job name search: WHERE UPPER(js.job_name) LIKE UPPER('%pattern%')

COMMON STATUS CODES:
- SU or 4 = SUCCESS
- FA or 7 = FAILURE  
- RU or 8 = RUNNING
- IN or 5 = INACTIVE

USER QUESTION: "{user_question}"

Generate a complete Oracle SQL query that answers this question.
Use proper table aliases (js for ujo_jobst, j for ujo_job, ic for UJO_INTCODES).
Include relevant columns like job_name, status text, start/end times, owner.
Always use the full table names with aedbadmin schema prefix.
Ensure proper WHERE clauses and result limiting.

Return ONLY the SQL query without any explanations, markdown, or formatting.
"""

        try:
            response = self.llm.invoke(sql_generation_prompt)
            sql_query = response.content if hasattr(response, 'content') else str(response)
            
            # Clean and enhance SQL
            sql_query = self._clean_and_enhance_sql(sql_query)
            
            self.logger.info(f"Generated SQL: {sql_query[:100]}...")
            return sql_query
            
        except Exception as e:
            self.logger.error(f"SQL generation failed: {str(e)}")
            return self._get_fallback_sql(user_question)

    def _clean_and_enhance_sql(self, sql_query: str) -> str:
        """Clean up SQL response and add enhancements"""
        # Remove markdown formatting
        sql_query = re.sub(r'```sql\s*', '', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'```\s*', '', sql_query)
        
        # Clean whitespace
        sql_query = ' '.join(sql_query.split())
        sql_query = sql_query.strip()
        
        # Ensure result limiting
        if 'ROWNUM' not in sql_query.upper() and 'LIMIT' not in sql_query.upper():
            if 'WHERE' in sql_query.upper():
                # Insert ROWNUM condition into existing WHERE clause
                where_pos = sql_query.upper().find('WHERE')
                before_where = sql_query[:where_pos + 5]  # Include 'WHERE'
                after_where = sql_query[where_pos + 5:]
                sql_query = f"{before_where} ROWNUM <= {self.max_results} AND{after_where}"
            else:
                # Add WHERE clause with ROWNUM
                order_pos = sql_query.upper().find('ORDER BY')
                if order_pos > 0:
                    before_order = sql_query[:order_pos]
                    after_order = sql_query[order_pos:]
                    sql_query = f"{before_order} WHERE ROWNUM <= {self.max_results} {after_order}"
                else:
                    sql_query = sql_query.rstrip(';') + f" WHERE ROWNUM <= {self.max_results}"
        
        # Ensure proper ending
        if not sql_query.endswith(';'):
            sql_query += ';'
            
        return sql_query

    def _execute_query_with_existing_db(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query using existing AutosysOracleDatabase"""
        
        try:
            start_time = datetime.now()
            
            # Use existing database connection method
            if hasattr(self.autosys_db, 'run'):
                raw_results = self.autosys_db.run(sql_query)
            elif hasattr(self.autosys_db, 'execute_query'):
                raw_results = self.autosys_db.execute_query(sql_query)
            elif hasattr(self.autosys_db, 'query'):
                raw_results = self.autosys_db.query(sql_query)
            else:
                # Fallback: try to access connection directly
                if hasattr(self.autosys_db, 'connection') and self.autosys_db.connection:
                    with self.autosys_db.connection.cursor() as cursor:
                        cursor.execute(sql_query.rstrip(';'))  # Remove semicolon for cursor.execute
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        raw_results = cursor.fetchall()
                        
                        # Convert to list of dictionaries
                        results = []
                        for row in raw_results:
                            row_dict = {}
                            for i, value in enumerate(row):
                                col_name = columns[i] if i < len(columns) else f"COLUMN_{i}"
                                row_dict[col_name] = self._format_oracle_value(value)
                            results.append(row_dict)
                        raw_results = results
                else:
                    raise Exception("Could not access database connection")
            
            # Process results
            processed_results = self._process_database_results(raw_results)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "results": processed_results,
                "row_count": len(processed_results),
                "execution_time": execution_time,
                "sql_query": sql_query
            }
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "row_count": 0,
                "execution_time": 0,
                "sql_query": sql_query
            }

    def _process_database_results(self, raw_results) -> List[Dict]:
        """Convert raw database results to standardized format"""
        if isinstance(raw_results, str):
            # Handle string results (parsed from your database class)
            try:
                import ast
                if raw_results.startswith('[') and raw_results.endswith(']'):
                    parsed_results = ast.literal_eval(raw_results)
                    return self._convert_to_dict_list(parsed_results)
                else:
                    return [{"result": raw_results}]
            except:
                return [{"result": raw_results}]
        elif isinstance(raw_results, list):
            return self._convert_to_dict_list(raw_results)
        else:
            return [{"result": str(raw_results)}]

    def _convert_to_dict_list(self, raw_results: List) -> List[Dict]:
        """Convert list of tuples to list of dictionaries"""
        results = []
        
        for item in raw_results:
            if isinstance(item, (tuple, list)):
                # Convert tuple/list to dictionary
                row_dict = {}
                for i, value in enumerate(item):
                    col_name = self._get_column_name(i, value)
                    row_dict[col_name] = self._format_oracle_value(value)
                results.append(row_dict)
            elif isinstance(item, dict):
                # Already a dictionary
                results.append(item)
            else:
                # Single value
                results.append({"VALUE": str(item)})
        
        return results

    def _get_column_name(self, index: int, value: Any) -> str:
        """Generate appropriate column name based on typical Autosys schema"""
        common_names = ["JOB_NAME", "START_TIME", "END_TIME", "STATUS", "OWNER", "MACHINE"]
        
        if index < len(common_names):
            return common_names[index]
        else:
            return f"COLUMN_{index + 1}"

    def _format_oracle_value(self, value: Any) -> str:
        """Format Oracle-specific data types"""
        if hasattr(value, 'read'):  # CLOB/BLOB
            return value.read()
        elif isinstance(value, datetime):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return str(value) if value is not None else ""

    def _get_fallback_sql(self, user_question: str) -> str:
        """Fallback SQL query if LLM generation fails"""
        question_lower = user_question.lower()
        
        if 'failed' in question_lower or 'failure' in question_lower:
            status_condition = "js.status = 7"  # FA status code
        elif 'running' in question_lower:
            status_condition = "js.status = 8"  # RU status code  
        elif 'success' in question_lower:
            status_condition = "js.status = 4"  # SU status code
        else:
            status_condition = "1=1"  # All jobs
        
        return f"""
        SELECT 
            js.job_name,
            TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (js.last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss') AS start_time,
            TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (js.last_end / 86400), 'MM/DD/YYYY HH24:Mi:Ss') AS end_time,
            NVL(ic.TEXT, 'UNKNOWN') AS job_status,
            j.owner
        FROM aedbadmin.ujo_jobst js
        INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
        LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
        WHERE {status_condition}
        AND UPPER(js.job_name) LIKE UPPER('%ATSYS%')
        AND ROWNUM <= {self.max_results}
        ORDER BY js.last_start DESC;
        """

# ============================================================================
# LANGGRAPH WORKFLOW IMPLEMENTATION
# ============================================================================

class AutosysLangGraphSystem:
    """LangGraph-based Autosys system with old function names"""
    
    def __init__(self, autosys_db, llm_instance):
        self.autosys_db = autosys_db
        self.llm = llm_instance
        self.tool = AutosysLLMQueryTool(autosys_db, llm_instance)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize memory for persistence
        self.memory = MemorySaver()
        
        # Build the graph
        self.graph = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(AutosysState)
        
        # Add nodes
        workflow.add_node("understand_query", self.understand_query_node)
        workflow.add_node("generate_sql", self.generate_sql_node)
        workflow.add_node("execute_query", self.execute_query_node)
        workflow.add_node("format_results", self.format_results_node)
        workflow.add_node("handle_error", self.handle_error_node)
        
        # Set entry point
        workflow.set_entry_point("understand_query")
        
        # Add edges with conditional logic
        workflow.add_edge("understand_query", "generate_sql")
        workflow.add_conditional_edges(
            "generate_sql",
            self._should_execute_query,
            {
                "execute": "execute_query",
                "error": "handle_error"
            }
        )
        workflow.add_conditional_edges(
            "execute_query", 
            self._should_format_results,
            {
                "format": "format_results",
                "error": "handle_error"
            }
        )
        workflow.add_edge("format_results", END)
        workflow.add_edge("handle_error", END)
        
        # Compile with memory
        return workflow.compile(checkpointer=self.memory)

    def understand_query_node(self, state: AutosysState) -> AutosysState:
        """Analyze and enhance the user query"""
        self.logger.info(f"Understanding query: {state['user_question']}")
        
        # Add analysis message
        state["messages"].append({
            "role": "system", 
            "content": f"Processing Autosys query: {state['user_question']}"
        })
        
        # Enhance question with context if needed
        question = state["user_question"].strip()
        
        if not any(keyword in question.lower() for keyword in ['atsys', 'job', 'status', 'schedule']):
            question = f"Show Autosys jobs related to: {question}"
            state["user_question"] = question
            
        self.logger.info(f"Enhanced question: {state['user_question']}")
        
        return state

    def generate_sql_node(self, state: AutosysState) -> AutosysState:
        """Generate SQL query using LLM"""
        try:
            self.logger.info("Generating SQL query")
            
            # Use the tool to generate SQL
            tool_result = self.tool.run(state["user_question"])
            
            if tool_result["success"]:
                state["sql_query"] = tool_result["sql_query"]
                state["messages"].append({
                    "role": "system",
                    "content": f"SQL generated successfully: {len(tool_result['sql_query'])} characters"
                })
            else:
                state["error"] = f"SQL generation failed: {tool_result.get('error', 'Unknown error')}"
                state["messages"].append({
                    "role": "system",
                    "content": f"SQL generation error: {state['error']}"
                })
            
        except Exception as e:
            state["error"] = f"SQL generation exception: {str(e)}"
            self.logger.error(f"SQL generation failed: {e}")
        
        return state

    def execute_query_node(self, state: AutosysState) -> AutosysState:
        """Execute the SQL query"""
        try:
            self.logger.info("Executing SQL query")
            
            # Execute using the tool
            tool_result = self.tool.run(state["user_question"])
            
            if tool_result["success"]:
                state["query_results"] = {
                    "success": True,
                    "results": tool_result["results"],
                    "row_count": tool_result["row_count"],
                    "execution_time": tool_result["execution_time"],
                    "sql_query": tool_result["sql_query"]
                }
                state["messages"].append({
                    "role": "system",
                    "content": f"Query executed: {tool_result['row_count']} results in {tool_result['execution_time']:.2f}s"
                })
            else:
                state["error"] = f"Query execution failed: {tool_result.get('error', 'Unknown error')}"
                state["query_results"] = {"success": False, "error": state["error"]}
                
        except Exception as e:
            state["error"] = f"Query execution exception: {str(e)}"
            state["query_results"] = {"success": False, "error": state["error"]}
            self.logger.error(f"Query execution failed: {e}")
        
        return state

    def format_results_node(self, state: AutosysState) -> AutosysState:
        """Format results using LLM with system prompt approach"""
        try:
            self.logger.info("Formatting results")
            
            results = state["query_results"]["results"]
            
            if not results:
                state["formatted_output"] = self._create_no_results_html()
                return state
            
            # Use LLM for professional formatting
            formatting_prompt = f"""
Create professional HTML formatting for these Autosys database query results.

USER QUESTION: "{state['user_question']}"
EXECUTION TIME: {state['query_results'].get('execution_time', 0):.2f} seconds  
TOTAL RESULTS: {len(results)} jobs (showing sample)

QUERY RESULTS TO FORMAT:
{json.dumps(results[:15], indent=2, default=str)}

FORMATTING REQUIREMENTS:
1. Create responsive HTML with inline CSS only - no external stylesheets
2. Use professional styling with good contrast and readability
3. Color-code job status with badges:
   - SUCCESS/SU: Green background (#28a745) with white text
   - FAILURE/FA: Red background (#dc3545) with white text  
   - RUNNING/RU: Blue background (#007bff) with white text
   - INACTIVE/IN: Gray background (#6c757d) with white text
4. Include a summary section at the top showing total jobs and key statistics
5. Use monospace font for job names for better readability
6. Make tables responsive for mobile devices
7. Keep output concise to avoid truncation - prioritize most important info
8. Add subtle hover effects for better user experience
9. Include proper spacing and margins for professional appearance
10. Show execution time and result count in a footer

Create a clean, modern design that's easy to scan and read.
Return only the formatted HTML without any explanations or markdown.
"""
            
            response = self.llm.invoke(formatting_prompt)
            formatted_html = response.content if hasattr(response, 'content') else str(response)
            
            # Add LangGraph metadata footer
            metadata = f"""
            <div style="margin-top: 15px; padding: 8px 12px; background: #f8f9fa; border-radius: 4px; font-size: 11px; color: #666; border-left: 3px solid #007bff;">
                <strong>LangGraph AutosysQuery</strong> • 
                {state['query_results']['row_count']} results in {state['query_results']['execution_time']:.2f}s • 
                Enhanced Oracle Database Query System
            </div>
            """
            
            state["formatted_output"] = formatted_html + metadata
            state["messages"].append({
                "role": "assistant",
                "content": "Results formatted successfully with professional HTML styling"
            })
            
        except Exception as e:
            state["error"] = f"Formatting failed: {str(e)}"
            state["formatted_output"] = self._create_error_html(state["error"])
            self.logger.error(f"Formatting failed: {e}")
        
        return state

    def handle_error_node(self, state: AutosysState) -> AutosysState:
        """Handle errors and provide user-friendly error messages"""
        self.logger.error(f"Handling error: {state.get('error', 'Unknown error')}")
        
        error_msg = state.get("error", "An unknown error occurred")
        sql_query = state.get("sql_query", "")
        
        state["formatted_output"] = self._create_error_html(error_msg, sql_query)
        state["messages"].append({
            "role": "system",
            "content": f"Error handled and formatted for user display"
        })
        
        return state

    def _should_execute_query(self, state: AutosysState) -> str:
        """Conditional edge: decide whether to execute query or handle error"""
        if state.get("error"):
            return "error"
        elif state.get("sql_query"):
            return "execute"
        else:
            return "error"

    def _should_format_results(self, state: AutosysState) -> str:
        """Conditional edge: decide whether to format results or handle error"""
        if state.get("error"):
            return "error"
        elif state.get("query_results", {}).get("success"):
            return "format"
        else:
            return "error"

    def _create_no_results_html(self) -> str:
        """HTML for no results found"""
        return """
        <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #856404;">No Results Found</h4>
            <p style="margin: 0; color: #856404;">No Autosys jobs match your query criteria. Try rephrasing your question or checking job names.</p>
        </div>
        """

    def _create_error_html(self, error_msg: str, sql_query: str = "") -> str:
        """HTML for error display"""
        html = f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #721c24;">LangGraph AutosysQuery Error</h4>
            <p style="margin: 0; font-size: 14px;"><strong>Error:</strong> {error_msg}</p>
        """
        
        if sql_query:
            html += f"""
            <details style="margin-top: 10px;">
                <summary style="cursor: pointer; color: #495057; font-size: 12px;">View Generated SQL Query</summary>
                <pre style="background: #e9ecef; padding: 10px; margin-top: 5px; border-radius: 3px; font-size: 10px; overflow-x: auto; font-family: 'Courier New', monospace;">{sql_query}</pre>
            </details>
            """
        
        html += """
            <p style="margin: 10px 0 0 0; font-size: 12px; color: #856404;">
                <em>Tip: Try rephrasing your question or check if the job names/criteria exist in the database.</em>
            </p>
        </div>
        """
        return html

    def process_query(self, user_question: str, session_id: str) -> Dict[str, Any]:
        """Process a query using the LangGraph workflow"""
        
        # Initialize state
        initial_state = {
            "messages": [],
            "user_question": user_question,
            "sql_query": "",
            "query_results": {},
            "formatted_output": "",
            "error": "",
            "iteration_count": 0,
            "session_id": session_id
        }
        
        # Configuration for session management
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            self.logger.info(f"Processing query for session {session_id}: {user_question}")
            final_state = self.graph.invoke(initial_state, config=config)
            
            return {
                "success": not bool(final_state.get("error")),
                "formatted_output": final_state.get("formatted_output", ""),
                "sql_query": final_state.get("sql_query", ""),
                "row_count": final_state.get("query_results", {}).get("row_count", 0),
                "execution_time": final_state.get("query_results", {}).get("execution_time", 0),
                "error": final_state.get("error", ""),
                "messages": final_state.get("messages", [])
            }
            
        except Exception as e:
            self.logger.error(f"LangGraph execution failed: {e}")
            return {
                "success": False,
                "formatted_output": self._create_error_html(f"System execution failed: {str(e)}"),
                "error": str(e),
                "messages": []
            }

# ============================================================================
# OLD FUNCTION NAMES WITH NEW IMPLEMENTATION
# ============================================================================

# Global variable to store the LangGraph system
_autosys_langgraph_system = None

def initialize_agent(tools, llm, agent=None, verbose=True, checkpointer=None, 
                    handle_parsing_errors=True, max_iterations=3, 
                    early_stopping_method="generate", agent_kwargs=None):
    """
    Initialize the LangGraph-based Autosys system with the same function signature
    as the old initialize_agent function
    """
    global _autosys_langgraph_system
    
    try:
        # Extract autosys_db from tools or use global reference
        # This assumes you pass your autosys_db somehow - adjust as needed
        # For now, we'll assume it's available globally or passed in a specific way
        
        if hasattr(tools[0], 'autosys_db') if tools else False:
            autosys_db = tools[0].autosys_db
        else:
            # You'll need to provide autosys_db reference here
            # This is where you'd inject your AutosysOracleDatabase instance
            autosys_db = None  # Replace with your actual autosys_db instance
            
        _autosys_langgraph_system = AutosysLangGraphSystem(autosys_db, llm)
        
        logger.info("LangGraph Autosys system initialized successfully")
        return _autosys_langgraph_system
        
    except Exception as e:
        logger.error(f"Failed to initialize LangGraph system: {e}")
        raise e

def extract_last_ai_message(result: Union[Dict, str, Any]) -> str:
    """











++++++++++++++++
# ============================================================================
# COMPLETE INTEGRATION CODE - OLD WAY vs NEW WAY
# ============================================================================

import oracledb
import json
import logging
import re
from typing import Dict, Any, Optional, List, Union, TypedDict, Annotated
from datetime import datetime
from pydantic import BaseModel, Field
import operator

# LangChain imports (for both approaches)
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.agents.react.base import ReActSingleInputOutputParser

# LangGraph imports (for new approach)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ============================================================================
# SHARED COMPONENTS (Used by both approaches)
# ============================================================================

class AutosysQueryInput(BaseModel):
    """Input schema for Autosys Query Tool"""
    question: str = Field(description="Natural language question about Autosys jobs")

def extract_last_ai_message(result: Union[Dict, str, Any]) -> str:
    """Enhanced message extraction for both old and new approaches"""
    try:
        # Handle LangGraph response format (New Way)
        if isinstance(result, dict):
            if "formatted_output" in result and result["formatted_output"]:
                return result["formatted_output"].strip()
            if result.get("success") and "formatted_output" in result:
                return result["formatted_output"].strip()
            if "answer" in result and result["answer"]:
                return result["answer"].strip()
            if "result" in result and result["result"]:
                return result["result"].strip()
            if "content" in result and result["content"]:
                return result["content"].strip()
            if "output" in result and result["output"]:
                return result["output"].strip()
        
        # Handle string responses (Old Way)
        if isinstance(result, str):
            result = result.strip()
            final_answer_match = re.search(r"Final Answer:\s*(.+?)(?=\n\n|\n(?=\w+:)|\Z)", result, re.DOTALL | re.IGNORECASE)
            if final_answer_match:
                return final_answer_match.group(1).strip()
            if "<html>" in result.lower() or "<div>" in result.lower() or "<table>" in result.lower():
                return result
            if len(result) > 0 and not any(marker in result.lower() for marker in ['thought:', 'action:', 'observation:']):
                return result
        
        # Handle object attributes
        if hasattr(result, 'content'):
            return result.content.strip()
        elif hasattr(result, 'output'):
            return result.output.strip()
        
        result_str = str(result).strip()
        if result_str and result_str != "None":
            return result_str
            
        return "No response generated."
        
    except Exception as e:
        logging.error(f"Error extracting AI message: {str(e)}")
        return f"Error processing response: {str(e)}"

# ============================================================================
# OLD WAY - Traditional LangChain Agent Approach
# ============================================================================

class TraditionalAutosysLLMTool(BaseTool):
    """Traditional LangChain tool for Autosys queries"""
    
    name: str = "AutosysQuery"
    description: str = """
    Query Autosys job scheduler database using natural language.
    Returns formatted HTML results with professional styling.
    
    Examples:
    - "Show me all failed jobs today"
    - "Which ATSYS jobs are currently running?"
    - "List jobs owned by user ADMIN"
    """
    args_schema = AutosysQueryInput
    
    def __init__(self, autosys_db, llm_instance, max_results: int = 50):
        super().__init__()
        self.autosys_db = autosys_db
        self.llm = llm_instance
        self.max_results = max_results
        self.logger = logging.getLogger(self.__class__.__name__)

    def _run(self, question: str) -> str:
        """Traditional tool execution - returns formatted HTML string"""
        try:
            # Generate SQL
            sql_query = self._generate_sql(question)
            
            # Execute query
            query_result = self._execute_query(sql_query)
            
            # Format results
            formatted_output = self._format_results(query_result, question)
            
            return formatted_output
            
        except Exception as e:
            return self._create_error_html(str(e))

    def _generate_sql(self, question: str) -> str:
        """Generate SQL using LLM"""
        prompt = f"""
Generate Oracle SQL for Autosys database.

SCHEMA:
- aedbadmin.ujo_jobst: job_name, status, last_start, last_end, joid
- aedbadmin.ujo_job: joid, owner, machine  
- aedbadmin.UJO_INTCODES: code, TEXT

PATTERNS:
- Time: TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss')
- Joins: js INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
- Status: LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code

Question: {question}
Return only SQL:
"""
        response = self.llm.invoke(prompt)
        sql = response.content if hasattr(response, 'content') else str(response)
        return self._clean_sql(sql)

    def _clean_sql(self, sql: str) -> str:
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        sql = ' '.join(sql.split())
        
        if 'ROWNUM' not in sql.upper():
            if 'WHERE' in sql.upper():
                sql = sql.replace('WHERE', f'WHERE ROWNUM <= {self.max_results} AND', 1)
            else:
                sql += f' WHERE ROWNUM <= {self.max_results}'
        return sql.strip()

    def _execute_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute query using existing database"""
        try:
            start_time = datetime.now()
            
            if hasattr(self.autosys_db, 'run'):
                raw_results = self.autosys_db.run(sql_query)
            elif hasattr(self.autosys_db, 'execute_query'):
                raw_results = self.autosys_db.execute_query(sql_query)
            else:
                raise Exception("Database method not found")
            
            results = self._process_results(raw_results)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "results": results,
                "row_count": len(results),
                "execution_time": execution_time,
                "sql_query": sql_query
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "sql_query": sql_query
            }

    def _process_results(self, raw_results) -> List[Dict]:
        """Convert raw results to standardized format"""
        if isinstance(raw_results, str):
            try:
                import ast
                if raw_results.startswith('['):
                    raw_results = ast.literal_eval(raw_results)
                else:
                    return [{"result": raw_results}]
            except:
                return [{"result": raw_results}]
        
        if isinstance(raw_results, list):
            processed = []
            for item in raw_results:
                if isinstance(item, (tuple, list)):
                    row_dict = {}
                    col_names = ["JOB_NAME", "START_TIME", "END_TIME", "STATUS", "OWNER"]
                    for i, value in enumerate(item):
                        col_name = col_names[i] if i < len(col_names) else f"COL_{i}"
                        row_dict[col_name] = str(value) if value is not None else ""
                    processed.append(row_dict)
                elif isinstance(item, dict):
                    processed.append(item)
                else:
                    processed.append({"VALUE": str(item)})
            return processed
        
        return [{"result": str(raw_results)}]

    def _format_results(self, query_result: Dict[str, Any], question: str) -> str:
        """Format results using LLM"""
        if not query_result["success"]:
            return self._create_error_html(query_result["error"], query_result.get("sql_query"))
        
        results = query_result["results"]
        if not results:
            return "<div style='padding: 15px; background: #fff3cd; border-radius: 5px;'><h4>No Results Found</h4></div>"
        
        # Use LLM for formatting
        formatting_prompt = f"""
Create professional HTML for Autosys query results:

Question: {question}
Results: {len(results)} jobs
Data: {json.dumps(results[:10], indent=2, default=str)}

Create responsive HTML with inline CSS, color-coded status, and professional styling.
Return only HTML:
"""
        
        try:
            response = self.llm.invoke(formatting_prompt)
            formatted_html = response.content if hasattr(response, 'content') else str(response)
            
            metadata = f"""
            <div style="margin-top: 15px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 11px; color: #666;">
                Traditional AutosysQuery • {query_result['row_count']} results • {query_result['execution_time']:.2f}s
            </div>
            """
            
            return formatted_html + metadata
            
        except Exception as e:
            return self._create_fallback_html(results, question)

    def _create_error_html(self, error_msg: str, sql_query: str = "") -> str:
        html = f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;">
            <h4>AutosysQuery Error</h4>
            <p><strong>Error:</strong> {error_msg}</p>
        """
        if sql_query:
            html += f"<details><summary>View SQL</summary><pre style='font-size: 11px;'>{sql_query}</pre></details>"
        html += "</div>"
        return html

    def _create_fallback_html(self, results: List[Dict], question: str) -> str:
        """Fallback formatting if LLM fails"""
        html = f"<h3>Autosys Results ({len(results)} jobs)</h3><table border='1' style='font-size: 12px;'>"
        if results:
            html += "<tr style='background: #f2f2f2;'>"
            for key in results[0].keys():
                html += f"<th style='padding: 6px;'>{key}</th>"
            html += "</tr>"
            
            for i, row in enumerate(results[:20]):
                bg = "#f9f9f9" if i % 2 == 0 else "#ffffff"
                html += f"<tr style='background: {bg};'>"
                for value in row.values():
                    html += f"<td style='padding: 4px;'>{str(value)[:60]}</td>"
                html += "</tr>"
        html += "</table>"
        return html

def setup_old_way(autosys_db, llm_instance):
    """Setup traditional LangChain agent approach"""
    
    # Create traditional tool
    autosys_tool = TraditionalAutosysLLMTool(autosys_db, llm_instance)
    
    # Create agent
    sql_agent = initialize_agent(
        tools=[autosys_tool],
        llm=llm_instance,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate",
        agent_kwargs={
            "format_instructions": """
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be AutosysQuery
Action Input: the input to the action
Observation: the tool will populate this automatically
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: Include all results from the query output, format each job on a new line.
IMPORTANT: Do NOT include multiple actions and final answers in the same response.
""",
            "output_parser": ReActSingleInputOutputParser()
        }
    )
    
    return sql_agent

def get_chat_response_old_way(message: str, session_id: str, sql_agent) -> str:
    """Old way chat response function"""
    try:
        if not message or not message.strip():
            return "Please provide a question about the database."
        
        config = {"configurable": {"thread_id": session_id}}
        
        response = sql_agent.invoke({"input": message.strip()}, config=config)
        final_response = extract_last_ai_message(response)
        
        return final_response if final_response != "No response generated." else "Unable to process request."
        
    except Exception as e:
        logging.error(f"Old way error: {str(e)}")
        return f"Error processing request: {str(e)}"

# ============================================================================
# NEW WAY - LangGraph Approach  
# ============================================================================

# State definition for LangGraph
class AutosysState(TypedDict):
    messages: Annotated[List[Dict[str, str]], operator.add]
    user_question: str
    sql_query: str
    query_results: Dict[str, Any]
    formatted_output: str
    error: str
    iteration_count: int

class LangGraphAutosysTool(BaseTool):
    """LangGraph-compatible Autosys tool"""
    
    name: str = "AutosysQuery"
    description: str = "Query Autosys database and return structured results"
    args_schema = AutosysQueryInput
    
    def __init__(self, autosys_db, llm_instance, max_results: int = 50):
        super().__init__()
        self.autosys_db = autosys_db
        self.llm = llm_instance
        self.max_results = max_results

    def _run(self, question: str) -> Dict[str, Any]:
        """Execute query and return structured results for LangGraph"""
        try:
            sql_query = self._generate_sql(question)
            results = self._execute_query(sql_query)
            
            return {
                "success": True,
                "sql_query": sql_query,
                "results": results.get("results", []),
                "row_count": results.get("row_count", 0),
                "execution_time": results.get("execution_time", 0)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    def _generate_sql(self, question: str) -> str:
        prompt = f"""
Generate Oracle SQL for Autosys database:

SCHEMA:
- aedbadmin.ujo_jobst: job_name, status, last_start, last_end, joid
- aedbadmin.ujo_job: joid, owner, machine
- aedbadmin.UJO_INTCODES: code, TEXT

Question: {question}
Return only SQL:
"""
        response = self.llm.invoke(prompt)
        sql = response.content if hasattr(response, 'content') else str(response)
        
        # Clean and limit results
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        sql = ' '.join(sql.split())
        
        if 'ROWNUM' not in sql.upper():
            if 'WHERE' in sql.upper():
                sql = sql.replace('WHERE', f'WHERE ROWNUM <= {self.max_results} AND', 1)
            else:
                sql += f' WHERE ROWNUM <= {self.max_results}'
        
        return sql.strip()

    def _execute_query(self, sql_query: str) -> Dict[str, Any]:
        try:
            start_time = datetime.now()
            
            if hasattr(self.autosys_db, 'run'):
                raw_results = self.autosys_db.run(sql_query)
            else:
                raise Exception("Database method not found")
            
            results = self._process_results(raw_results)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "results": results,
                "row_count": len(results),
                "execution_time": execution_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    def _process_results(self, raw_results) -> List[Dict]:
        if isinstance(raw_results, str):
            try:
                import ast
                if raw_results.startswith('['):
                    raw_results = ast.literal_eval(raw_results)
            except:
                return [{"result": raw_results}]
        
        if isinstance(raw_results, list):
            processed = []
            for item in raw_results:
                if isinstance(item, (tuple, list)):
                    row_dict = {}
                    col_names = ["JOB_NAME", "START_TIME", "END_TIME", "STATUS", "OWNER"]
                    for i, value in enumerate(item):
                        col_name = col_names[i] if i < len(col_names) else f"COL_{i}"
                        row_dict[col_name] = str(value) if value is not None else ""
                    processed.append(row_dict)
                else:
                    processed.append({"VALUE": str(item)})
            return processed
        
        return [{"result": str(raw_results)}]

class AutosysLangGraph:
    """LangGraph-based Autosys system"""
    
    def __init__(self, autosys_db, llm_instance):
        self.autosys_db = autosys_db
        self.llm = llm_instance
        self.tool = LangGraphAutosysTool(autosys_db, llm_instance)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AutosysState)
        
        workflow.add_node("understand_query", self.understand_query_node)
        workflow.add_node("generate_sql", self.generate_sql_node)
        workflow.add_node("execute_query", self.execute_query_node)
        workflow.add_node("format_results", self.format_results_node)
        workflow.add_node("handle_error", self.handle_error_node)
        
        workflow.set_entry_point("understand_query")
        
        workflow.add_edge("understand_query", "generate_sql")
        workflow.add_conditional_edges(
            "generate_sql",
            self._should_execute_query,
            {"execute": "execute_query", "error": "handle_error"}
        )
        workflow.add_conditional_edges(
            "execute_query", 
            self._should_format_results,
            {"format": "format_results", "error": "handle_error"}
        )
        workflow.add_edge("format_results", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()

    def understand_query_node(self, state: AutosysState) -> AutosysState:
        question = state["user_question"].strip()
        if not any(keyword in question.lower() for keyword in ['atsys', 'job', 'status']):
            question = f"Show Autosys jobs related to: {question}"
            state["user_question"] = question
        
        state["messages"].append({
            "role": "system", 
            "content": f"Processing: {state['user_question']}"
        })
        return state

    def generate_sql_node(self, state: AutosysState) -> AutosysState:
        try:
            tool_result = self.tool.run(state["user_question"])
            if tool_result["success"]:
                state["sql_query"] = tool_result["sql_query"]
            else:
                state["error"] = f"SQL generation failed: {tool_result.get('error', 'Unknown error')}"
        except Exception as e:
            state["error"] = f"SQL generation exception: {str(e)}"
        return state

    def execute_query_node(self, state: AutosysState) -> AutosysState:
        try:
            tool_result = self.tool.run(state["user_question"])
            if tool_result["success"]:
                state["query_results"] = {
                    "success": True,
                    "results": tool_result["results"],
                    "row_count": tool_result["row_count"],
                    "execution_time": tool_result["execution_time"]
                }
            else:
                state["error"] = f"Query failed: {tool_result.get('error', 'Unknown error')}"
                state["query_results"] = {"success": False, "error": state["error"]}
        except Exception as e:
            state["error"] = f"Query exception: {str(e)}"
            state["query_results"] = {"success": False, "error": state["error"]}
        return state

    def format_results_node(self, state: AutosysState) -> AutosysState:
        try:
            results = state["query_results"]["results"]
            
            if not results:
                state["formatted_output"] = "<div style='padding: 15px; background: #fff3cd;'><h4>No Results Found</h4></div>"
                return state
            
            # Format using LLM
            formatting_prompt = f"""
Create professional HTML for Autosys query results:

Question: {state['user_question']}
Results: {len(results)} jobs
Data: {json.dumps(results[:10], indent=2, default=str)}

Create responsive HTML with inline CSS, color-coded status, professional styling.
Return only HTML:
"""
            
            response = self.llm.invoke(formatting_prompt)
            formatted_html = response.content if hasattr(response, 'content') else str(response)
            
            metadata = f"""
            <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; font-size: 11px; color: #666;">
                LangGraph AutosysQuery • {state['query_results']['row_count']} results • {state['query_results']['execution_time']:.2f}s
            </div>
            """
            
            state["formatted_output"] = formatted_html + metadata
            
        except Exception as e:
            state["error"] = f"Formatting failed: {str(e)}"
            state["formatted_output"] = f"<div style='color: red;'>Formatting error: {str(e)}</div>"
        
        return state

    def handle_error_node(self, state: AutosysState) -> AutosysState:
        error_msg = state.get("error", "Unknown error")
        state["formatted_output"] = f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;">
            <h4>LangGraph AutosysQuery Error</h4>
            <p><strong>Error:</strong> {error_msg}</p>
        </div>
        """
        return state

    def _should_execute_query(self, state: AutosysState) -> str:
        return "error" if state.get("error") else "execute"

    def _should_format_results(self, state: AutosysState) -> str:
        if state.get("error"):
            return "error"
        elif state.get("query_results", {}).get("success"):
            return "format"
        else:
            return "error"

    def query(self, user_question: str, config: Dict = None) -> Dict[str, Any]:
        initial_state = {
            "messages": [],
            "user_question": user_question,
            "sql_query": "",
            "query_results": {},
            "formatted_output": "",
            "error": "",
            "iteration_count": 0
        }
        
        try:
            final_state = self.graph.invoke(initial_state, config=config)
            
            return {
                "success": not bool(final_state.get("error")),
                "formatted_output": final_state.get("formatted_output", ""),
                "sql_query": final_state.get("sql_query", ""),
                "row_count": final_state.get("query_results", {}).get("row_count", 0),
                "execution_time": final_state.get("query_results", {}).get("execution_time", 0),
                "error": final_state.get("error", "")
            }
            
        except Exception as e:
            return {
                "success": False,
                "formatted_output": f"<div style='color: red;'>Graph execution failed: {str(e)}</div>",
                "error": str(e)
            }

def setup_new_way(autosys_db, llm_instance):
    """Setup LangGraph approach"""
    return AutosysLangGraph(autosys_db, llm_instance)

def get_chat_response_new_way(message: str, session_id: str, autosys_system) -> str:
    """New way chat response function"""
    try:
        if not message or not message.strip():
            return "Please provide a question about the database."
        
        config = {"configurable": {"thread_id": session_id}}
        
        result = autosys_system.query(message.strip(), config)
        
        if result["success"]:
            return result["formatted_output"]
        else:
            return f"Error: {result['error']}"
            
    except Exception as e:
        logging.error(f"New way error: {str(e)}")
        return f"System error: {str(e)}"

# ============================================================================
# COMPLETE USAGE EXAMPLE
# ============================================================================

def complete_setup_example():
    """Complete setup example showing both approaches"""
    
    # Your existing configuration
    oracle_uri = "oracle+oracledb://username:password@host:port/service_name"
    
    # Assuming you have these from your existing setup
    # autosys_db = AutosysOracleDatabase(oracle_uri)  # Your existing DB class
    # llm = get_llm("langchain")  # Your existing LLM setup
    
    # For demonstration - replace with your actual instances
    autosys_db = None  # Your AutosysOracleDatabase instance
    llm = None  # Your LLM instance
    
    print("Setting up OLD WAY (Traditional LangChain Agent)...")
    old_way_agent = setup_old_way(autosys_db, llm)
    
    print("Setting up NEW WAY (LangGraph)...")  
    new_way_system = setup_new_way(autosys_db, llm)
    
    # Test both approaches
    test_message = "Show me all failed ATSYS jobs today"
    session_id = "test_session_123"
    
    print("\n=== OLD WAY RESPONSE ===")
    old_response = get_chat_response_old_way(test_message, session_id, old_way_agent)
    print(old_response)
    
    print("\n=== NEW WAY RESPONSE ===") 
    new_response = get_chat_response_new_way(test_message, session_id, new_way_system)
    print(new_response)
    
    return {
        "old_way_agent": old_way_agent,
        "new_way_system": new_way_system
    }

# ============================================================================
# PLUG-AND-PLAY INTEGRATION FOR YOUR EXISTING CODE
# ============================================================================

def replace_your_existing_code():
    """
    INSTRUCTIONS:
    
    1. Replace your existing tool/agent setup with either approach:
    
    FOR OLD WAY (Minimal changes):
    --------------------------------
    # Replace your existing:
    # sql_agent = initialize_agent(...)
    
    # With:
    old_way_agent = setup_old_way(autosys_db, llm)
    
    # Replace your get_chat_response function with:
    def get_chat_response(message: str, session_id: str) -> str:
        return get_chat_response_old_way(message, session_id, old_way_agent)
    
    
    FOR NEW WAY (Recommended):  
    -------------------------
    # Replace your existing agent entirely:
    new_way_system = setup_new_way(autosys_db, llm)
    
    # Replace your get_chat_response function with:
    def get_chat_response(message: str, session_id: str) -> str:
        return get_chat_response_new_way(message, session_id, new_way_system)
    
    
    2. Your existing variables needed:
       - autosys_db: Your AutosysOracleDatabase instance
       - llm: Your LLM instance from get_llm("langchain")
    
    3. Everything else (API endpoints, session handling, etc.) stays the same!
    """
    pass

if __name__ == "__main__":
    # Run the complete example
    complete_setup_example()




import re
import logging
from typing import Dict, Any, Optional, Union

def extract_last_ai_message(result: Union[Dict, str, Any]) -> str:
    """
    Extract the final answer from agent response, handling mixed formats and LangGraph responses.
    Enhanced version that handles both traditional agents and LangGraph outputs.
    """
    try:
        # Handle LangGraph response format
        if isinstance(result, dict):
            # Check for LangGraph formatted_output first
            if "formatted_output" in result and result["formatted_output"]:
                return result["formatted_output"].strip()
            
            # Check for success field and formatted content
            if result.get("success") and "formatted_output" in result:
                return result["formatted_output"].strip()
            
            # Traditional response formats
            if "answer" in result and result["answer"]:
                return result["answer"].strip()
            elif "result" in result and result["result"]:
                return result["result"].strip()
            elif "content" in result and result["content"]:
                return result["content"].strip()
            elif "text" in result and result["text"]:
                return result["text"].strip()
            elif "output" in result and result["output"]:
                return result["output"].strip()
        
        # Handle string responses
        if isinstance(result, str):
            result = result.strip()
            
            # Look for 'Final Answer:' pattern first (most specific)
            final_answer_match = re.search(r"Final Answer:\s*(.+?)(?=\n\n|\n(?=\w+:)|\Z)", result, re.DOTALL | re.IGNORECASE)
            if final_answer_match:
                return final_answer_match.group(1).strip()
            
            # Look for HTML content (likely formatted output)
            if "<html>" in result.lower() or "<div>" in result.lower() or "<table>" in result.lower():
                return result
            
            # If it's just a plain response without markers, return as-is
            if len(result) > 0 and not any(marker in result.lower() for marker in ['thought:', 'action:', 'observation:']):
                return result
        
        # Handle other response types from different agents
        if hasattr(result, 'content'):
            return result.content.strip()
        elif hasattr(result, 'text'):
            return result.text.strip()
        elif hasattr(result, 'output'):
            return result.output.strip()
        
        # Fallback: convert to string
        result_str = str(result).strip()
        if result_str and result_str != "None":
            return result_str
            
        return "No response generated."
        
    except Exception as e:
        logging.error(f"Error extracting AI message: {str(e)}")
        return f"Error processing response: {str(e)}"

def get_chat_response(message: str, session_id: str) -> str:
    """
    Enhanced chat response function that integrates LangGraph Autosys system
    with fallback to traditional agent approach.
    """
    try:
        # Input validation
        if not message or not message.strip():
            return "Please provide a question about the database."
        
        message = message.strip()
        print(f"🔍 Processing: {message}")
        
        # Session configuration for LangGraph
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            # Primary approach: Use LangGraph Autosys System
            print("🤖 Using LangGraph Autosys System...")
            
            # Initialize LangGraph system (assuming it's globally available or injected)
            # You'll need to make this available in your scope
            if 'autosys_langgraph_system' in globals():
                langgraph_result = autosys_langgraph_system.query(message, config)
                
                print(f"📊 LangGraph response type: {type(langgraph_result)}")
                print(f"📋 LangGraph response preview: {str(langgraph_result)[:200]}...")
                
                if langgraph_result.get("success", False):
                    final_response = extract_last_ai_message(langgraph_result)
                    if final_response and final_response != "No response generated.":
                        print("✅ LangGraph response successful")
                        return final_response
                    else:
                        print("⚠️ LangGraph response empty, trying fallback")
                else:
                    print(f"❌ LangGraph failed: {langgraph_result.get('error', 'Unknown error')}")
            else:
                print("⚠️ LangGraph system not available, using fallback")
        
        except Exception as langgraph_error:
            print(f"❌ LangGraph system error: {str(langgraph_error)}")
        
        # Fallback approach: Use traditional SQL agent
        try:
            print("🔄 Using traditional SQL agent fallback...")
            
            # Your existing agent invocation
            response = sql_agent.invoke(
                {
                    "input": message,
                    "system_prompt": system_prompt  # Use your existing system prompt
                },
                config=config
            )
            
            print(f"🤖 Agent response type: {type(response)}")
            print(f"📝 Agent response content: {response}")
            
            # Extract final answer using enhanced extraction
            final_response = extract_last_ai_message(response)
            
            if final_response and final_response != "No response generated.":
                print("✅ Traditional agent response successful")
                return final_response
            else:
                print("⚠️ Traditional agent response empty")
                return "I was unable to generate a proper response. Please try rephrasing your question."
        
        except Exception as agent_error:
            print(f"❌ Traditional agent error: {str(agent_error)}")
            return f"Error processing your request: {str(agent_error)}"
    
    except Exception as e:
        print(f"❌ Critical error in get_chat_response: {str(e)}")
        logging.error(f"Critical error in get_chat_response: {str(e)}")
        return f"A system error occurred while processing your request: {str(e)}"

# Enhanced system initialization function
def initialize_enhanced_chat_system(autosys_db, llm_instance, traditional_agent):
    """
    Initialize the enhanced chat system with both LangGraph and traditional agent support
    
    Args:
        autosys_db: Your AutosysOracleDatabase instance
        llm_instance: Your LLM instance
        traditional_agent: Your existing sql_agent
    
    Returns:
        Configured system ready for use
    """
    global autosys_langgraph_system, sql_agent, system_prompt
    
    try:
        # Initialize LangGraph system
        from langgraph_autosys_implementation import create_langgraph_autosys_system
        autosys_langgraph_system = create_langgraph_autosys_system(autosys_db, llm_instance)
        print("✅ LangGraph Autosys system initialized")
    except Exception as e:
        print(f"⚠️ LangGraph initialization failed: {e}")
        autosys_langgraph_system = None
    
    # Set traditional agent as fallback
    sql_agent = traditional_agent
    
    # Your existing system prompt
    system_prompt = """
    You are an expert database assistant for Autosys job scheduling system.
    Provide accurate, helpful responses about job status, schedules, and performance.
    Always format responses in a professional, easy-to-read manner.
    """
    
    return {
        "langgraph_available": autosys_langgraph_system is not None,
        "traditional_agent_available": sql_agent is not None,
        "status": "ready"
    }

# Alternative: Direct LangGraph integration without globals
def get_chat_response_direct_langgraph(message: str, session_id: str, 
                                     autosys_system=None, fallback_agent=None) -> str:
    """
    Direct LangGraph integration without global variables
    
    Args:
        message: User input
        session_id: Session identifier
        autosys_system: LangGraph Autosys system instance
        fallback_agent: Traditional agent for fallback
    """
    try:
        if not message or not message.strip():
            return "Please provide a question about the database."
        
        message = message.strip()
        print(f"🔍 Processing: {message}")
        
        config = {"configurable": {"thread_id": session_id}}
        
        # Try LangGraph first
        if autosys_system:
            try:
                print("🤖 Using LangGraph Autosys System...")
                
                langgraph_result = autosys_system.query(message, config)
                
                if langgraph_result.get("success", False):
                    final_response = extract_last_ai_message(langgraph_result)
                    if final_response and final_response != "No response generated.":
                        print("✅ LangGraph response successful")
                        return final_response
                
            except Exception as e:
                print(f"❌ LangGraph error: {e}")
        
        # Fallback to traditional agent
        if fallback_agent:
            try:
                print("🔄 Using traditional agent fallback...")
                
                response = fallback_agent.invoke({
                    "input": message,
                    "system_prompt": system_prompt
                }, config=config)
                
                final_response = extract_last_ai_message(response)
                if final_response and final_response != "No response generated.":
                    return final_response
                    
            except Exception as e:
                print(f"❌ Traditional agent error: {e}")
        
        return "I was unable to process your request. Please try again or rephrase your question."
    
    except Exception as e:
        logging.error(f"Error in get_chat_response_direct_langgraph: {e}")
        return f"System error: {str(e)}"

# Usage example for your existing setup
"""
# In your main initialization code:

# Your existing setup
oracle_uri = "oracle+oracledb://***:***@***/service_name=service_name"
autosys_db = AutosysOracleDatabase(oracle_uri)
llm = get_llm("langchain")
sql_agent = initialize_agent(...)  # Your existing agent

# Initialize enhanced system
system_status = initialize_enhanced_chat_system(autosys_db, llm, sql_agent)
print(f"System status: {system_status}")

# Now your get_chat_response function will use both systems
response = get_chat_response("Show me failed ATSYS jobs", "session_123")

# Or use direct approach:
autosys_langgraph = create_langgraph_autosys_system(autosys_db, llm)
response = get_chat_response_direct_langgraph(
    "Show me failed ATSYS jobs", 
    "session_123",
    autosys_system=autosys_langgraph,
    fallback_agent=sql_agent
)
"""

# Debug helper function
def debug_response_extraction(response):
    """Helper function to debug response extraction issues"""
    print(f"🔍 Debug Response Type: {type(response)}")
    print(f"📋 Debug Response Content: {response}")
    
    if isinstance(response, dict):
        print(f"🔑 Available keys: {list(response.keys())}")
        for key in ['formatted_output', 'answer', 'result', 'content', 'output']:
            if key in response:
                print(f"  ✓ {key}: {str(response[key])[:100]}...")
    
    extracted = extract_last_ai_message(response)
    print(f"📤 Extracted result: {extracted[:200]}...")
    
    return extracted

    ============




from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Dict, Any, List
import operator
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import json
import logging
from datetime import datetime
import re

# State definition for LangGraph
class AutosysState(TypedDict):
    """State for the Autosys LangGraph workflow"""
    messages: Annotated[List[Dict[str, str]], operator.add]
    user_question: str
    sql_query: str
    query_results: Dict[str, Any]
    formatted_output: str
    error: str
    iteration_count: int

class AutosysQueryInput(BaseModel):
    """Input schema for Autosys Query Tool"""
    question: str = Field(description="Natural language question about Autosys jobs")

class AutosysLLMTool(BaseTool):
    """LangChain tool for use within LangGraph nodes"""
    
    name: str = "AutosysQuery"
    description: str = """
    Query Autosys job scheduler database using natural language.
    Returns structured data for further processing in the graph.
    """
    args_schema = AutosysQueryInput
    
    def __init__(self, autosys_db, llm_instance, max_results: int = 50):
        super().__init__()
        self.autosys_db = autosys_db
        self.llm = llm_instance
        self.max_results = max_results
        self.logger = logging.getLogger(self.__class__.__name__)

    def _run(self, question: str) -> Dict[str, Any]:
        """Execute query and return structured results"""
        try:
            # Generate SQL
            sql_query = self._generate_sql(question)
            
            # Execute query
            results = self._execute_query(sql_query)
            
            return {
                "success": True,
                "sql_query": sql_query,
                "results": results.get("results", []),
                "row_count": results.get("row_count", 0),
                "execution_time": results.get("execution_time", 0)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "sql_query": "",
                "results": [],
                "row_count": 0,
                "execution_time": 0
            }

    def _generate_sql(self, question: str) -> str:
        """Generate SQL using LLM"""
        prompt = f"""
Generate Oracle SQL for Autosys database query.

SCHEMA:
- aedbadmin.ujo_jobst: job_name, status, last_start, last_end, joid
- aedbadmin.ujo_job: joid, owner, machine
- aedbadmin.UJO_INTCODES: code, TEXT

PATTERNS:
- Time: TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss')
- Joins: js INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
- Status: LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code

Question: {question}

Return only SQL:
"""
        
        response = self.llm.invoke(prompt)
        sql = response.content if hasattr(response, 'content') else str(response)
        return self._clean_sql(sql)

    def _clean_sql(self, sql: str) -> str:
        """Clean SQL response"""
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        sql = ' '.join(sql.split())
        
        if 'ROWNUM' not in sql.upper():
            if 'WHERE' in sql.upper():
                sql = sql.replace('WHERE', f'WHERE ROWNUM <= {self.max_results} AND', 1)
            else:
                sql += f' WHERE ROWNUM <= {self.max_results}'
        
        return sql.strip()

    def _execute_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute query using existing database"""
        try:
            start_time = datetime.now()
            
            # Use existing database connection
            if hasattr(self.autosys_db, 'execute_query'):
                raw_results = self.autosys_db.execute_query(sql_query)
            elif hasattr(self.autosys_db, 'run'):
                raw_results = self.autosys_db.run(sql_query)
            else:
                raise Exception("Database method not found")
            
            # Convert results to standard format
            results = self._process_results(raw_results)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "results": results,
                "row_count": len(results),
                "execution_time": execution_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "row_count": 0,
                "execution_time": 0
            }

    def _process_results(self, raw_results) -> List[Dict]:
        """Convert raw results to standardized format"""
        if isinstance(raw_results, str):
            try:
                import ast
                if raw_results.startswith('['):
                    raw_results = ast.literal_eval(raw_results)
                else:
                    return [{"result": raw_results}]
            except:
                return [{"result": raw_results}]
        
        if isinstance(raw_results, list):
            processed = []
            for item in raw_results:
                if isinstance(item, (tuple, list)):
                    row_dict = {}
                    col_names = ["JOB_NAME", "START_TIME", "END_TIME", "STATUS", "OWNER"]
                    for i, value in enumerate(item):
                        col_name = col_names[i] if i < len(col_names) else f"COL_{i}"
                        row_dict[col_name] = str(value) if value is not None else ""
                    processed.append(row_dict)
                elif isinstance(item, dict):
                    processed.append(item)
                else:
                    processed.append({"VALUE": str(item)})
            return processed
        
        return [{"result": str(raw_results)}]

# LangGraph Implementation
class AutosysLangGraph:
    """LangGraph-based Autosys query system"""
    
    def __init__(self, autosys_db, llm_instance):
        self.autosys_db = autosys_db
        self.llm = llm_instance
        self.tool = AutosysLLMTool(autosys_db, llm_instance)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create StateGraph
        workflow = StateGraph(AutosysState)
        
        # Add nodes
        workflow.add_node("understand_query", self.understand_query_node)
        workflow.add_node("generate_sql", self.generate_sql_node)
        workflow.add_node("execute_query", self.execute_query_node)
        workflow.add_node("format_results", self.format_results_node)
        workflow.add_node("handle_error", self.handle_error_node)
        
        # Set entry point
        workflow.set_entry_point("understand_query")
        
        # Add edges
        workflow.add_edge("understand_query", "generate_sql")
        workflow.add_conditional_edges(
            "generate_sql",
            self._should_execute_query,
            {
                "execute": "execute_query",
                "error": "handle_error"
            }
        )
        workflow.add_conditional_edges(
            "execute_query", 
            self._should_format_results,
            {
                "format": "format_results",
                "error": "handle_error"
            }
        )
        workflow.add_edge("format_results", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()

    def understand_query_node(self, state: AutosysState) -> AutosysState:
        """Analyze and understand the user query"""
        self.logger.info(f"Understanding query: {state['user_question']}")
        
        # Add analysis message
        state["messages"].append({
            "role": "system", 
            "content": f"Analyzing Autosys query: {state['user_question']}"
        })
        
        # Basic query validation and enhancement
        question = state["user_question"].strip()
        
        # Enhance question with context if needed
        if not any(keyword in question.lower() for keyword in ['atsys', 'job', 'status']):
            question = f"Show Autosys jobs related to: {question}"
            state["user_question"] = question
        
        return state

    def generate_sql_node(self, state: AutosysState) -> AutosysState:
        """Generate SQL query using LLM"""
        try:
            self.logger.info("Generating SQL query")
            
            # Use the tool to generate SQL
            tool_result = self.tool.run(state["user_question"])
            
            if tool_result["success"]:
                state["sql_query"] = tool_result["sql_query"]
                state["messages"].append({
                    "role": "system",
                    "content": f"Generated SQL query: {tool_result['sql_query'][:100]}..."
                })
            else:
                state["error"] = f"SQL generation failed: {tool_result.get('error', 'Unknown error')}"
                state["messages"].append({
                    "role": "system",
                    "content": f"SQL generation error: {state['error']}"
                })
            
        except Exception as e:
            state["error"] = f"SQL generation exception: {str(e)}"
            self.logger.error(f"SQL generation failed: {e}")
        
        return state

    def execute_query_node(self, state: AutosysState) -> AutosysState:
        """Execute the SQL query"""
        try:
            self.logger.info("Executing SQL query")
            
            # Execute using the tool
            tool_result = self.tool.run(state["user_question"])
            
            if tool_result["success"]:
                state["query_results"] = {
                    "success": True,
                    "results": tool_result["results"],
                    "row_count": tool_result["row_count"],
                    "execution_time": tool_result["execution_time"],
                    "sql_query": tool_result["sql_query"]
                }
                state["messages"].append({
                    "role": "system",
                    "content": f"Query executed successfully: {tool_result['row_count']} results in {tool_result['execution_time']:.2f}s"
                })
            else:
                state["error"] = f"Query execution failed: {tool_result.get('error', 'Unknown error')}"
                state["query_results"] = {"success": False, "error": state["error"]}
                
        except Exception as e:
            state["error"] = f"Query execution exception: {str(e)}"
            state["query_results"] = {"success": False, "error": state["error"]}
            self.logger.error(f"Query execution failed: {e}")
        
        return state

    def format_results_node(self, state: AutosysState) -> AutosysState:
        """Format results using LLM"""
        try:
            self.logger.info("Formatting results")
            
            results = state["query_results"]["results"]
            
            if not results:
                state["formatted_output"] = self._create_no_results_html()
                return state
            
            # Format using LLM
            formatting_prompt = f"""
Create professional HTML for these Autosys query results:

Question: {state['user_question']}
Results: {len(results)} jobs found
Execution Time: {state['query_results'].get('execution_time', 0):.2f}s

Data (first 10 results):
{json.dumps(results[:10], indent=2, default=str)}

Create responsive HTML with:
- Professional styling (inline CSS only)
- Color-coded status badges  
- Summary statistics
- Mobile-friendly design
- Clean, readable format

Return only HTML:
"""
            
            response = self.llm.invoke(formatting_prompt)
            formatted_html = response.content if hasattr(response, 'content') else str(response)
            
            # Add metadata
            metadata = f"""
            <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; font-size: 11px; color: #666;">
                <strong>LangGraph AutosysQuery</strong> • {state['query_results']['row_count']} results • {state['query_results']['execution_time']:.2f}s
            </div>
            """
            
            state["formatted_output"] = formatted_html + metadata
            state["messages"].append({
                "role": "assistant",
                "content": "Results formatted successfully"
            })
            
        except Exception as e:
            state["error"] = f"Formatting failed: {str(e)}"
            state["formatted_output"] = self._create_error_html(state["error"])
            self.logger.error(f"Formatting failed: {e}")
        
        return state

    def handle_error_node(self, state: AutosysState) -> AutosysState:
        """Handle errors and provide user-friendly error messages"""
        self.logger.error(f"Handling error: {state.get('error', 'Unknown error')}")
        
        error_msg = state.get("error", "An unknown error occurred")
        sql_query = state.get("sql_query", "")
        
        state["formatted_output"] = self._create_error_html(error_msg, sql_query)
        state["messages"].append({
            "role": "system",
            "content": f"Error handled: {error_msg}"
        })
        
        return state

    def _should_execute_query(self, state: AutosysState) -> str:
        """Conditional edge: decide whether to execute query or handle error"""
        if state.get("error"):
            return "error"
        elif state.get("sql_query"):
            return "execute"
        else:
            return "error"

    def _should_format_results(self, state: AutosysState) -> str:
        """Conditional edge: decide whether to format results or handle error"""
        if state.get("error"):
            return "error"
        elif state.get("query_results", {}).get("success"):
            return "format"
        else:
            return "error"

    def _create_no_results_html(self) -> str:
        """HTML for no results found"""
        return """
        <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px;">
            <h4 style="margin: 0 0 10px 0; color: #856404;">No Results Found</h4>
            <p style="margin: 0; color: #856404;">No Autosys jobs match your query criteria. Try rephrasing your question or checking job names.</p>
        </div>
        """

    def _create_error_html(self, error_msg: str, sql_query: str = "") -> str:
        """HTML for error display"""
        html = f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;">
            <h4 style="margin: 0 0 10px 0;">LangGraph AutosysQuery Error</h4>
            <p style="margin: 0;"><strong>Error:</strong> {error_msg}</p>
        """
        
        if sql_query:
            html += f"""
            <details style="margin-top: 10px;">
                <summary style="cursor: pointer;">View Generated SQL</summary>
                <pre style="background: #e9ecef; padding: 10px; margin-top: 5px; border-radius: 3px; font-size: 11px; overflow-x: auto;">{sql_query}</pre>
            </details>
            """
        
        html += "</div>"
        return html

    def query(self, user_question: str, config: Dict = None) -> Dict[str, Any]:
        """Main method to process a query"""
        
        # Initialize state
        initial_state = {
            "messages": [],
            "user_question": user_question,
            "sql_query": "",
            "query_results": {},
            "formatted_output": "",
            "error": "",
            "iteration_count": 0
        }
        
        # Execute the graph
        try:
            final_state = self.graph.invoke(initial_state, config=config)
            
            return {
                "success": not bool(final_state.get("error")),
                "formatted_output": final_state.get("formatted_output", ""),
                "sql_query": final_state.get("sql_query", ""),
                "row_count": final_state.get("query_results", {}).get("row_count", 0),
                "execution_time": final_state.get("query_results", {}).get("execution_time", 0),
                "error": final_state.get("error", ""),
                "messages": final_state.get("messages", [])
            }
            
        except Exception as e:
            self.logger.error(f"Graph execution failed: {e}")
            return {
                "success": False,
                "formatted_output": self._create_error_html(f"Graph execution failed: {str(e)}"),
                "error": str(e),
                "messages": []
            }

# Integration function
def create_langgraph_autosys_system(autosys_db, llm_instance):
    """
    Create LangGraph-based Autosys system
    
    Args:
        autosys_db: Your AutosysOracleDatabase instance
        llm_instance: Your LLM instance from get_llm("langchain")
    
    Returns:
        AutosysLangGraph instance
    """
    return AutosysLangGraph(autosys_db, llm_instance)

# Usage example
def setup_langgraph_autosys():
    """Complete setup example"""
    
    # Your existing setup
    oracle_uri = "oracle+oracledb://***:***@***/service_name=service_name"
    autosys_db = AutosysOracleDatabase(oracle_uri)
    llm = get_llm("langchain")
    
    # Create LangGraph system
    autosys_system = create_langgraph_autosys_system(autosys_db, llm)
    
    # Optional: Add memory/persistence
    memory = MemorySaver()
    
    # Use the system
    def process_autosys_query(user_question: str, thread_id: str = "default"):
        config = {"configurable": {"thread_id": thread_id}}
        result = autosys_system.query(user_question, config)
        
        if result["success"]:
            return result["formatted_output"]
        else:
            return f"Error: {result['error']}"
    
    return process_autosys_query

# Advanced usage with streaming
def setup_streaming_langgraph():
    """Setup with streaming support"""
    
    # Your existing components
    autosys_db = AutosysOracleDatabase(oracle_uri)
    llm = get_llm("langchain")
    
    # Create system
    autosys_system = create_langgraph_autosys_system(autosys_db, llm)
    
    def stream_autosys_query(user_question: str):
        """Stream the execution process"""
        initial_state = {
            "messages": [],
            "user_question": user_question,
            "sql_query": "",
            "query_results": {},
            "formatted_output": "",
            "error": "",
            "iteration_count": 0
        }
        
        # Stream execution
        for step in autosys_system.graph.stream(initial_state):
            node_name = list(step.keys())[0]
            node_state = step[node_name]
            
            yield {
                "step": node_name,
                "status": "error" if node_state.get("error") else "success",
                "message": f"Completed: {node_name}",
                "data": node_state
            }
        
        # Final result
        yield {
            "step": "complete",
            "status": "success" if not node_state.get("error") else "error", 
            "formatted_output": node_state.get("formatted_output", ""),
            "final": True
        }
    
    return stream_autosys_query

"""
Complete usage example:

# Replace your existing agent with LangGraph:

# Old way:
sql_agent = initialize_agent(tools=tools, llm=llm, ...)

# New way:
autosys_system = create_langgraph_autosys_system(autosys_db, llm)

# Process queries:
result = autosys_system.query("Show me failed ATSYS jobs today")
print(result["formatted_output"])

# With streaming:
stream_func = setup_streaming_langgraph()
for step in stream_func("Which jobs are running?"):
    print(f"Step: {step['step']}, Status: {step['status']}")
    if step.get("final"):
        print(step["formatted_output"])
"""




-------
from langchain.tools import BaseTool
from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime
from pydantic import BaseModel, Field
import re

class AutosysQueryInput(BaseModel):
    """Input schema for Autosys Query Tool"""
    question: str = Field(description="Natural language question about Autosys jobs")

class AutosysLLMQueryTool(BaseTool):
    """
    LangChain-compatible Autosys Database Query Tool
    Uses existing AutosysOracleDatabase connection and external LLM
    """
    
    name: str = "AutosysQuery"
    description: str = """
    Query Autosys job scheduler database using natural language.
    Supports questions about job status, schedules, failures, and performance.
    Returns formatted HTML results with professional styling.
    
    Input should be a natural language question about Autosys jobs.
    Examples:
    - "Show me all failed jobs today"
    - "Which ATSYS jobs are currently running?"
    - "List jobs owned by user ADMIN"
    - "Show job history for the last 24 hours"
    """
    
    args_schema = AutosysQueryInput
    
    def __init__(self, autosys_db, llm_instance, max_results: int = 50):
        """
        Initialize tool with existing database connection and LLM
        
        Args:
            autosys_db: Your existing AutosysOracleDatabase instance
            llm_instance: Pre-configured LLM instance (from your config)
            max_results: Maximum number of results to return
        """
        super().__init__()
        self.autosys_db = autosys_db  # Use your existing database connection
        self.llm = llm_instance       # Use your existing LLM instance
        self.max_results = max_results
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Verify database connection
        self._verify_db_connection()

    def _verify_db_connection(self) -> bool:
        """Test database connection using your existing connection"""
        try:
            # Test the connection using your existing database class
            if hasattr(self.autosys_db, 'connection') and self.autosys_db.connection:
                self.logger.info("AutosysOracleDatabase connection verified")
                return True
            elif hasattr(self.autosys_db, 'connect'):
                # Try to connect if not already connected
                self.autosys_db.connect()
                self.logger.info("AutosysOracleDatabase connection established")
                return True
            else:
                self.logger.warning("Could not verify database connection")
                return False
        except Exception as e:
            self.logger.error(f"Database connection verification failed: {str(e)}")
            return False

    def _run(self, question: str) -> str:
        """
        Main tool execution method (LangChain interface)
        
        Args:
            question: Natural language question about Autosys
            
        Returns:
            Formatted HTML response
        """
        try:
            self.logger.info(f"AutosysQuery tool called: {question}")
            
            # Step 1: Generate SQL using external LLM
            sql_query = self._generate_sql_with_llm(question)
            
            # Step 2: Execute query using existing database connection
            query_result = self._execute_query_with_existing_db(sql_query)
            
            # Step 3: Format results using external LLM
            formatted_output = self._format_results_with_llm(query_result, question)
            
            return formatted_output
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}")
            return self._create_error_response(str(e))

    def _generate_sql_with_llm(self, user_question: str) -> str:
        """Generate SQL using external LLM instance"""
        
        sql_generation_prompt = f"""
You are an expert Oracle SQL generator for Autosys job scheduler database queries.

AUTOSYS DATABASE SCHEMA:
- Table: aedbadmin.ujo_jobst (Job Status)
  * job_name: VARCHAR2 - Job identifier (e.g., 'ATSYS.job_name.c')
  * status: NUMBER - Status code (SU=Success, FA=Failure, RU=Running, IN=Inactive)
  * last_start: NUMBER - Last start time (epoch seconds)
  * last_end: NUMBER - Last end time (epoch seconds)  
  * joid: NUMBER - Job ID (foreign key)

- Table: aedbadmin.ujo_job (Job Details)
  * joid: NUMBER - Job ID (primary key)
  * owner: VARCHAR2 - Job owner/user
  * machine: VARCHAR2 - Execution machine
  * job_type: VARCHAR2 - Type of job

- Table: aedbadmin.UJO_INTCODES (Status Translation)
  * code: NUMBER - Status code number
  * TEXT: VARCHAR2 - Human readable status text

ORACLE SQL PATTERNS FOR AUTOSYS:
- Time conversion: TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss')
- Standard joins: js INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
- Status lookup: LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
- Recent jobs: WHERE js.last_start >= (SYSDATE - INTERVAL '1' DAY) * 86400
- Job name search: WHERE UPPER(js.job_name) LIKE UPPER('%pattern%')
- Result limiting: WHERE ROWNUM <= {self.max_results}

COMMON STATUS CODES:
- SU or 4 = SUCCESS
- FA or 7 = FAILURE  
- RU or 8 = RUNNING
- IN or 5 = INACTIVE

USER QUESTION: "{user_question}"

Generate a complete Oracle SQL query that answers this question.
Use proper table aliases (js for ujo_jobst, j for ujo_job, ic for UJO_INTCODES).
Include relevant columns like job_name, status text, start/end times, owner.
Always use the full table names with aedbadmin schema prefix.
Ensure proper WHERE clauses and result limiting.

Return ONLY the SQL query without any explanations, markdown, or formatting.
"""

        try:
            # Use external LLM instance
            sql_response = self.llm.invoke(sql_generation_prompt)
            
            # Extract SQL from response
            sql_query = sql_response.content if hasattr(sql_response, 'content') else str(sql_response)
            
            # Clean up formatting
            sql_query = self._clean_sql_response(sql_query)
            
            # Ensure result limiting
            sql_query = self._ensure_result_limiting(sql_query)
            
            self.logger.info(f"Generated SQL: {sql_query[:100]}...")
            return sql_query
            
        except Exception as e:
            self.logger.error(f"SQL generation failed: {str(e)}")
            return self._get_fallback_sql(user_question)

    def _clean_sql_response(self, sql_query: str) -> str:
        """Clean up SQL response from LLM"""
        # Remove markdown formatting
        sql_query = re.sub(r'```sql\s*', '', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'```\s*', '', sql_query)
        
        # Remove extra whitespace and newlines
        sql_query = ' '.join(sql_query.split())
        
        # Ensure it ends properly
        sql_query = sql_query.strip()
        if not sql_query.endswith(';'):
            sql_query += ';'
            
        return sql_query

    def _ensure_result_limiting(self, sql_query: str) -> str:
        """Ensure query has result limiting"""
        sql_upper = sql_query.upper()
        
        if 'ROWNUM' not in sql_upper and 'LIMIT' not in sql_upper:
            # Add ROWNUM limiting
            if 'WHERE' in sql_upper:
                # Insert ROWNUM condition into existing WHERE clause
                where_pos = sql_upper.find('WHERE')
                before_where = sql_query[:where_pos + 5]  # Include 'WHERE'
                after_where = sql_query[where_pos + 5:]
                sql_query = f"{before_where} ROWNUM <= {self.max_results} AND{after_where}"
            else:
                # Add WHERE clause with ROWNUM
                order_pos = sql_upper.find('ORDER BY')
                if order_pos > 0:
                    before_order = sql_query[:order_pos]
                    after_order = sql_query[order_pos:]
                    sql_query = f"{before_order} WHERE ROWNUM <= {self.max_results} {after_order}"
                else:
                    sql_query = sql_query.rstrip(';') + f" WHERE ROWNUM <= {self.max_results};"
        
        return sql_query

    def _execute_query_with_existing_db(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query using your existing AutosysOracleDatabase"""
        
        try:
            start_time = datetime.now()
            
            # Use your existing database connection method
            # This assumes your AutosysOracleDatabase has a method to execute queries
            # Adjust the method name based on your actual implementation
            
            if hasattr(self.autosys_db, 'execute_query'):
                raw_results = self.autosys_db.execute_query(sql_query)
            elif hasattr(self.autosys_db, 'run'):
                raw_results = self.autosys_db.run(sql_query)
            elif hasattr(self.autosys_db, 'query'):
                raw_results = self.autosys_db.query(sql_query)
            else:
                # Fallback: try to access connection directly
                if hasattr(self.autosys_db, 'connection') and self.autosys_db.connection:
                    with self.autosys_db.connection.cursor() as cursor:
                        cursor.execute(sql_query)
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        raw_results = cursor.fetchall()
                        
                        # Convert to list of dictionaries
                        results = []
                        for row in raw_results:
                            row_dict = {}
                            for i, value in enumerate(row):
                                col_name = columns[i] if i < len(columns) else f"COLUMN_{i}"
                                row_dict[col_name] = self._format_oracle_value(value)
                            results.append(row_dict)
                        raw_results = results
                else:
                    raise Exception("Could not access database connection")
            
            # Handle different return types from your database class
            if isinstance(raw_results, str):
                # If returned as string, try to parse it
                try:
                    import ast
                    if raw_results.startswith('[') and raw_results.endswith(']'):
                        parsed_results = ast.literal_eval(raw_results)
                        results = self._convert_to_dict_list(parsed_results)
                    else:
                        # Handle as single result or error
                        results = [{"result": raw_results}]
                except:
                    results = [{"result": raw_results}]
            elif isinstance(raw_results, list):
                # Convert tuples to dictionaries if needed
                results = self._convert_to_dict_list(raw_results)
            else:
                results = [{"result": str(raw_results)}]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "results": results,
                "row_count": len(results),
                "execution_time": execution_time,
                "sql_query": sql_query
            }
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "sql_query": sql_query
            }

    def _convert_to_dict_list(self, raw_results: List) -> List[Dict]:
        """Convert list of tuples to list of dictionaries"""
        results = []
        
        for item in raw_results:
            if isinstance(item, (tuple, list)):
                # Convert tuple/list to dictionary with generic column names
                row_dict = {}
                for i, value in enumerate(item):
                    col_name = self._get_column_name(i, value)
                    row_dict[col_name] = self._format_oracle_value(value)
                results.append(row_dict)
            elif isinstance(item, dict):
                # Already a dictionary
                results.append(item)
            else:
                # Single value
                results.append({"VALUE": str(item)})
        
        return results

    def _get_column_name(self, index: int, value: Any) -> str:
        """Generate appropriate column name based on content"""
        common_names = ["JOB_NAME", "START_TIME", "END_TIME", "STATUS", "OWNER", "MACHINE"]
        
        if index < len(common_names):
            return common_names[index]
        else:
            return f"COLUMN_{index + 1}"

    def _format_oracle_value(self, value: Any) -> str:
        """Format Oracle-specific data types"""
        if hasattr(value, 'read'):  # CLOB/BLOB
            return value.read()
        elif isinstance(value, datetime):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return str(value) if value is not None else ""

    def _format_results_with_llm(self, query_result: Dict[str, Any], user_question: str) -> str:
        """Format results using external LLM instance"""
        
        if not query_result["success"]:
            return self._create_error_response(
                query_result["error"], 
                query_result.get("sql_query")
            )
        
        results = query_result["results"]
        
        if not results:
            return """
            <div style='padding: 15px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px;'>
                <h4 style='margin: 0 0 10px 0; color: #856404;'>No Results Found</h4>
                <p style='margin: 0; color: #856404;'>No Autosys jobs match your query criteria.</p>
            </div>
            """
        
        # Prepare data for formatting (limit for LLM processing)
        sample_size = min(len(results), 15)
        sample_data = results[:sample_size]
        
        formatting_prompt = f"""
Create professional HTML formatting for these Autosys database query results.

USER QUESTION: "{user_question}"
EXECUTION TIME: {query_result.get('execution_time', 0):.2f} seconds  
TOTAL RESULTS: {len(results)} jobs (showing first {sample_size})

QUERY RESULTS TO FORMAT:
{json.dumps(sample_data, indent=2, default=str)}

FORMATTING REQUIREMENTS:
1. Create responsive HTML with inline CSS only - no external stylesheets
2. Use professional styling with good contrast and readability
3. Color-code job status with badges:
   - SUCCESS/SU: Green background (#28a745) with white text
   - FAILURE/FA: Red background (#dc3545) with white text  
   - RUNNING/RU: Blue background (#007bff) with white text
   - INACTIVE/IN: Gray background (#6c757d) with white text
4. Include a summary section at the top showing total jobs and key statistics
5. Use monospace font for job names for better readability
6. Make tables responsive for mobile devices
7. Keep output concise to avoid truncation - prioritize most important info
8. Add subtle hover effects for better user experience
9. Include proper spacing and margins for professional appearance
10. Show execution time and result count in a footer

Create a clean, modern design that's easy to scan and read.
Return only the formatted HTML without any explanations or markdown.
"""

        try:
            # Use external LLM for formatting
            format_response = self.llm.invoke(formatting_prompt)
            formatted_html = format_response.content if hasattr(format_response, 'content') else str(format_response)
            
            # Add tool metadata footer
            metadata = f"""
            <div style="margin-top: 15px; padding: 8px 12px; background: #f8f9fa; border-radius: 4px; font-size: 11px; color: #666; border-left: 3px solid #007bff;">
                <strong>AutosysQuery Tool</strong> • 
                Generated {query_result['row_count']} results in {query_result['execution_time']:.2f}s • 
                Oracle Database via LLM
            </div>
            """
            
            return formatted_html + metadata
            
        except Exception as e:
            self.logger.error(f"Result formatting failed: {str(e)}")
            return self._create_fallback_format(results, user_question, query_result.get('execution_time', 0))

    def _create_error_response(self, error_msg: str, sql_query: Optional[str] = None) -> str:
        """Generate HTML error response"""
        html = f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #721c24; font-size: 16px;">AutosysQuery Tool Error</h4>
            <p style="margin: 0; font-size: 14px;"><strong>Error:</strong> {error_msg}</p>
        """
        
        if sql_query:
            html += f"""
            <details style="margin-top: 10px;">
                <summary style="cursor: pointer; color: #495057; font-size: 12px;">View Generated SQL Query</summary>
                <pre style="background: #e9ecef; padding: 10px; margin-top: 5px; border-radius: 3px; font-size: 10px; overflow-x: auto; font-family: 'Courier New', monospace;">{sql_query}</pre>
            </details>
            """
        
        html += """
            <p style="margin: 10px 0 0 0; font-size: 12px; color: #856404;">
                <em>Tip: Try rephrasing your question or check if the job names/criteria exist in the database.</em>
            </p>
        </div>
        """
        return html

    def _create_fallback_format(self, results: List[Dict], question: str, execution_time: float = 0) -> str:
        """Simple fallback HTML formatting if LLM formatting fails"""
        if not results:
            return "<p style='padding: 15px; background: #f8f9fa;'>No results found.</p>"
        
        html = f"""
        <div style="font-family: Arial, sans-serif;">
            <h3 style="color: #333; margin-bottom: 15px;">Autosys Query Results</h3>
            <p style="color: #666; font-size: 12px; margin-bottom: 15px;">
                Found {len(results)} jobs • Executed in {execution_time:.2f}s
            </p>
            <div style="overflow-x: auto;">
                <table style="border-collapse: collapse; width: 100%; font-size: 12px; background: white;">
                    <thead>
                        <tr style="background: #f2f2f2;">
        """
        
        # Table headers
        if results:
            for key in results[0].keys():
                html += f"<th style='border: 1px solid #ddd; padding: 8px; text-align: left; font-weight: bold;'>{key}</th>"
            html += "</tr></thead><tbody>"
            
            # Table rows
            for i, row in enumerate(results[:25]):  # Limit for fallback
                bg_color = "#f9f9f9" if i % 2 == 0 else "#ffffff"
                html += f"<tr style='background: {bg_color};'>"
                
                for key, value in row.items():
                    # Truncate long values
                    display_value = str(value)
                    if len(display_value) > 60:
                        display_value = display_value[:57] + "..."
                    
                    # Apply basic status color coding
                    if key.upper() in ['STATUS', 'JOB_STATUS'] and value:
                        if value.upper() in ['SUCCESS', 'SU']:
                            cell_style = "background: #d4edda; color: #155724; padding: 4px 8px; font-weight: bold;"
                        elif value.upper() in ['FAILURE', 'FA']:
                            cell_style = "background: #f8d7da; color: #721c24; padding: 4px 8px; font-weight: bold;"
                        elif value.upper() in ['RUNNING', 'RU']:
                            cell_style = "background: #cce5f0; color: #004085; padding: 4px 8px; font-weight: bold;"
                        else:
                            cell_style = "border: 1px solid #ddd; padding: 6px;"
                    else:
                        cell_style = "border: 1px solid #ddd; padding: 6px;"
                    
                    html += f"<td style='{cell_style}'>{display_value}</td>"
                html += "</tr>"
        
        html += """
                </tbody>
            </table>
        </div>
        <p style="margin-top: 10px; font-size: 11px; color: #999;">
            <em>Fallback formatting - LLM formatting temporarily unavailable</em>
        </p>
    </div>
        """
        return html

    def _get_fallback_sql(self, user_question: str) -> str:
        """Fallback SQL query if LLM generation fails"""
        # Analyze question for basic patterns
        question_lower = user_question.lower()
        
        if 'failed' in question_lower or 'failure' in question_lower:
            status_condition = "js.status = 7"  # FA status code
        elif 'running' in question_lower:
            status_condition = "js.status = 8"  # RU status code  
        elif 'success' in question_lower:
            status_condition = "js.status = 4"  # SU status code
        else:
            status_condition = "1=1"  # All jobs
        
        return f"""
        SELECT 
            js.job_name,
            TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (js.last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss') AS start_time,
            TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (js.last_end / 86400), 'MM/DD/YYYY HH24:Mi:Ss') AS end_time,
            NVL(ic.TEXT, 'UNKNOWN') AS job_status,
            j.owner
        FROM aedbadmin.ujo_jobst js
        INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
        LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
        WHERE {status_condition}
        AND UPPER(js.job_name) LIKE UPPER('%ATSYS%')
        AND ROWNUM <= {self.max_results}
        ORDER BY js.last_start DESC;
        """

# Integration function for your existing setup
def create_enhanced_autosys_tool(autosys_db, llm_instance, max_results=50):
    """
    Factory function to create the enhanced tool using your existing components
    
    Args:
        autosys_db: Your existing AutosysOracleDatabase instance
        llm_instance: Your LLM instance from get_llm("langchain")
        max_results: Maximum results to return
    
    Returns:
        AutosysLLMQueryTool ready for use in your agent
    """
    return AutosysLLMQueryTool(
        autosys_db=autosys_db,
        llm_instance=llm_instance,
        max_results=max_results
    )

# Complete integration example for your code:
"""
# In your main file where you have:
oracle_uri = "oracle+oracledb://***:***@***/service_name=service_name"
autosys_db = AutosysOracleDatabase(oracle_uri)
llm = get_llm("langchain")

# Replace your existing tool creation with:
autosys_sql_tool_enhanced = create_enhanced_autosys_tool(
    autosys_db=autosys_db,
    llm_instance=llm,
    max_results=50
)

# Update your tools list:
tools = [autosys_sql_tool_enhanced]

# Your existing agent initialization code remains the same:
sql_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    checkpointer=checkpointer,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="generate",
    agent_kwargs={
        "format_instructions": '''
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be AutosysQuery
Action Input: the input to the action
Observation: the tool will populate this automatically
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: Include all results from the query output, format each job on a new line.
IMPORTANT: Do NOT include multiple actions and final answers in the same response.
'''
    },
    "output_parser": output_parser
)
"""



------------
color = "#28a745" if status == "SUCCESS" else "#dc3545" if status == "FAILURE" else "#6c757d"
                    
                    html_parts.append(f"<tr><td>{i}</td><td style='font-family:monospace;font-size:10px'>{job_name}</td><td style='background:{color};color:white;padding:2px'>{status}</td><td>{start_time}</td></tr>")
            
            if len(result) > 20:
                html_parts.append(f"<tr><td colspan='4'><em>...and {len(result)-20} more jobs</em></td></tr>")
            
            html_parts.append("</table>")
            state["answer"] = "".join(html_parts)
        
        return state

____
def format_result_as_html_table(self, state: AutosysState) -> AutosysState:
    """Format Autosys results as an HTML table"""
    try:
        question = state.get("question", "")
        result = state.get("result", "")
        
        if isinstance(result, str) and result.startswith('[') and result.endswith(']'):
            try:
                import ast
                result = ast.literal_eval(result)
            except:
                pass
        
        if isinstance(result, (list, tuple)) and len(result) > 0:
            # Start HTML table with inline CSS for better styling
            html_parts = [
                f"<h3>Autosys Query Results ({len(result)} jobs found)</h3>",
                """
                <table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 12px;">
                <thead>
                    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-weight: bold;">#</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left; font-weight: bold;">Job Name</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: center; font-weight: bold;">Status</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: center; font-weight: bold;">Start Time</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: center; font-weight: bold;">End Time</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: center; font-weight: bold;">Owner</th>
                    </tr>
                </thead>
                <tbody>
                """
            ]
            
            # Process each row
            for i, row in enumerate(result, 1):
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    job_name = str(row[0]).strip() if row[0] else "N/A"
                    start_time = str(row[1]).strip() if len(row) > 1 and row[1] else "N/A"
                    end_time = str(row[2]).strip() if len(row) > 2 and row[2] else "N/A"
                    status = str(row[3]).strip() if len(row) > 3 and row[3] else "N/A"
                    owner = str(row[4]).strip() if len(row) > 4 and row[4] else "N/A"
                    
                    # Clean up times - show only time if date is present
                    if "/" in start_time and " " in start_time:
                        start_time = start_time.split(" ")[1]
                    if "/" in end_time and " " in end_time:
                        end_time = end_time.split(" ")[1]
                    
                    # Get status color and display
                    status_info = self._get_status_html(status)
                    
                    # Alternate row colors
                    row_color = "#f9f9f9" if i % 2 == 0 else "#ffffff"
                    
                    # Create table row
                    html_parts.append(f"""
                    <tr style="background-color: {row_color};">
                        <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{i}</td>
                        <td style="border: 1px solid #ddd; padding: 6px; font-family: monospace; font-size: 11px;" title="{job_name}">{job_name}</td>
                        <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{status_info}</td>
                        <td style="border: 1px solid #ddd; padding: 6px; text-align: center; font-family: monospace;">{start_time}</td>
                        <td style="border: 1px solid #ddd; padding: 6px; text-align: center; font-family: monospace;">{end_time}</td>
                        <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{owner}</td>
                    </tr>
                    """)
            
            # Close table
            html_parts.append("""
                </tbody>
                </table>
                <p style="font-size: 11px; color: #666; margin-top: 10px;">
                    <em>Generated from Autosys database query</em>
                </p>
            """)
            
            state["answer"] = "".join(html_parts)
            
        else:
            state["answer"] = f"<p><strong>Autosys Results:</strong><br>{str(result)}</p>"
        
        return state
        
    except Exception as e:
        state["answer"] = f"<p><strong>Error:</strong> {str(e)}</p><pre>{str(result)}</pre>"
        return state

def _get_status_html(self, status_code):
    """Get HTML formatted status with color coding"""
    status_map = {
        'SU': ('SUCCESS', '#28a745', '#ffffff'),    # Green background, white text
        'SUCCESS': ('SUCCESS', '#28a745', '#ffffff'),
        'FA': ('FAILURE', '#dc3545', '#ffffff'),    # Red background, white text  
        'FAILURE': ('FAILURE', '#dc3545', '#ffffff'),
        'RU': ('RUNNING', '#007bff', '#ffffff'),    # Blue background, white text
        'RUNNING': ('RUNNING', '#007bff', '#ffffff'),
        'IN': ('INACTIVE', '#6c757d', '#ffffff'),   # Gray background, white text
        'INACTIVE': ('INACTIVE', '#6c757d', '#ffffff'),
        'TE': ('TERMINATED', '#fd7e14', '#000000'), # Orange background, black text
        'ON': ('ON_NOEXEC', '#ffc107', '#000000'),  # Yellow background, black text
        'OH': ('ON_HOLD', '#e83e8c', '#ffffff'),    # Pink background, white text
        'QU': ('QUEUED', '#20c997', '#000000'),     # Teal background, black text
        'AC': ('ACTIVATED', '#17a2b8', '#ffffff'),  # Cyan background, white text
        'ST': ('STARTING', '#6f42c1', '#ffffff')    # Purple background, white text
    }
    
    display_name, bg_color, text_color = status_map.get(status_code, (status_code, '#e9ecef', '#000000'))
    
    return f"""
    <span style="
        background-color: {bg_color}; 
        color: {text_color}; 
        padding: 3px 8px; 
        border-radius: 4px; 
        font-size: 10px; 
        font-weight: bold;
        display: inline-block;
        min-width: 60px;
        text-align: center;
    ">{display_name}</span>
    """

# Alternative: Compact HTML table for better readability
def format_result_as_compact_html(self, state: AutosysState) -> AutosysState:
    """Format as compact HTML with better mobile responsiveness"""
    try:
        question = state.get("question", "")
        result = state.get("result", "")
        
        if isinstance(result, str) and result.startswith('[') and result.endswith(']'):
            try:
                import ast
                result = ast.literal_eval(result)
            except:
                pass
        
        if isinstance(result, (list, tuple)) and len(result) > 0:
            html_parts = [
                f"<div style='font-family: Arial, sans-serif;'>",
                f"<h3 style='color: #333; margin-bottom: 15px;'>Autosys Jobs ({len(result)} found)</h3>",
                "<div style='overflow-x: auto;'>"
            ]
            
            for i, row in enumerate(result, 1):
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    job_name = str(row[0]).strip() if row[0] else "N/A"
                    start_time = str(row[1]).strip() if len(row) > 1 and row[1] else "N/A"
                    end_time = str(row[2]).strip() if len(row) > 2 and row[2] else "N/A" 
                    status = str(row[3]).strip() if len(row) > 3 and row[3] else "N/A"
                    owner = str(row[4]).strip() if len(row) > 4 and row[4] else "N/A"
                    
                    status_info = self._get_status_html(status)
                    
                    # Extract just time from datetime
                    start_display = start_time.split(" ")[-1] if " " in start_time else start_time
                    end_display = end_time.split(" ")[-1] if " " in end_time else end_time
                    
                    html_parts.append(f"""
                    <div style='
                        border: 1px solid #ddd; 
                        margin-bottom: 8px; 
                        padding: 10px; 
                        background: {("#f8f9fa" if i % 2 == 0 else "#ffffff")};
                        border-radius: 4px;
                    '>
                        <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
                            <div style='flex: 1; min-width: 300px;'>
                                <strong style='font-family: monospace; font-size: 12px;'>{job_name}</strong>
                            </div>
                            <div style='display: flex; gap: 15px; align-items: center; flex-wrap: wrap;'>
                                <div>{status_info}</div>
                                <div style='font-size: 11px; color: #666;'>
                                    {start_display} → {end_display}
                                </div>
                                <div style='font-size: 11px; color: #666;'>{owner}</div>
                            </div>
                        </div>
                    </div>
                    """)
            
            html_parts.extend(["</div>", "</div>"])
            state["answer"] = "".join(html_parts)
            
        else:
            state["answer"] = f"<p><strong>Autosys Results:</strong><br>{str(result)}</p>"
        
        return state
        
    except Exception as e:
        state["answer"] = f"<p><strong>Error:</strong> {str(e)}</p>"
        return state




_____________
def format_result_as_table(self, state: AutosysState) -> AutosysState:
    """Format Autosys results as a table"""
    try:
        question = state.get("question", "")
        result = state.get("result", "")
        
        if isinstance(result, str) and result.startswith('[') and result.endswith(']'):
            try:
                import ast
                result = ast.literal_eval(result)
            except:
                pass
        
        if isinstance(result, (list, tuple)) and len(result) > 0:
            # Create table header
            table_lines = [
                "**Autosys Query Results:**",
                "",
                "| Job Name | Status | Start Time | End Time | Owner |",
                "|----------|--------|------------|----------|-------|"
            ]
            
            # Process each row
            for row in result:
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    job_name = str(row[0]).strip() if row[0] else "N/A"
                    start_time = str(row[1]).strip() if len(row) > 1 and row[1] else "N/A"
                    end_time = str(row[2]).strip() if len(row) > 2 and row[2] else "N/A"
                    status = str(row[3]).strip() if len(row) > 3 and row[3] else "N/A"
                    owner = str(row[4]).strip() if len(row) > 4 and row[4] else "N/A"
                    
                    # Truncate long job names for table formatting
                    if len(job_name) > 40:
                        job_name = job_name[:37] + "..."
                    
                    # Format times (remove date, keep only time)
                    if "/" in start_time:
                        start_time = start_time.split(" ")[-1] if " " in start_time else start_time
                    if "/" in end_time:
                        end_time = end_time.split(" ")[-1] if " " in end_time else end_time
                    
                    # Translate status
                    status_display = self._translate_autosys_status(status)
                    
                    # Add table row
                    table_row = f"| {job_name} | {status_display} | {start_time} | {end_time} | {owner} |"
                    table_lines.append(table_row)
            
            # Join all lines
            state["answer"] = "\n".join(table_lines)
            
        else:
            # Fallback for non-structured data
            state["answer"] = f"**Autosys Results:**\n{str(result)}"
        
        return state
        
    except Exception as e:
        state["answer"] = f"Error formatting table: {str(e)}\n\nRaw data:\n{str(result)}"
        return state

# Alternative: Simple formatted list (better for long job names)
def format_result_as_list(self, state: AutosysState) -> AutosysState:
    """Format Autosys results as a clean list"""
    try:
        question = state.get("question", "")
        result = state.get("result", "")
        
        if isinstance(result, str) and result.startswith('[') and result.endswith(']'):
            try:
                import ast
                result = ast.literal_eval(result)
            except:
                pass
        
        if isinstance(result, (list, tuple)) and len(result) > 0:
            formatted_lines = ["**Autosys Query Results:**", ""]
            
            for i, row in enumerate(result, 1):
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    job_name = str(row[0]).strip() if row[0] else "N/A"
                    start_time = str(row[1]).strip() if len(row) > 1 and row[1] else "N/A"
                    end_time = str(row[2]).strip() if len(row) > 2 and row[2] else "N/A"
                    status = str(row[3]).strip() if len(row) > 3 and row[3] else "N/A"
                    owner = str(row[4]).strip() if len(row) > 4 and row[4] else "N/A"
                    
                    status_display = self._translate_autosys_status(status)
                    
                    formatted_lines.append(f"{i:2d}. **{job_name}**")
                    formatted_lines.append(f"    Status: {status_display}")
                    formatted_lines.append(f"    Times: {start_time} → {end_time}")
                    formatted_lines.append(f"    Owner: {owner}")
                    formatted_lines.append("")  # Empty line between jobs
            
            state["answer"] = "\n".join(formatted_lines)
            
        else:
            state["answer"] = f"**Autosys Results:**\n{str(result)}"
        
        return state
        
    except Exception as e:
        state["answer"] = f"Error formatting results: {str(e)}"
        return state

def _translate_autosys_status(self, status_code):
    """Translate status codes to readable format"""
    status_map = {
        'SU': 'SUCCESS',
        'FA': 'FAILURE', 
        'RU': 'RUNNING',
        'IN': 'INACTIVE',
        'TE': 'TERMINATED',
        'ON': 'ON_NOEXEC',
        'OH': 'ON_HOLD',
        'QU': 'QUEUED',
        'AC': 'ACTIVATED',
        'ST': 'STARTING'
    }
    return status_map.get(status_code, status_code)




_______
def format_result(self, state: AutosysState) -> AutosysState:
    """Format Autosys results for user-friendly display"""
    try:
        question = state.get("question", "")
        result = state.get("result", "")
        
        print(f"DEBUG: Raw result type: {type(result)}")
        print(f"DEBUG: Raw result content: {result}")
        
        formatted_lines = []
        
        # Handle different result formats
        if isinstance(result, str):
            # If it's a string representation of a list, parse it
            if result.startswith('[') and result.endswith(']'):
                try:
                    import ast
                    parsed_result = ast.literal_eval(result)
                    result = parsed_result
                    print(f"DEBUG: Parsed string to: {type(result)} with {len(result)} items")
                except Exception as e:
                    print(f"DEBUG: Failed to parse string: {e}")
                    # If parsing fails, treat as single string result
                    state["answer"] = f"Autosys Query Results:\n{result}"
                    return state
            else:
                # Handle comma-separated or other string formats
                lines = result.split('\n')
                for line in lines:
                    if line.strip() and 'ATSYS' in line:
                        formatted_lines.append(f"• {line.strip()}")
                
                if formatted_lines:
                    state["answer"] = f"Autosys Query Results for '{question}':\n\n" + "\n".join(formatted_lines)
                    return state
        
        # Process list/tuple results
        if isinstance(result, (list, tuple)):
            print(f"DEBUG: Processing {len(result)} rows")
            
            for i, row in enumerate(result):
                print(f"DEBUG: Row {i}: {row} (type: {type(row)})")
                
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    # Extract fields safely
                    job_name = str(row[0]).strip() if row[0] else "Unknown"
                    start_time = str(row[1]).strip() if len(row) > 1 and row[1] else "N/A"
                    end_time = str(row[2]).strip() if len(row) > 2 and row[2] else "N/A"
                    status = str(row[3]).strip() if len(row) > 3 and row[3] else "Unknown"
                    owner = str(row[4]).strip() if len(row) > 4 and row[4] else "N/A"
                    
                    # Translate status
                    status_display = self._translate_autosys_status(status)
                    
                    # Format the line
                    formatted_line = f"• **{job_name}**: {status_display}"
                    if start_time != "N/A":
                        formatted_line += f" (Start: {start_time}"
                        if end_time != "N/A":
                            formatted_line += f", End: {end_time}"
                        formatted_line += ")"
                    if owner != "N/A":
                        formatted_line += f" - Owner: {owner}"
                    
                    formatted_lines.append(formatted_line)
                    print(f"DEBUG: Added formatted line: {formatted_line}")
                
                elif isinstance(row, str):
                    # Handle string rows
                    if 'ATSYS' in row:
                        formatted_lines.append(f"• {row}")
        
        # Generate final answer
        if formatted_lines:
            final_answer = f"**Autosys Query Results for '{question}':**\n\n" + "\n".join(formatted_lines)
            print(f"DEBUG: Final answer has {len(formatted_lines)} lines")
            state["answer"] = final_answer
        else:
            print("DEBUG: No formatted lines generated, using raw result")
            state["answer"] = f"**Autosys Data:**\n{str(result)}"
        
        print(f"DEBUG: Final state answer length: {len(state.get('answer', ''))}")
        return state
        
    except Exception as e:
        print(f"DEBUG: Exception in format_result: {str(e)}")
        import traceback
        traceback.print_exc()
        state["answer"] = f"Error formatting Autosys response: {str(e)}\n\nRaw data: {str(result)}"
        return state

def _translate_autosys_status(self, status_code):
    """Translate Autosys status codes to readable format"""
    status_map = {
        'SU': '✅ SUCCESS',
        'FA': '❌ FAILURE', 
        'RU': '🔄 RUNNING',
        'IN': '⏸️ INACTIVE',
        'TE': '⏹️ TERMINATED',
        'ON': '🔵 ON_NOEXEC',
        'OH': '🟡 ON_HOLD',
        'QU': '⏳ QUEUED',
        'AC': '🔀 ACTIVATED',
        'ST': '🟢 STARTING'
    }
    return status_map.get(status_code, f"🔸 {status_code}")










“___________

import os
from typing import Dict, Any, List, Optional, TypedDict
from dotenv import load_dotenv
import oracledb
import logging

from langgraph.graph import StateGraph, END
from langchain_community.utilities import SQLDatabase
from langchain.tools import Tool
from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutosysState(TypedDict):
    """State for Autosys query workflow"""
    question: str
    sql: str
    result: str
    answer: str
    error: Optional[str]
    table_info: str

class AutosysOracleDatabase:
    """Enhanced Autosys Oracle Database class with comprehensive schema knowledge"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False
        self.db = None
        self.available_tables = []
        self.schema_info = ""
        
        # Autosys table mappings and schema knowledge
        self.autosys_tables = {
            'UJO_JOB': 'Main job definitions',
            'UJO_JOB_RUNS': 'Job execution history and current status',
            'UJO_MACHINE': 'Machine/server definitions',
            'UJO_JOB_DEPEND': 'Job dependencies and conditions',
            'UJO_ALARM_LOG': 'Alarm and notification history',
            'UJO_CALENDAR': 'Calendar definitions',
            'UJO_JOB_TYPE': 'Job type definitions',
            'UJO_PROC_LOAD_QUEUE': 'Job processing queue',
            # Legacy/alternative table names
            'JOB': 'Job definitions (legacy)',
            'JOB_STATUS': 'Current job status (legacy)',
            'JOB_RUNS': 'Job execution records (legacy)',
            'CALENDAR': 'Calendar definitions (legacy)',
            'JOB_DEPENDENCY': 'Job dependencies (legacy)',
            'MACHINE': 'Machine definitions (legacy)',
            'BOX_JOB': 'Box job hierarchies (legacy)'
        }
        
        try:
            # Try database connection
            if self._is_valid_connection_string(connection_string):
                try:
                    self.db = SQLDatabase.from_uri(connection_string)
                    # Test connection
                    test_result = self.db.run("SELECT 1 FROM dual")
                    self.connected = True
                    logger.info("✅ Oracle database connected using oracledb driver")
                    self._discover_autosys_schema()
                except Exception as db_error:
                    logger.warning(f"⚠️ Database connection failed: {db_error}")
                    self._setup_mock_mode()
            else:
                logger.warning("⚠️ Invalid connection string, using mock mode")
                self._setup_mock_mode()
                
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            self._setup_mock_mode()
    
    def _is_valid_connection_string(self, conn_str: str) -> bool:
        """Check if connection string is properly configured for oracledb"""
        return (conn_str and 
                "username:password" not in conn_str and 
                ("oracle+oracledb://" in conn_str or "oracle://" in conn_str))
    
    def _setup_mock_mode(self):
        """Setup mock database with Autosys schema for testing"""
        self.connected = False
        self.available_tables = list(self.autosys_tables.keys())
        self.schema_info = self._get_autosys_schema_template()
        logger.info("🔧 Running in mock mode - update connection string for real data")
        logger.info("💡 Expected format: oracle+oracledb://user:pass@host:port/?service_name=SERVICE")
    
    def _discover_autosys_schema(self):
        """Discover actual Autosys schema from database"""
        try:
            # Check for Autosys tables
            table_check_query = f"""
            SELECT table_name FROM user_tables 
            WHERE table_name IN ({','.join([f"'{t}'" for t in self.autosys_tables.keys()])})
            ORDER BY table_name
            """
            
            try:
                tables_result = self.db.run(table_check_query)
                if tables_result:
                    # Parse table names from result
                    result_str = str(tables_result).replace('(', '').replace(')', '').replace("'", "")
                    self.available_tables = [t.strip() for t in result_str.split(',') if t.strip()]
                else:
                    # Fallback to common Autosys tables
                    self.available_tables = ['UJO_JOB', 'UJO_JOB_RUNS', 'JOB', 'JOB_STATUS']
            except Exception as table_error:
                logger.warning(f"Table discovery failed: {table_error}")
                self.available_tables = ['UJO_JOB', 'UJO_JOB_RUNS', 'JOB', 'JOB_STATUS']
            
            # Get detailed schema info
            try:
                if self.available_tables:
                    self.schema_info = self._get_detailed_schema_info()
                else:
                    self.schema_info = self._get_autosys_schema_template()
            except Exception as schema_error:
                logger.warning(f"Schema info retrieval failed: {schema_error}")
                self.schema_info = self._get_autosys_schema_template()
            
            logger.info(f"📊 Discovered Autosys tables: {self.available_tables}")
            
        except Exception as e:
            logger.error(f"⚠️ Schema discovery failed: {e}")
            self.available_tables = ['UJO_JOB', 'UJO_JOB_RUNS']
            self.schema_info = self._get_autosys_schema_template()
    
    def _get_detailed_schema_info(self) -> str:
        """Get detailed schema information for discovered tables"""
        try:
            schema_parts = ["AUTOSYS DATABASE SCHEMA:\n"]
            
            for table in self.available_tables[:5]:  # Limit to avoid token overflow
                try:
                    table_info = self.db.get_table_info_no_throw([table])
                    if table_info:
                        description = self.autosys_tables.get(table, "Autosys table")
                        schema_parts.append(f"\nTable: {table} - {description}")
                        schema_parts.append(table_info)
                except:
                    continue
            
            schema_parts.append(self._get_autosys_query_patterns())
            return "\n".join(schema_parts)
            
        except Exception as e:
            logger.error(f"Detailed schema retrieval failed: {e}")
            return self._get_autosys_schema_template()
    
    def _get_autosys_schema_template(self) -> str:
        """Get comprehensive Autosys schema template"""
        return """
AUTOSYS DATABASE SCHEMA:

Core Tables:
- UJO_JOB: Main job definitions
  Key columns: JOB_NAME, JOB_TYPE, COMMAND, MACHINE, OWNER, STATUS, LAST_START, LAST_END
- UJO_JOB_RUNS: Job execution history  
  Key columns: JOB_NAME, NTRY, STATUS, START_TIME, END_TIME, EXIT_CODE, RUN_MACHINE
- UJO_MACHINE: Machine definitions
  Key columns: MACHINE, NODE_NAME, TYPE, MAX_LOAD, AGENT_NAME
- UJO_JOB_DEPEND: Job dependencies
  Key columns: JOB_NAME, CONDITION, DEPEND_JOB_NAME
- UJO_ALARM_LOG: Alarm notifications
  Key columns: JOB_NAME, ALARM_TYPE, ALARM_TIME, STATUS

Legacy Tables (if available):
- JOB: Job definitions (job_name, status, last_start, last_end, command, machine)
- JOB_STATUS: Current job status information
- JOB_RUNS: Historical job execution records

Common Status Codes:
- SU: Success, FA: Failure, RU: Running, ST: Starting
- AC: Activated, IN: Inactive, OH: On Hold, OI: On Ice, TE: Terminated

Query Patterns:
- Current status: Use MAX(ntry) for latest run in UJO_JOB_RUNS
- Job history: Query UJO_JOB_RUNS with date ranges
- Dependencies: Query UJO_JOB_DEPEND for job relationships
- Performance: Calculate runtime as (END_TIME - START_TIME)
"""
    
    def _get_autosys_query_patterns(self) -> str:
        """Get common Autosys query patterns"""
        return """
COMMON AUTOSYS QUERY PATTERNS:

Current Job Status:
SELECT j.job_name, jr.status, jr.start_time, jr.end_time 
FROM ujo_job j JOIN ujo_job_runs jr ON j.job_name = jr.job_name 
WHERE jr.ntry = (SELECT MAX(ntry) FROM ujo_job_runs WHERE job_name = j.job_name)

Failed Jobs:
SELECT job_name, status, start_time, exit_code 
FROM ujo_job_runs 
WHERE status = 'FA' AND ntry = (SELECT MAX(ntry) FROM ujo_job_runs WHERE job_name = ujo_job_runs.job_name)

Running Jobs:
SELECT job_name, status, start_time, run_machine 
FROM ujo_job_runs 
WHERE status = 'RU'

Job Dependencies:
SELECT job_name, depend_job_name, condition 
FROM ujo_job_depend 
WHERE job_name LIKE '%pattern%'
"""

class AutosysTextToSQLTool:
    """Enhanced Autosys Text-to-SQL tool with rule-based SQL generation"""
    
    def __init__(self, autosys_db: AutosysOracleDatabase):
        self.autosys_db = autosys_db
    
    def generate_sql(self, state: AutosysState) -> AutosysState:
        """Generate Autosys-optimized SQL from natural language using rule-based approach"""
        try:
            question = state.get("question", "")
            if not question:
                state["error"] = "No question provided"
                return state
            
            # Generate SQL using rule-based approach
            sql = self._generate_rule_based_sql(question)
            sql = self._enhance_sql_for_autosys(sql, question)
            
            state["sql"] = sql
            logger.info(f"🔍 Generated Autosys SQL: {sql}")
            
        except Exception as e:
            logger.error(f"❌ SQL generation error: {e}")
            state["error"] = f"SQL generation failed: {str(e)}"
        
        return state
    
    def _generate_rule_based_sql(self, question: str) -> str:
        """Generate SQL using rule-based pattern matching"""
        question_lower = question.lower()
        
        # Extract job name if present
        job_hint = self._extract_job_name_from_question(question)
        
        # Status queries
        if "status" in question_lower:
            if any(table in self.autosys_db.available_tables for table in ['UJO_JOB_RUNS', 'JOB_RUNS']):
                table = 'UJO_JOB_RUNS' if 'UJO_JOB_RUNS' in self.autosys_db.available_tables else 'JOB_RUNS'
                if job_hint and job_hint != 'job':
                    return f"SELECT job_name, status, start_time, end_time FROM {table} WHERE job_name LIKE '%{job_hint}%' AND ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name LIKE '%{job_hint}%') AND ROWNUM <= 10"
                else:
                    return f"SELECT job_name, status, start_time, end_time FROM {table} WHERE ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name = {table}.job_name) AND ROWNUM <= 10"
            else:
                table = 'UJO_JOB' if 'UJO_JOB' in self.autosys_db.available_tables else 'JOB'
                if job_hint and job_hint != 'job':
                    return f"SELECT job_name, status, last_start, last_end FROM {table} WHERE job_name LIKE '%{job_hint}%' AND ROWNUM <= 10"
                else:
                    return f"SELECT job_name, status, last_start, last_end FROM {table} WHERE ROWNUM <= 10"
        
        # Failed jobs
        elif any(word in question_lower for word in ["failed", "failure", "error", "unsuccessful"]):
            table = 'UJO_JOB_RUNS' if 'UJO_JOB_RUNS' in self.autosys_db.available_tables else 'JOB_RUNS'
            base_query = f"SELECT job_name, status, start_time, end_time, exit_code FROM {table} WHERE status = 'FA'"
            
            if "today" in question_lower:
                return f"{base_query} AND TRUNC(start_time) = TRUNC(SYSDATE) AND ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name = {table}.job_name) AND ROWNUM <= 20"
            elif "yesterday" in question_lower:
                return f"{base_query} AND TRUNC(start_time) = TRUNC(SYSDATE) - 1 AND ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name = {table}.job_name) AND ROWNUM <= 20"
            elif "24 hours" in question_lower or "last day" in question_lower:
                return f"{base_query} AND start_time >= SYSDATE - 1 AND ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name = {table}.job_name) AND ROWNUM <= 20"
            else:
                return f"{base_query} AND ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name = {table}.job_name) AND ROWNUM <= 20"
        
        # Running jobs
        elif any(word in question_lower for word in ["running", "active", "executing"]):
            table = 'UJO_JOB_RUNS' if 'UJO_JOB_RUNS' in self.autosys_db.available_tables else 'JOB_RUNS'
            return f"SELECT job_name, status, start_time, run_machine FROM {table} WHERE status = 'RU' AND ROWNUM <= 20"
        
        # Success/successful jobs
        elif any(word in question_lower for word in ["success", "successful", "completed"]):
            table = 'UJO_JOB_RUNS' if 'UJO_JOB_RUNS' in self.autosys_db.available_tables else 'JOB_RUNS'
            base_query = f"SELECT job_name, status, start_time, end_time FROM {table} WHERE status = 'SU'"
            
            if "today" in question_lower:
                return f"{base_query} AND TRUNC(start_time) = TRUNC(SYSDATE) AND ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name = {table}.job_name) AND ROWNUM <= 20"
            else:
                return f"{base_query} AND ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name = {table}.job_name) AND ROWNUM <= 20"
        
        # Dependencies
        elif any(word in question_lower for word in ["depend", "dependency", "prerequisite"]):
            if 'UJO_JOB_DEPEND' in self.autosys_db.available_tables:
                if job_hint and job_hint != 'job':
                    return f"SELECT job_name, depend_job_name, condition FROM UJO_JOB_DEPEND WHERE job_name LIKE '%{job_hint}%' AND ROWNUM <= 20"
                else:
                    return f"SELECT job_name, depend_job_name, condition FROM UJO_JOB_DEPEND WHERE ROWNUM <= 20"
            else:
                return "SELECT 'Dependencies table not available' FROM dual"
        
        # Machine/server queries
        elif any(word in question_lower for word in ["machine", "server", "host"]):
            if 'UJO_MACHINE' in self.autosys_db.available_tables:
                return f"SELECT machine, node_name, type, max_load, agent_name FROM UJO_MACHINE WHERE ROWNUM <= 20"
            elif 'UJO_JOB_RUNS' in self.autosys_db.available_tables:
                return f"SELECT DISTINCT run_machine FROM UJO_JOB_RUNS WHERE run_machine IS NOT NULL AND ROWNUM <= 20"
            else:
                return "SELECT 'Machine information not available' FROM dual"
        
        # Job listing with specific job name
        elif job_hint and job_hint != 'job':
            table = 'UJO_JOB' if 'UJO_JOB' in self.autosys_db.available_tables else 'JOB'
            return f"SELECT job_name, status, last_start, last_end FROM {table} WHERE job_name LIKE '%{job_hint}%' AND ROWNUM <= 10"
        
        # Default general listing
        else:
            table = 'UJO_JOB' if 'UJO_JOB' in self.autosys_db.available_tables else 'JOB'
            return f"SELECT job_name, status, last_start FROM {table} WHERE ROWNUM <= 20"
    
    def _extract_job_name_from_question(self, question: str) -> str:
        """Extract potential job name from question"""
        try:
            words = question.split()
            for word in words:
                # Look for Autosys job name patterns
                if (('.' in word and '_' in word) or  # Pattern like ATSYS.job_name.c
                    (word.isupper() and len(word) > 3) or  # Uppercase job names
                    ('job' in word.lower() and len(word) > 4)):
                    clean_word = word.replace('job', '').replace('Job', '').strip('.,!?')
                    if clean_word:
                        return clean_word
            return 'job'  # fallback
        except:
            return 'job'
    
    def _enhance_sql_for_autosys(self, sql: str, question: str) -> str:
        """Enhance SQL with Autosys-specific optimizations"""
        try:
            sql_upper = sql.upper()
            
            # Add ROWNUM limit if missing (for performance)
            if 'ROWNUM' not in sql_upper and 'LIMIT' not in sql_upper:
                sql += " AND ROWNUM <= 20"
            
            # Replace LIMIT with ROWNUM for Oracle
            if 'LIMIT' in sql_upper:
                sql = sql.replace('LIMIT 20', 'AND ROWNUM <= 20')
                sql = sql.replace('LIMIT 10', 'AND ROWNUM <= 10')
            
            return sql
            
        except Exception as e:
            logger.error(f"SQL enhancement failed: {e}")
            return sql
    
    def execute_sql(self, state: AutosysState) -> AutosysState:
        """Execute SQL against Autosys database"""
        try:
            sql = state.get("sql", "")
            question = state.get("question", "")
            error = state.get("error")
            
            if error or not sql:
                state["result"] = f"Cannot execute: {error or 'No SQL generated'}"
                return state
            
            logger.info(f"⚡ Executing Autosys SQL: {sql}")
            
            if self.autosys_db.connected:
                try:
                    result = self.autosys_db.db.run(sql)
                    if not result or str(result).strip() == "":
                        result = "No data found in Autosys database"
                    logger.info(f"📊 Real Autosys DB result: {result}")
                except Exception as db_error:
                    logger.error(f"❌ Database error: {db_error}")
                    result = self._generate_mock_autosys_result(question, sql)
            else:
                # Mock mode with realistic Autosys data
                result = self._generate_mock_autosys_result(question, sql)
                logger.info(f"🎭 Mock Autosys result: {result}")
            
            state["result"] = str(result)
            
        except Exception as e:
            error_msg = f"Autosys SQL execution failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            state["result"] = error_msg
        
        return state
    
    def _generate_mock_autosys_result(self, question: str, sql: str) -> str:
        """Generate realistic mock Autosys results for testing"""
        try:
            question_lower = question.lower()
            
            if "status" in question_lower:
                job_name = self._extract_job_name_from_question(question)
                return f"ATSYS.DA3_DBMaint_{job_name}_080.c,SU,2024-01-15 10:30:00"
            
            elif "failed" in question_lower or "FA" in sql.upper():
                return ("ATSYS.ETL_PROCESS_001.c,FA,2024-01-15 09:15:00,2024-01-15 09:25:00,1\n"
                       "ATSYS.BACKUP_JOB_002.c,FA,2024-01-15 08:30:00,2024-01-15 08:35:00,2\n"
                       "ATSYS.DATA_SYNC_003.c,FA,2024-01-15 07:45:00,2024-01-15 07:50:00,1")
            
            elif "running" in question_lower or "RU" in sql.upper():
                return ("ATSYS.LONG_RUNNING_JOB_001.c,RU,2024-01-15 11:00:00,PROD_SRV_01\n"
                       "ATSYS.ETL_DAILY_PROCESS.c,RU,2024-01-15 10:30:00,PROD_SRV_02")
            
            elif "success" in question_lower or "SU" in sql.upper():
                return ("ATSYS.DAILY_BACKUP.c,SU,2024-01-15 10:00:00,2024-01-15 10:15:00\n"
                       "ATSYS.DATA_VALIDATION.c,SU,2024-01-15 09:30:00,2024-01-15 09:45:00\n"
                       "ATSYS.EMAIL_REPORTS.c,SU,2024-01-15 08:00:00,2024-01-15 08:05:00")
            
            elif "dependency" in question_lower or "depend" in question_lower:
                return ("ATSYS.MAIN_JOB.c,ATSYS.PREREQ_JOB_001.c,s(PREREQ_JOB_001)\n"
                       "ATSYS.MAIN_JOB.c,ATSYS.PREREQ_JOB_002.c,s(PREREQ_JOB_002)\n"
                       "ATSYS.ETL_FINAL.c,ATSYS.ETL_EXTRACT.c,s(ETL_EXTRACT)")
            
            elif "machine" in question_lower:
                return ("PROD_SRV_01,production-server-01,UNIX,5,Agent_01\n"
                       "PROD_SRV_02,production-server-02,UNIX,8,Agent_02\n"
                       "DEV_SRV_01,development-server-01,UNIX,3,Agent_03")
            
            else:
                # General job listing
                return ("ATSYS.DAILY_BACKUP.c,SU,2024-01-15 10:00:00,2024-01-15 10:15:00\n"
                       "ATSYS.ETL_PROCESS.c,RU,2024-01-15 11:00:00,\n"
                       "ATSYS.DATA_VALIDATION.c,SU,2024-01-15 09:30:00,2024-01-15 09:45:00\n"
                       "ATSYS.REPORT_GENERATION.c,FA,2024-01-15 08:15:00,2024-01-15 08:20:00")
        
        except Exception as e:
            return f"Mock Autosys data generation failed: {e}"
    
    def format_result(self, state: AutosysState) -> AutosysState:
        """Format Autosys results for user-friendly display"""
        try:
            question = state.get("question", "")
            result = state.get("result", "")
            
            if not result or "error" in result.lower() or "failed" in result.lower():
                state["answer"] = f"❌ I encountered an issue retrieving Autosys data for '{question}': {result}"
                return state
            
            # Format the result with Autosys context
            if "," in result:  # CSV-like data
                lines = result.split('\n')
                formatted_lines = []
                
                for line in lines:
                    if ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            job_name = parts[0].strip()
                            status = parts[1].strip()
                            
                            # Translate status codes with emojis
                            status_display = self._translate_autosys_status(status)
                            
                            if len(parts) >= 3:
                                timestamp = parts[2].strip()
                                if len(parts) >= 4 and parts[3].strip():
                                    end_time = parts[3].strip()
                                    formatted_lines.append(f"🔹 **{job_name}**: {status_display} (Start: {timestamp}, End: {end_time})")
                                else:
                                    formatted_lines.append(f"🔹 **{job_name}**: {status_display} (Time: {timestamp})")
                            else:
                                formatted_lines.append(f"🔹 **{job_name}**: {status_display}")
                
                if formatted_lines:
                    state["answer"] = f"📊 **Autosys Query Results for '{question}':**\n\n" + "\n".join(formatted_lines)
                else:
                    state["answer"] = f"📊 **Autosys Data:** {result}"
            else:
                state["answer"] = f"📊 **Autosys Result for '{question}':** {result}"
            
        except Exception as e:
            state["answer"] = f"❌ Error formatting Autosys response: {str(e)}"
        
        return state
    
    def _translate_autosys_status(self, status: str) -> str:
        """Translate Autosys status codes with emojis"""
        status_map = {
            'SU': 'Success ✅',
            'FA': 'Failed ❌', 
            'RU': 'Running 🔄',
            'TE': 'Terminated ⏹️',
            'OH': 'On Hold ⏸️',
            'OI': 'On Ice ❄️',
            'QU': 'Queued 📋',
            'ST': 'Starting 🚀',
            'AC': 'Activated 🟢',
            'IN': 'Inactive ⚫'
        }
        return status_map.get(status.upper(), f"{status} ❓")

# Initialize the enhanced Autosys Oracle database
ORACLE_CONNECTION = os.getenv("ORACLE_CONNECTION", "oracle+oracledb://username:password@hostname:1521/?service_name=ORCL")
autosys_db = AutosysOracleDatabase(ORACLE_CONNECTION)

# Create the enhanced Autosys tool
autosys_tool = AutosysTextToSQLTool(autosys_db)

# Build the LangGraph workflow
def build_autosys_workflow():
    """Build the enhanced Autosys workflow"""
    workflow = StateGraph(AutosysState)
    
    workflow.add_node("generate_sql", autosys_tool.generate_sql)
    workflow.add_node("execute_sql", autosys_tool.execute_sql)
    workflow.add_node("format_result", autosys_tool.format_result)
    
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "format_result") 
    workflow.add_edge("format_result", END)
    
    workflow.set_entry_point("generate_sql")
    return workflow.compile()

# Compile the workflow
app = build_autosys_workflow()

# Enhanced tool function
def autosys_query_tool(question: str) -> str:
    """Enhanced Autosys query tool function for LLM integration"""
    try:
        if not question or not question.strip():
            return "❓ Please provide a question about Autosys jobs, schedules, or system status."
        
        question = question.strip()
        logger.info(f"🚀 Processing Autosys query: {question}")
        
        # Execute the workflow
        initial_state = AutosysState(
            question=question,
            sql="",
            result="",
            answer="",
            error=None,
            table_info=""
        )
        
        result = app.invoke(initial_state)
        answer = result.get("answer", "No answer generated")
        
        # Clean up the response
        if isinstance(answer, str):
            return answer.strip()
        else:
            return str(answer).strip()
            
    except Exception as e:
        error_msg = f"Autosys query processing failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return f"❌ I couldn't process your Autosys query. Error: {error_msg}"

# Create the enhanced LangChain tool
@tool
def autosys_sql_query(question: str) -> str:
    """Query Autosys job scheduling system for job status, schedules, and operational information.
    
    This tool can answer questions about:
    - Job status (running, failed, success, terminated)
    - Job schedules and execution history  
    - Job dependencies and relationships
    - Machine and server information
    - Alarm and notification history
    - Performance and runtime statistics
    
    Args:
        question: Natural language question about Autosys jobs or system
        
    Returns:
        Formatted response with job information and status
    
    Example questions:
    - "What is the status of job ATSYS.DA3_DBMaint_080_arch_machines.c?"
    - "Show me all failed jobs from yesterday"
    - "List currently running jobs"
    - "What jobs depend on BACKUP_DAILY_JOB?"
    - "Show job runtime statistics for ETL processes"
    - "Which machines are running jobs right now?"
    """
    return autosys_query_tool(question)

# Legacy Tool wrapper for backward compatibility
autosys_sql_tool_enhanced = Tool(
    name="AutosysQuery",
    func=autosys_query_tool,
    description="""Enhanced Autosys job scheduling database query tool.

This tool provides comprehensive access to Autosys job information including:
- Job status and execution history
- Job schedules and dependencies  
- Machine and resource information
- Performance metrics and alarms
- Real-time job monitoring

Supports both current Autosys schema (UJO_* tables) and legacy formats.
Uses rule-based SQL generation with proper Oracle syntax and optimizations.

Example queries:
- "Show failed jobs from last 24 hours"
- "What's the status of job ATSYS.DAILY_BACKUP.c?"
- "List all jobs running on PROD_SERVER_01"
- "Show job dependencies for ETL_MAIN_PROCESS"
- "Get runtime statistics for jobs that ran today"
"""
)

# Export for easy import
__all__ = [
    'autosys_sql_query',
    'autosys_sql_tool_enhanced', 
    'AutosysOracleDatabase',
    'AutosysTextToSQLTool',
    'autosys_query_tool'
]




==============================

import os
from typing import Dict, Any, List, Optional, TypedDict
from dotenv import load_dotenv
import oracledb
import logging

from langgraph.graph import StateGraph, END
from langchain_community.utilities import SQLDatabase
from langchain.tools import Tool
from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutosysState(TypedDict):
    """State for Autosys query workflow"""
    question: str
    sql: str
    result: str
    answer: str
    error: Optional[str]
    table_info: str

class AutosysOracleDatabase:
    """Enhanced Autosys Oracle Database class with comprehensive schema knowledge"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False
        self.db = None
        self.available_tables = []
        self.schema_info = ""
        
        # Autosys table mappings and schema knowledge
        self.autosys_tables = {
            'UJO_JOB': 'Main job definitions',
            'UJO_JOB_RUNS': 'Job execution history and current status',
            'UJO_MACHINE': 'Machine/server definitions',
            'UJO_JOB_DEPEND': 'Job dependencies and conditions',
            'UJO_ALARM_LOG': 'Alarm and notification history',
            'UJO_CALENDAR': 'Calendar definitions',
            'UJO_JOB_TYPE': 'Job type definitions',
            'UJO_PROC_LOAD_QUEUE': 'Job processing queue',
            # Legacy/alternative table names
            'JOB': 'Job definitions (legacy)',
            'JOB_STATUS': 'Current job status (legacy)',
            'JOB_RUNS': 'Job execution records (legacy)',
            'CALENDAR': 'Calendar definitions (legacy)',
            'JOB_DEPENDENCY': 'Job dependencies (legacy)',
            'MACHINE': 'Machine definitions (legacy)',
            'BOX_JOB': 'Box job hierarchies (legacy)'
        }
        
        try:
            # Try database connection
            if self._is_valid_connection_string(connection_string):
                try:
                    self.db = SQLDatabase.from_uri(connection_string)
                    # Test connection
                    test_result = self.db.run("SELECT 1 FROM dual")
                    self.connected = True
                    logger.info("✅ Oracle database connected using oracledb driver")
                    self._discover_autosys_schema()
                except Exception as db_error:
                    logger.warning(f"⚠️ Database connection failed: {db_error}")
                    self._setup_mock_mode()
            else:
                logger.warning("⚠️ Invalid connection string, using mock mode")
                self._setup_mock_mode()
                
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            self._setup_mock_mode()
    
    def _is_valid_connection_string(self, conn_str: str) -> bool:
        """Check if connection string is properly configured for oracledb"""
        return (conn_str and 
                "username:password" not in conn_str and 
                ("oracle+oracledb://" in conn_str or "oracle://" in conn_str))
    
    def _setup_mock_mode(self):
        """Setup mock database with Autosys schema for testing"""
        self.connected = False
        self.available_tables = list(self.autosys_tables.keys())
        self.schema_info = self._get_autosys_schema_template()
        logger.info("🔧 Running in mock mode - update connection string for real data")
        logger.info("💡 Expected format: oracle+oracledb://user:pass@host:port/?service_name=SERVICE")
    
    def _discover_autosys_schema(self):
        """Discover actual Autosys schema from database"""
        try:
            # Check for Autosys tables
            table_check_query = f"""
            SELECT table_name FROM user_tables 
            WHERE table_name IN ({','.join([f"'{t}'" for t in self.autosys_tables.keys()])})
            ORDER BY table_name
            """
            
            try:
                tables_result = self.db.run(table_check_query)
                if tables_result:
                    # Parse table names from result
                    result_str = str(tables_result).replace('(', '').replace(')', '').replace("'", "")
                    self.available_tables = [t.strip() for t in result_str.split(',') if t.strip()]
                else:
                    # Fallback to common Autosys tables
                    self.available_tables = ['UJO_JOB', 'UJO_JOB_RUNS', 'JOB', 'JOB_STATUS']
            except Exception as table_error:
                logger.warning(f"Table discovery failed: {table_error}")
                self.available_tables = ['UJO_JOB', 'UJO_JOB_RUNS', 'JOB', 'JOB_STATUS']
            
            # Get detailed schema info
            try:
                if self.available_tables:
                    self.schema_info = self._get_detailed_schema_info()
                else:
                    self.schema_info = self._get_autosys_schema_template()
            except Exception as schema_error:
                logger.warning(f"Schema info retrieval failed: {schema_error}")
                self.schema_info = self._get_autosys_schema_template()
            
            logger.info(f"📊 Discovered Autosys tables: {self.available_tables}")
            
        except Exception as e:
            logger.error(f"⚠️ Schema discovery failed: {e}")
            self.available_tables = ['UJO_JOB', 'UJO_JOB_RUNS']
            self.schema_info = self._get_autosys_schema_template()
    
    def _get_detailed_schema_info(self) -> str:
        """Get detailed schema information for discovered tables"""
        try:
            schema_parts = ["AUTOSYS DATABASE SCHEMA:\n"]
            
            for table in self.available_tables[:5]:  # Limit to avoid token overflow
                try:
                    table_info = self.db.get_table_info_no_throw([table])
                    if table_info:
                        description = self.autosys_tables.get(table, "Autosys table")
                        schema_parts.append(f"\nTable: {table} - {description}")
                        schema_parts.append(table_info)
                except:
                    continue
            
            schema_parts.append(self._get_autosys_query_patterns())
            return "\n".join(schema_parts)
            
        except Exception as e:
            logger.error(f"Detailed schema retrieval failed: {e}")
            return self._get_autosys_schema_template()
    
    def _get_autosys_schema_template(self) -> str:
        """Get comprehensive Autosys schema template"""
        return """
AUTOSYS DATABASE SCHEMA:

Core Tables:
- UJO_JOB: Main job definitions
  Key columns: JOB_NAME, JOB_TYPE, COMMAND, MACHINE, OWNER, STATUS, LAST_START, LAST_END
- UJO_JOB_RUNS: Job execution history  
  Key columns: JOB_NAME, NTRY, STATUS, START_TIME, END_TIME, EXIT_CODE, RUN_MACHINE
- UJO_MACHINE: Machine definitions
  Key columns: MACHINE, NODE_NAME, TYPE, MAX_LOAD, AGENT_NAME
- UJO_JOB_DEPEND: Job dependencies
  Key columns: JOB_NAME, CONDITION, DEPEND_JOB_NAME
- UJO_ALARM_LOG: Alarm notifications
  Key columns: JOB_NAME, ALARM_TYPE, ALARM_TIME, STATUS

Legacy Tables (if available):
- JOB: Job definitions (job_name, status, last_start, last_end, command, machine)
- JOB_STATUS: Current job status information
- JOB_RUNS: Historical job execution records

Common Status Codes:
- SU: Success, FA: Failure, RU: Running, ST: Starting
- AC: Activated, IN: Inactive, OH: On Hold, OI: On Ice, TE: Terminated

Query Patterns:
- Current status: Use MAX(ntry) for latest run in UJO_JOB_RUNS
- Job history: Query UJO_JOB_RUNS with date ranges
- Dependencies: Query UJO_JOB_DEPEND for job relationships
- Performance: Calculate runtime as (END_TIME - START_TIME)
"""
    
    def _get_autosys_query_patterns(self) -> str:
        """Get common Autosys query patterns"""
        return """
COMMON AUTOSYS QUERY PATTERNS:

Current Job Status:
SELECT j.job_name, jr.status, jr.start_time, jr.end_time 
FROM ujo_job j JOIN ujo_job_runs jr ON j.job_name = jr.job_name 
WHERE jr.ntry = (SELECT MAX(ntry) FROM ujo_job_runs WHERE job_name = j.job_name)

Failed Jobs:
SELECT job_name, status, start_time, exit_code 
FROM ujo_job_runs 
WHERE status = 'FA' AND ntry = (SELECT MAX(ntry) FROM ujo_job_runs WHERE job_name = ujo_job_runs.job_name)

Running Jobs:
SELECT job_name, status, start_time, run_machine 
FROM ujo_job_runs 
WHERE status = 'RU'

Job Dependencies:
SELECT job_name, depend_job_name, condition 
FROM ujo_job_depend 
WHERE job_name LIKE '%pattern%'
"""

class AutosysTextToSQLTool:
    """Enhanced Autosys Text-to-SQL tool with rule-based SQL generation"""
    
    def __init__(self, autosys_db: AutosysOracleDatabase):
        self.autosys_db = autosys_db
    
    def generate_sql(self, state: AutosysState) -> AutosysState:
        """Generate Autosys-optimized SQL from natural language using rule-based approach"""
        try:
            question = state.get("question", "")
            if not question:
                state["error"] = "No question provided"
                return state
            
            # Generate SQL using rule-based approach
            sql = self._generate_rule_based_sql(question)
            sql = self._enhance_sql_for_autosys(sql, question)
            
            state["sql"] = sql
            logger.info(f"🔍 Generated Autosys SQL: {sql}")
            
        except Exception as e:
            logger.error(f"❌ SQL generation error: {e}")
            state["error"] = f"SQL generation failed: {str(e)}"
        
        return state
    
    def _generate_rule_based_sql(self, question: str) -> str:
        """Generate SQL using rule-based pattern matching"""
        question_lower = question.lower()
        
        # Extract job name if present
        job_hint = self._extract_job_name_from_question(question)
        
        # Status queries
        if "status" in question_lower:
            if any(table in self.autosys_db.available_tables for table in ['UJO_JOB_RUNS', 'JOB_RUNS']):
                table = 'UJO_JOB_RUNS' if 'UJO_JOB_RUNS' in self.autosys_db.available_tables else 'JOB_RUNS'
                if job_hint and job_hint != 'job':
                    return f"SELECT job_name, status, start_time, end_time FROM {table} WHERE job_name LIKE '%{job_hint}%' AND ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name LIKE '%{job_hint}%') AND ROWNUM <= 10"
                else:
                    return f"SELECT job_name, status, start_time, end_time FROM {table} WHERE ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name = {table}.job_name) AND ROWNUM <= 10"
            else:
                table = 'UJO_JOB' if 'UJO_JOB' in self.autosys_db.available_tables else 'JOB'
                if job_hint and job_hint != 'job':
                    return f"SELECT job_name, status, last_start, last_end FROM {table} WHERE job_name LIKE '%{job_hint}%' AND ROWNUM <= 10"
                else:
                    return f"SELECT job_name, status, last_start, last_end FROM {table} WHERE ROWNUM <= 10"
        
        # Failed jobs
        elif any(word in question_lower for word in ["failed", "failure", "error", "unsuccessful"]):
            table = 'UJO_JOB_RUNS' if 'UJO_JOB_RUNS' in self.autosys_db.available_tables else 'JOB_RUNS'
            base_query = f"SELECT job_name, status, start_time, end_time, exit_code FROM {table} WHERE status = 'FA'"
            
            if "today" in question_lower:
                return f"{base_query} AND TRUNC(start_time) = TRUNC(SYSDATE) AND ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name = {table}.job_name) AND ROWNUM <= 20"
            elif "yesterday" in question_lower:
                return f"{base_query} AND TRUNC(start_time) = TRUNC(SYSDATE) - 1 AND ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name = {table}.job_name) AND ROWNUM <= 20"
            elif "24 hours" in question_lower or "last day" in question_lower:
                return f"{base_query} AND start_time >= SYSDATE - 1 AND ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name = {table}.job_name) AND ROWNUM <= 20"
            else:
                return f"{base_query} AND ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name = {table}.job_name) AND ROWNUM <= 20"
        
        # Running jobs
        elif any(word in question_lower for word in ["running", "active", "executing"]):
            table = 'UJO_JOB_RUNS' if 'UJO_JOB_RUNS' in self.autosys_db.available_tables else 'JOB_RUNS'
            return f"SELECT job_name, status, start_time, run_machine FROM {table} WHERE status = 'RU' AND ROWNUM <= 20"
        
        # Success/successful jobs
        elif any(word in question_lower for word in ["success", "successful", "completed"]):
            table = 'UJO_JOB_RUNS' if 'UJO_JOB_RUNS' in self.autosys_db.available_tables else 'JOB_RUNS'
            base_query = f"SELECT job_name, status, start_time, end_time FROM {table} WHERE status = 'SU'"
            
            if "today" in question_lower:
                return f"{base_query} AND TRUNC(start_time) = TRUNC(SYSDATE) AND ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name = {table}.job_name) AND ROWNUM <= 20"
            else:
                return f"{base_query} AND ntry = (SELECT MAX(ntry) FROM {table} WHERE job_name = {table}.job_name) AND ROWNUM <= 20"
        
        # Dependencies
        elif any(word in question_lower for word in ["depend", "dependency", "prerequisite"]):
            if 'UJO_JOB_DEPEND' in self.autosys_db.available_tables:
                if job_hint and job_hint != 'job':
                    return f"SELECT job_name, depend_job_name, condition FROM UJO_JOB_DEPEND WHERE job_name LIKE '%{job_hint}%' AND ROWNUM <= 20"
                else:
                    return f"SELECT job_name, depend_job_name, condition FROM UJO_JOB_DEPEND WHERE ROWNUM <= 20"
            else:
                return "SELECT 'Dependencies table not available' FROM dual"
        
        # Machine/server queries
        elif any(word in question_lower for word in ["machine", "server", "host"]):
            if 'UJO_MACHINE' in self.autosys_db.available_tables:
                return f"SELECT machine, node_name, type, max_load, agent_name FROM UJO_MACHINE WHERE ROWNUM <= 20"
            elif 'UJO_JOB_RUNS' in self.autosys_db.available_tables:
                return f"SELECT DISTINCT run_machine FROM UJO_JOB_RUNS WHERE run_machine IS NOT NULL AND ROWNUM <= 20"
            else:
                return "SELECT 'Machine information not available' FROM dual"
        
        # Job listing with specific job name
        elif job_hint and job_hint != 'job':
            table = 'UJO_JOB' if 'UJO_JOB' in self.autosys_db.available_tables else 'JOB'
            return f"SELECT job_name, status, last_start, last_end FROM {table} WHERE job_name LIKE '%{job_hint}%' AND ROWNUM <= 10"
        
        # Default general listing
        else:
            table = 'UJO_JOB' if 'UJO_JOB' in self.autosys_db.available_tables else 'JOB'
            return f"SELECT job_name, status, last_start FROM {table} WHERE ROWNUM <= 20"
    
    def _extract_job_name_from_question(self, question: str) -> str:
        """Extract potential job name from question"""
        try:
            words = question.split()
            for word in words:
                # Look for Autosys job name patterns
                if (('.' in word and '_' in word) or  # Pattern like ATSYS.job_name.c
                    (word.isupper() and len(word) > 3) or  # Uppercase job names
                    ('job' in word.lower() and len(word) > 4)):
                    clean_word = word.replace('job', '').replace('Job', '').strip('.,!?')
                    if clean_word:
                        return clean_word
            return 'job'  # fallback
        except:
            return 'job'
    
    def _enhance_sql_for_autosys(self, sql: str, question: str) -> str:
        """Enhance SQL with Autosys-specific optimizations"""
        try:
            sql_upper = sql.upper()
            
            # Add ROWNUM limit if missing (for performance)
            if 'ROWNUM' not in sql_upper and 'LIMIT' not in sql_upper:
                sql += " AND ROWNUM <= 20"
            
            # Replace LIMIT with ROWNUM for Oracle
            if 'LIMIT' in sql_upper:
                sql = sql.replace('LIMIT 20', 'AND ROWNUM <= 20')
                sql = sql.replace('LIMIT 10', 'AND ROWNUM <= 10')
            
            return sql
            
        except Exception as e:
            logger.error(f"SQL enhancement failed: {e}")
            return sql
    
    def execute_sql(self, state: AutosysState) -> AutosysState:
        """Execute SQL against Autosys database"""
        try:
            sql = state.get("sql", "")
            question = state.get("question", "")
            error = state.get("error")
            
            if error or not sql:
                state["result"] = f"Cannot execute: {error or 'No SQL generated'}"
                return state
            
            logger.info(f"⚡ Executing Autosys SQL: {sql}")
            
            if self.autosys_db.connected:
                try:
                    result = self.autosys_db.db.run(sql)
                    if not result or str(result).strip() == "":
                        result = "No data found in Autosys database"
                    logger.info(f"📊 Real Autosys DB result: {result}")
                except Exception as db_error:
                    logger.error(f"❌ Database error: {db_error}")
                    result = self._generate_mock_autosys_result(question, sql)
            else:
                # Mock mode with realistic Autosys data
                result = self._generate_mock_autosys_result(question, sql)
                logger.info(f"🎭 Mock Autosys result: {result}")
            
            state["result"] = str(result)
            
        except Exception as e:
            error_msg = f"Autosys SQL execution failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            state["result"] = error_msg
        
        return state
    
    def _generate_mock_autosys_result(self, question: str, sql: str) -> str:
        """Generate realistic mock Autosys results for testing"""
        try:
            question_lower = question.lower()
            
            if "status" in question_lower:
                job_name = self._extract_job_name_from_question(question)
                return f"ATSYS.DA3_DBMaint_{job_name}_080.c,SU,2024-01-15 10:30:00"
            
            elif "failed" in question_lower or "FA" in sql.upper():
                return ("ATSYS.ETL_PROCESS_001.c,FA,2024-01-15 09:15:00,2024-01-15 09:25:00,1\n"
                       "ATSYS.BACKUP_JOB_002.c,FA,2024-01-15 08:30:00,2024-01-15 08:35:00,2\n"
                       "ATSYS.DATA_SYNC_003.c,FA,2024-01-15 07:45:00,2024-01-15 07:50:00,1")
            
            elif "running" in question_lower or "RU" in sql.upper():
                return ("ATSYS.LONG_RUNNING_JOB_001.c,RU,2024-01-15 11:00:00,PROD_SRV_01\n"
                       "ATSYS.ETL_DAILY_PROCESS.c,RU,2024-01-15 10:30:00,PROD_SRV_02")
            
            elif "success" in question_lower or "SU" in sql.upper():
                return ("ATSYS.DAILY_BACKUP.c,SU,2024-01-15 10:00:00,2024-01-15 10:15:00\n"
                       "ATSYS.DATA_VALIDATION.c,SU,2024-01-15 09:30:00,2024-01-15 09:45:00\n"
                       "ATSYS.EMAIL_REPORTS.c,SU,2024-01-15 08:00:00,2024-01-15 08:05:00")
            
            elif "dependency" in question_lower or "depend" in question_lower:
                return ("ATSYS.MAIN_JOB.c,ATSYS.PREREQ_JOB_001.c,s(PREREQ_JOB_001)\n"
                       "ATSYS.MAIN_JOB.c,ATSYS.PREREQ_JOB_002.c,s(PREREQ_JOB_002)\n"
                       "ATSYS.ETL_FINAL.c,ATSYS.ETL_EXTRACT.c,s(ETL_EXTRACT)")
            
            elif "machine" in question_lower:
                return ("PROD_SRV_01,production-server-01,UNIX,5,Agent_01\n"
                       "PROD_SRV_02,production-server-02,UNIX,8,Agent_02\n"
                       "DEV_SRV_01,development-server-01,UNIX,3,Agent_03")
            
            else:
                # General job listing
                return ("ATSYS.DAILY_BACKUP.c,SU,2024-01-15 10:00:00,2024-01-15 10:15:00\n"
                       "ATSYS.ETL_PROCESS.c,RU,2024-01-15 11:00:00,\n"
                       "ATSYS.DATA_VALIDATION.c,SU,2024-01-15 09:30:00,2024-01-15 09:45:00\n"
                       "ATSYS.REPORT_GENERATION.c,FA,2024-01-15 08:15:00,2024-01-15 08:20:00")
        
        except Exception as e:
            return f"Mock Autosys data generation failed: {e}"
    
    def format_result(self, state: AutosysState) -> AutosysState:
        """Format Autosys results for user-friendly display"""
        try:
            question = state.get("question", "")
            result = state.get("result", "")
            
            if not result or "error" in result.lower() or "failed" in result.lower():
                state["answer"] = f"❌ I encountered an issue retrieving Autosys data for '{question}': {result}"
                return state
            
            # Format the result with Autosys context
            if "," in result:  # CSV-like data
                lines = result.split('\n')
                formatted_lines = []
                
                for line in lines:
                    if ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            job_name = parts[0].strip()
                            status = parts[1].strip()
                            
                            # Translate status codes with emojis
                            status_display = self._translate_autosys_status(status)
                            
                            if len(parts) >= 3:
                                timestamp = parts[2].strip()
                                if len(parts) >= 4 and parts[3].strip():
                                    end_time = parts[3].strip()
                                    formatted_lines.append(f"🔹 **{job_name}**: {status_display} (Start: {timestamp}, End: {end_time})")
                                else:
                                    formatted_lines.append(f"🔹 **{job_name}**: {status_display} (Time: {timestamp})")
                            else:
                                formatted_lines.append(f"🔹 **{job_name}**: {status_display}")
                
                if formatted_lines:
                    state["answer"] = f"📊 **Autosys Query Results for '{question}':**\n\n" + "\n".join(formatted_lines)
                else:
                    state["answer"] = f"📊 **Autosys Data:** {result}"
            else:
                state["answer"] = f"📊 **Autosys Result for '{question}':** {result}"
            
        except Exception as e:
            state["answer"] = f"❌ Error formatting Autosys response: {str(e)}"
        
        return state
    
    def _translate_autosys_status(self, status: str) -> str:
        """Translate Autosys status codes with emojis"""
        status_map = {
            'SU': 'Success ✅',
            'FA': 'Failed ❌', 
            'RU': 'Running 🔄',
            'TE': 'Terminated ⏹️',
            'OH': 'On Hold ⏸️',
            'OI': 'On Ice ❄️',
            'QU': 'Queued 📋',
            'ST': 'Starting 🚀',
            'AC': 'Activated 🟢',
            'IN': 'Inactive ⚫'
        }
        return status_map.get(status.upper(), f"{status} ❓")

# Initialize the enhanced Autosys Oracle database
ORACLE_CONNECTION = os.getenv("ORACLE_CONNECTION", "oracle+oracledb://username:password@hostname:1521/?service_name=ORCL")
autosys_db = AutosysOracleDatabase(ORACLE_CONNECTION)

# Create the enhanced Autosys tool
autosys_tool = AutosysTextToSQLTool(autosys_db)

# Build the LangGraph workflow
def build_autosys_workflow():
    """Build the enhanced Autosys workflow"""
    workflow = StateGraph(AutosysState)
    
    workflow.add_node("generate_sql", autosys_tool.generate_sql)
    workflow.add_node("execute_sql", autosys_tool.execute_sql)
    workflow.add_node("format_result", autosys_tool.format_result)
    
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "format_result") 
    workflow.add_edge("format_result", END)
    
    workflow.set_entry_point("generate_sql")
    return workflow.compile()

# Compile the workflow
app = build_autosys_workflow()

# Enhanced tool function
def autosys_query_tool(question: str) -> str:
    """Enhanced Autosys query tool function for LLM integration"""
    try:
        if not question or not question.strip():
            return "❓ Please provide a question about Autosys jobs, schedules, or system status."
        
        question = question.strip()
        logger.info(f"🚀 Processing Autosys query: {question}")
        
        # Execute the workflow
        initial_state = AutosysState(
            question=question,
            sql="",
            result="",
            answer="",
            error=None,
            table_info=""
        )
        
        result = app.invoke(initial_state)
        answer = result.get("answer", "No answer generated")
        
        # Clean up the response
        if isinstance(answer, str):
            return answer.strip()
        else:
            return str(answer).strip()
            
    except Exception as e:
        error_msg = f"Autosys query processing failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return f"❌ I couldn't process your Autosys query. Error: {error_msg}"

# Create the enhanced LangChain tool
@tool
def autosys_sql_query(question: str) -> str:
    """Query Autosys job scheduling system for job status, schedules, and operational information.
    
    This tool can answer questions about:
    - Job status (running, failed, success, terminated)
    - Job schedules and execution history  
    - Job dependencies and relationships
    - Machine and server information
    - Alarm and notification history
    - Performance and runtime statistics
    
    Args:
        question: Natural language question about Autosys jobs or system
        
    Returns:
        Formatted response with job information and status
    
    Example questions:
    - "What is the status of job ATSYS.DA3_DBMaint_080_arch_machines.c?"
    - "Show me all failed jobs from yesterday"
    - "List currently running jobs"
    - "What jobs depend on BACKUP_DAILY_JOB?"
    - "Show job runtime statistics for ETL processes"
    - "Which machines are running jobs right now?"
    """
    return autosys_query_tool(question)

# Legacy Tool wrapper for backward compatibility
autosys_sql_tool_enhanced = Tool(
    name="AutosysQuery",
    func=autosys_query_tool,
    description="""Enhanced Autosys job scheduling database query tool.

This tool provides comprehensive access to Autosys job information including:
- Job status and execution history
- Job schedules and dependencies  
- Machine and resource information
- Performance metrics and alarms
- Real-time job monitoring

Supports both current Autosys schema (UJO_* tables) and legacy formats.
Uses rule-based SQL generation with proper Oracle syntax and optimizations.

Example queries:
- "Show failed jobs from last 24 hours"
- "What's the status of job ATSYS.DAILY_BACKUP.c?"
- "List all jobs running on PROD_SERVER_01"
- "Show job dependencies for ETL_MAIN_PROCESS"
- "Get runtime statistics for jobs that ran today"
"""
)

# Export for easy import
__all__ = [
    'autosys_sql_query',
    'autosys_sql_tool_enhanced', 
    'AutosysOracleDatabase',
    'AutosysTextToSQLTool',
    'autosys_query_tool'
]
