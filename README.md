
# ============================================================================
# MULTI-DATABASE AUTOSYS SYSTEM WITH INSTANCE SELECTION
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
# MULTI-DATABASE CONFIGURATION
# ============================================================================

class DatabaseInstance:
    """Represents a single database instance"""
    def __init__(self, instance_name: str, autosys_db, description: str = ""):
        self.instance_name = instance_name
        self.autosys_db = autosys_db
        self.description = description
        self.is_connected = self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test if database connection is available"""
        try:
            if hasattr(self.autosys_db, 'connection') and self.autosys_db.connection:
                return True
            return False
        except:
            return False

class DatabaseManager:
    """Manages multiple database instances"""
    def __init__(self):
        self.instances: Dict[str, DatabaseInstance] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_instance(self, instance_name: str, autosys_db, description: str = ""):
        """Add a database instance"""
        instance = DatabaseInstance(instance_name, autosys_db, description)
        self.instances[instance_name.upper()] = instance
        self.logger.info(f"Added database instance: {instance_name}")
        return instance
    
    def get_instance(self, instance_name: str) -> Optional[DatabaseInstance]:
        """Get database instance by name"""
        return self.instances.get(instance_name.upper())
    
    def list_instances(self) -> List[str]:
        """List all available instances"""
        return [name for name, instance in self.instances.items() if instance.is_connected]
    
    def get_instance_info(self) -> str:
        """Get formatted instance information"""
        if not self.instances:
            return "No database instances configured."
        
        info_lines = []
        for name, instance in self.instances.items():
            status = "✅ Connected" if instance.is_connected else "❌ Disconnected"
            desc = f" - {instance.description}" if instance.description else ""
            info_lines.append(f"• {name}: {status}{desc}")
        
        return "\n".join(info_lines)

# ============================================================================
# ENHANCED STATE FOR MULTI-DATABASE
# ============================================================================

class AutosysState(TypedDict):
    """Enhanced state for multi-database Autosys workflow"""
    messages: Annotated[List[Dict[str, str]], operator.add]
    user_question: str
    is_general_conversation: bool
    extracted_instance: str
    extracted_job_calendar: str
    needs_clarification: bool
    sql_query: str
    query_results: Dict[str, Any]
    formatted_output: str
    error: str
    session_id: str

# ============================================================================
# MESSAGE ROUTING AND EXTRACTION LOGIC
# ============================================================================

def is_autosys_related_query(message: str) -> bool:
    """Determine if the message requires Autosys database query"""
    
    # Check for casual inputs first
    casual_inputs = [
        "hi", "hello", "hey", "how are you", "good morning", "good evening",
        "what's up", "how's it going", "yo", "greetings", "thanks", "thank you"
    ]
    
    message_lower = message.lower().strip()
    
    if any(casual_word in message_lower for casual_word in casual_inputs):
        return False
    
    # Keywords that indicate database/job queries
    autosys_keywords = [
        'job', 'jobs', 'atsys', 'autosys', 'schedule', 'status', 'failed', 'failure',
        'running', 'success', 'database', 'query', 'select', 'show', 'list', 'find',
        'search', 'owner', 'machine', 'execution', 'sql', 'table', 'count', 'report',
        'calendar', 'instance'
    ]
    
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
# ENHANCED AUTOSYS TOOL WITH INSTANCE SUPPORT
# ============================================================================

class AutosysQueryInput(BaseModel):
    """Input schema for Autosys Query Tool"""
    question: str = Field(description="Natural language question about Autosys jobs")

class MultiDatabaseAutosysQueryTool(BaseTool):
    """Enhanced tool for querying multiple Autosys database instances"""
    
    name: str = "AutosysQuery"
    description: str = "Query Autosys job scheduler database across multiple instances"
    args_schema = AutosysQueryInput
    
    def __init__(self, db_manager: DatabaseManager, llm_instance, max_results: int = 50):
        super().__init__()
        self.db_manager = db_manager
        self.llm = llm_instance
        self.max_results = max_results
        self.logger = logging.getLogger(self.__class__.__name__)

    def _run(self, question: str, instance_name: str = "") -> Dict[str, Any]:
        """Execute database query with instance selection"""
        try:
            # If no instance specified, return available instances
            if not instance_name:
                return {
                    "success": False,
                    "needs_instance": True,
                    "available_instances": self.db_manager.list_instances(),
                    "error": "Please specify database instance"
                }
            
            # Get database instance
            instance = self.db_manager.get_instance(instance_name)
            if not instance:
                return {
                    "success": False,
                    "error": f"Database instance '{instance_name}' not found",
                    "available_instances": self.db_manager.list_instances()
                }
            
            # Generate SQL
            sql_query = self._generate_sql(question, instance_name)
            
            # Execute query  
            query_result = self._execute_query(sql_query, instance)
            
            return {
                "success": query_result["success"],
                "instance_used": instance_name,
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

    def _generate_sql(self, user_question: str, instance_name: str) -> str:
        """Generate SQL using LLM with intelligent parameter incorporation"""
        
        sql_prompt = f"""
Generate Oracle SQL for Autosys job scheduler database with intelligent parameter handling.

DATABASE INSTANCE: {instance_name}
USER QUERY: "{user_question}"

SCHEMA:
- aedbadmin.ujo_jobst: job_name, status (SU=Success, FA=Failure, RU=Running), last_start, last_end, joid
- aedbadmin.ujo_job: joid, owner, machine, job_type  
- aedbadmin.UJO_INTCODES: code, TEXT (status descriptions)

SQL PATTERNS:
- Time: TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss')
- Joins: js INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
- Status: LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
- Recent: WHERE js.last_start >= (SYSDATE - INTERVAL '1' DAY) * 86400

INTELLIGENT QUERY CONSTRUCTION:
- If user mentions specific job name, add: WHERE UPPER(js.job_name) LIKE UPPER('%job_name%')
- If user mentions calendar, search for calendar-related jobs: WHERE UPPER(js.job_name) LIKE UPPER('%calendar_name%')
- If user asks for failed jobs: WHERE js.status = 7 OR ic.TEXT = 'FAILURE'
- If user asks for running jobs: WHERE js.status = 8 OR ic.TEXT = 'RUNNING'  
- If user asks for successful jobs: WHERE js.status = 4 OR ic.TEXT = 'SUCCESS'
- If user mentions "today": WHERE js.last_start >= TRUNC(SYSDATE) * 86400
- If user mentions "yesterday": WHERE js.last_start >= (TRUNC(SYSDATE) - 1) * 86400 AND js.last_start < TRUNC(SYSDATE) * 86400

QUERY OPTIMIZATION:
- Always include execution time information when relevant
- Sort by relevance (status first, then time)
- Include owner information for job management context
- Add machine information for deployment details

Return only the optimized SQL query without explanations:
"""
        
        try:
            response = self.llm.invoke(sql_prompt)
            sql = response.content if hasattr(response, 'content') else str(response)
            return self._clean_sql(sql)
        except Exception as e:
            self.logger.error(f"Enhanced SQL generation failed: {e}")
            return self._fallback_sql_with_context(user_question)

    def _clean_sql(self, sql: str) -> str:
        """Clean and enhance SQL query"""
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        sql = ' '.join(sql.split()).strip()
        
        if 'ROWNUM' not in sql.upper():
            if 'WHERE' in sql.upper():
                sql = sql.replace('WHERE', f'WHERE ROWNUM <= {self.max_results} AND', 1)
            else:
                sql += f' WHERE ROWNUM <= {self.max_results}'
        
        return sql

    def _execute_query(self, sql_query: str, instance: DatabaseInstance) -> Dict[str, Any]:
        """Execute SQL query using specific database instance"""
        try:
            start_time = datetime.now()
            
            if hasattr(instance.autosys_db, 'run'):
                raw_results = instance.autosys_db.run(sql_query)
            elif hasattr(instance.autosys_db, 'execute_query'):
                raw_results = instance.autosys_db.execute_query(sql_query)
            else:
                raise Exception(f"Database connection method not found for instance {instance.instance_name}")
            
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

    def _fallback_sql_with_context(self, user_question: str) -> str:
        """Enhanced fallback SQL query with context awareness"""
        question_lower = user_question.lower()
        
        # Build WHERE conditions based on user intent
        conditions = []
        
        # Status-based conditions
        if any(word in question_lower for word in ['failed', 'failure', 'error']):
            conditions.append("(js.status = 7 OR ic.TEXT = 'FAILURE')")
        elif any(word in question_lower for word in ['running', 'active']):
            conditions.append("(js.status = 8 OR ic.TEXT = 'RUNNING')")
        elif any(word in question_lower for word in ['success', 'successful', 'completed']):
            conditions.append("(js.status = 4 OR ic.TEXT = 'SUCCESS')")
        
        # Time-based conditions
        if 'today' in question_lower:
            conditions.append("js.last_start >= TRUNC(SYSDATE) * 86400")
        elif 'yesterday' in question_lower:
            conditions.append("js.last_start >= (TRUNC(SYSDATE) - 1) * 86400 AND js.last_start < TRUNC(SYSDATE) * 86400")
        elif any(word in question_lower for word in ['recent', 'latest', 'last']):
            conditions.append("js.last_start >= (SYSDATE - INTERVAL '1' DAY) * 86400")
        
        # Default ATSYS filter if no specific job mentioned
        if not any(char in user_question for char in ['.', '_']) or 'atsys' in question_lower:
            conditions.append("UPPER(js.job_name) LIKE UPPER('%ATSYS%')")
        
        # Combine conditions
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        return f"""
        SELECT 
            js.job_name,
            TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (js.last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss') AS start_time,
            TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (js.last_end / 86400), 'MM/DD/YYYY HH24:Mi:Ss') AS end_time,
            NVL(ic.TEXT, 'UNKNOWN') AS status,
            j.owner,
            j.machine
        FROM aedbadmin.ujo_jobst js
        INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
        LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
        WHERE {where_clause}
        AND ROWNUM <= {self.max_results}
        ORDER BY js.last_start DESC
        """

    def request_clarification_node(self, state: AutosysState) -> AutosysState:
        """Enhanced clarification request with intelligent suggestions"""
        
        available_instances = self.db_manager.list_instances()
        instance_info = self.db_manager.get_instance_info()
        
        # Analyze what information is missing and provide targeted guidance
        extracted_job = state.get("extracted_job_calendar", "")
        user_question = state.get("user_question", "")
        
        # Create contextual clarification message
        clarification_parts = []
        
        if not state.get("extracted_instance"):
            clarification_parts.append("database instance")
        
        missing_info = " and ".join(clarification_parts) if clarification_parts else "additional information"
        
        # Generate smart suggestions based on the user's original query
        suggestion_examples = []
        base_query = user_question.lower()
        
        if 'failed' in base_query:
            suggestion_examples = [
                f"Show failed jobs in PROD instance",
                f"List failed jobs in DEV environment", 
                f"Check failed job ABC123 in TEST"
            ]
        elif 'running' in base_query:
            suggestion_examples = [
                f"Show running jobs in PROD",
                f"List active jobs in DEV instance",
                f"Check running job XYZ456 in PROD"
            ]
        elif extracted_job:
            suggestion_examples = [
                f"Show job {extracted_job} status in PROD",
                f"Check {extracted_job} in DEV instance",
                f"Get details for {extracted_job} in TEST environment"
            ]
        else:
            suggestion_examples = [
                f"Show job status in PROD instance",
                f"List failed jobs in DEV environment",
                f"Check job ABC123 in TEST"
            ]
        
        suggestions_html = "<br>".join([f"• {example}" for example in suggestion_examples])
        
        clarification_html = f"""
        <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #856404;">Please Specify {missing_info.title()}</h4>
            <p style="margin: 0 0 10px 0; color: #856404;">
                I need to know which database instance to query for: <em>"{user_question}"</em>
            </p>
            
            <div style="background: #f8f9fa; border-radius: 4px; padding: 10px; margin: 10px 0;">
                <strong>Available Instances:</strong><br>
                <div style="font-family: monospace; font-size: 12px; margin: 5px 0; white-space: pre-line;">{instance_info}</div>
            </div>
            
            <div style="background: #e7f3ff; border-radius: 4px; padding: 10px; margin: 10px 0;">
                <strong>Example queries:</strong><br>
                <div style="font-size: 13px; margin: 5px 0; line-height: 1.4;">{suggestions_html}</div>
            </div>
            
            <p style="margin: 10px 0 0 0; color: #856404; font-size: 12px;">
                <em>Just mention the instance name (PROD, DEV, TEST, etc.) in your next message!</em>
            </p>
        </div>
        """
        
        state["formatted_output"] = clarification_html
        return state

# ============================================================================
# ENHANCED LANGGRAPH WORKFLOW SYSTEM
# ============================================================================

class MultiDatabaseAutosysLangGraphSystem:
    """Complete LangGraph-based system for multi-database Autosys queries"""
    
    def __init__(self, db_manager: DatabaseManager, llm_instance):
        self.db_manager = db_manager
        self.llm = llm_instance
        self.tool = MultiDatabaseAutosysQueryTool(db_manager, llm_instance)
        self.memory = MemorySaver()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Build workflow
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the enhanced LangGraph workflow"""
        
        workflow = StateGraph(AutosysState)
        
        # Add nodes
        workflow.add_node("route_message", self.route_message_node)
        workflow.add_node("handle_conversation", self.handle_conversation_node)
        workflow.add_node("extract_parameters", self.extract_parameters_node)
        workflow.add_node("request_clarification", self.request_clarification_node)
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
                "database": "extract_parameters"
            }
        )
        
        workflow.add_edge("handle_conversation", END)
        
        workflow.add_conditional_edges(
            "extract_parameters",
            self._needs_clarification,
            {
                "clarification": "request_clarification",
                "proceed": "generate_sql"
            }
        )
        
        workflow.add_edge("request_clarification", END)
        
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
        
        message = state["user_question"].strip().lower()
        
        # Prioritize casual greetings
        casual_inputs = [
            "hi", "hello", "hey", "how are you", "good morning", "good evening",
            "what's up", "how's it going", "yo", "greetings"
        ]
        
        if any(casual_word in message for casual_word in casual_inputs):
            is_conversation = True
        else:
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

You can handle friendly conversation and database queries across multiple instances.

Available database instances: {', '.join(self.db_manager.list_instances())}

User message: "{state['user_question']}"

Respond in a friendly, professional manner. If they ask about capabilities, mention you can query multiple Autosys database instances.
"""
            
            response = self.llm.invoke(conversation_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            state["formatted_output"] = f"""
            <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                <div style="max-width: 70%; background: #e9ecef; border-radius: 18px; padding: 12px 16px; color: #212529; font-size: 14px; line-height: 1.4; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                    {content}
                </div>
            </div>
            """
            
        except Exception as e:
            state["formatted_output"] = f"""
            <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                <div style="max-width: 70%; background: #e9ecef; border-radius: 18px; padding: 12px 16px; color: #212529; font-size: 14px; line-height: 1.4;">
                    Hi! I can help with general questions and query multiple Autosys database instances. How can I assist you?
                </div>
            </div>
            """
        
        return state

    def extract_parameters_node(self, state: AutosysState) -> AutosysState:
        """Extract instance, job, and calendar information from user query using LLM"""
        
        try:
            available_instances = self.db_manager.list_instances()
            
            extraction_prompt = f"""
You are an expert at extracting Autosys parameters from user queries.

Available database instances: {', '.join(available_instances)}
User query: "{state['user_question']}"

Analyze the query and extract:
1. Database instance name (PROD, DEV, TEST, etc.)
2. Job name (specific job identifier)
3. Calendar name (calendar identifier)
4. Query intent (status check, job details, calendar info, etc.)

Be intelligent in extraction:
- Look for instance keywords: "prod", "production", "dev", "development", "test", "uat"
- Identify job patterns: job names often contain periods, underscores, or specific formats
- Recognize calendar references: "calendar", "cal", specific calendar names
- Understand variations: "job ABC123", "calendar XYZ", "PROD environment"

Format response as JSON:
{{
    "instance": "detected_instance_or_null",
    "job_name": "specific_job_name_or_null", 
    "calendar_name": "calendar_name_or_null",
    "query_intent": "status/details/schedule/list/etc",
    "confidence": "high/medium/low",
    "missing_info": ["list", "of", "missing", "required", "info"]
}}

Examples:
- "Show job ABC.DEF.123 status in PROD" → {{"instance": "PROD", "job_name": "ABC.DEF.123", "calendar_name": null, "query_intent": "status", "confidence": "high"}}
- "List failed jobs" → {{"instance": null, "job_name": null, "calendar_name": null, "query_intent": "list", "confidence": "medium", "missing_info": ["instance"]}}

Return only the JSON:
"""
            
            response = self.llm.invoke(extraction_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse extraction results
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    extraction_data = json.loads(json_match.group())
                    
                    # Store extracted information
                    state["extracted_instance"] = extraction_data.get("instance", "").upper() if extraction_data.get("instance") else ""
                    state["extracted_job_calendar"] = extraction_data.get("job_name", "") or extraction_data.get("calendar_name", "")
                    
                    # Additional extracted info for context
                    query_intent = extraction_data.get("query_intent", "")
                    confidence = extraction_data.get("confidence", "low")
                    missing_info = extraction_data.get("missing_info", [])
                    
                    # Log extraction details
                    self.logger.info(f"Extracted - Instance: {state['extracted_instance']}, Job/Calendar: {state['extracted_job_calendar']}, Intent: {query_intent}, Confidence: {confidence}")
                    
                    # Enhanced logic for clarification needs
                    needs_clarification = False
                    
                    # Check if instance is missing and we have multiple instances
                    if not state["extracted_instance"]:
                        if len(available_instances) > 1:
                            needs_clarification = True
                        elif len(available_instances) == 1:
                            # Auto-select single instance
                            state["extracted_instance"] = available_instances[0]
                            self.logger.info(f"Auto-selected single available instance: {state['extracted_instance']}")
                    
                    # Validate extracted instance exists
                    if state["extracted_instance"] and state["extracted_instance"] not in available_instances:
                        # Try fuzzy matching
                        matched_instance = self._fuzzy_match_instance(state["extracted_instance"], available_instances)
                        if matched_instance:
                            state["extracted_instance"] = matched_instance
                            self.logger.info(f"Fuzzy matched instance: {matched_instance}")
                        else:
                            needs_clarification = True
                            self.logger.warning(f"Instance {state['extracted_instance']} not found in available instances")
                    
                    state["needs_clarification"] = needs_clarification
                    
                else:
                    self.logger.warning("Could not parse JSON from extraction response")
                    state["extracted_instance"] = ""
                    state["extracted_job_calendar"] = ""
                    state["needs_clarification"] = len(available_instances) > 1
                    
            except Exception as parse_error:
                self.logger.error(f"JSON parsing failed: {parse_error}")
                state["extracted_instance"] = ""
                state["extracted_job_calendar"] = ""
                state["needs_clarification"] = len(available_instances) > 1
            
        except Exception as e:
            self.logger.error(f"Parameter extraction failed: {str(e)}")
            state["error"] = f"Parameter extraction failed: {str(e)}"
            state["needs_clarification"] = True
        
        return state
    
    def _fuzzy_match_instance(self, user_instance: str, available_instances: List[str]) -> Optional[str]:
        """Fuzzy match user input to available instances"""
        user_lower = user_instance.lower()
        
        # Direct mapping for common variations
        instance_mapping = {
            'prod': 'PROD', 'production': 'PROD', 'prd': 'PROD',
            'dev': 'DEV', 'development': 'DEV', 'devel': 'DEV',
            'test': 'TEST', 'testing': 'TEST', 'tst': 'TEST',
            'uat': 'UAT', 'user acceptance': 'UAT', 'acceptance': 'UAT',
            'stage': 'STAGE', 'staging': 'STAGE', 'stg': 'STAGE'
        }
        
        # Check direct mapping
        if user_lower in instance_mapping:
            mapped = instance_mapping[user_lower]
            if mapped in available_instances:
                return mapped
        
        # Check if user input is contained in or contains available instance
        for instance in available_instances:
            if user_lower in instance.lower() or instance.lower() in user_lower:
                return instance
        
        return None

    def request_clarification_node(self, state: AutosysState) -> AutosysState:
        """Request clarification for missing parameters"""
        
        available_instances = self.db_manager.list_instances()
        instance_info = self.db_manager.get_instance_info()
        
        clarification_html = f"""
        <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #856404;">Please Specify Database Instance</h4>
            <p style="margin: 0 0 10px 0; color: #856404;">I need to know which database instance to query.</p>
            
            <div style="background: #f8f9fa; border-radius: 4px; padding: 10px; margin: 10px 0;">
                <strong>Available Instances:</strong><br>
                <pre style="margin: 5px 0; font-size: 12px;">{instance_info}</pre>
            </div>
            
            <p style="margin: 10px 0 0 0; color: #856404; font-size: 13px;">
                <em>Please specify the instance name in your question (e.g., "Show failed jobs in PROD instance")</em>
            </p>
        </div>
        """
        
        state["formatted_output"] = clarification_html
        return state

    def generate_sql_node(self, state: AutosysState) -> AutosysState:
        """Generate SQL query using tool"""
        
        try:
            tool_result = self.tool.run(state["user_question"], state["extracted_instance"])
            
            if tool_result["success"]:
                state["sql_query"] = tool_result["sql_query"]
            elif tool_result.get("needs_instance"):
                state["needs_clarification"] = True
            else:
                state["error"] = f"SQL generation failed: {tool_result.get('error', 'Unknown error')}"
                
        except Exception as e:
            state["error"] = f"SQL generation exception: {str(e)}"
        
        return state

    def execute_query_node(self, state: AutosysState) -> AutosysState:
        """Execute database query"""
        
        try:
            tool_result = self.tool.run(state["user_question"], state["extracted_instance"])
            
            if tool_result["success"]:
                state["query_results"] = {
                    "success": True,
                    "results": tool_result["results"],
                    "row_count": tool_result["row_count"],
                    "execution_time": tool_result["execution_time"],
                    "instance_used": tool_result["instance_used"]
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
            instance_used = state["query_results"].get("instance_used", "Unknown")
            
            if not results:
                state["formatted_output"] = f"""
                <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px;">
                    <h4 style="margin: 0 0 10px 0; color: #856404;">No Results Found</h4>
                    <p style="margin: 0; color: #856404;">No Autosys jobs match your query criteria in instance: <strong>{instance_used}</strong></p>
                </div>
                """
                return state
            
            # Format using LLM
            format_prompt = f"""
Create professional HTML for these Autosys query results:

Question: {state['user_question']}
Database Instance: {instance_used}
Results: {len(results)} jobs found
Data: {json.dumps(results[:10], indent=2, default=str)}

Requirements:
- Responsive HTML with inline CSS
- Color-coded status badges (SUCCESS=green, FAILURE=red, RUNNING=blue, INACTIVE=gray)
- Professional styling with good contrast
- Include instance name prominently
- Mobile-friendly design
- Include summary statistics

Return only HTML:
"""
            
            response = self.llm.invoke(format_prompt)
            formatted_html = response.content if hasattr(response, 'content') else str(response)
            
            # Add metadata with instance info
            metadata = f"""
            <div style="margin-top: 15px; padding: 8px 12px; background: #f8f9fa; border-radius: 4px; font-size: 11px; color: #666; border-left: 3px solid #007bff;">
                <strong>AutosysQuery</strong> • Instance: <strong>{instance_used}</strong> • {state['query_results']['row_count']} results • {state['query_results']['execution_time']:.2f}s
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
        
        # Add available instances info
        instances = self.db_manager.list_instances()
        if instances:
            error_html += f"""
            <div style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 3px;">
                <strong>Available Instances:</strong> {', '.join(instances)}
            </div>
            """
        
        error_html += "</div>"
        state["formatted_output"] = error_html
        
        return state

    # Conditional edge functions
    def _should_handle_conversation(self, state: AutosysState) -> str:
        return "conversation" if state.get("is_general_conversation") else "database"

    def _needs_clarification(self, state: AutosysState) -> str:
        return "clarification" if state.get("needs_clarification") else "proceed"

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
            "extracted_instance": "",
            "extracted_job_calendar": "",
            "needs_clarification": False,
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
                "needs_clarification": final_state.get("needs_clarification", False),
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

def setup_autosys_multi_database_system(database_configs: Dict[str, Any], llm_instance):
    """
    Setup the complete multi-database Autosys LangGraph system
    
    Args:
        database_configs: Dictionary of database configurations
        {
            "PROD": {
                "autosys_db": AutosysOracleDatabase(prod_uri),
                "description": "Production environment"
            },
            "DEV": {
                "autosys_db": AutosysOracleDatabase(dev_uri), 
                "description": "Development environment"
            },
            "TEST": {
                "autosys_db": AutosysOracleDatabase(test_uri),
                "description": "Testing environment"
            }
        }
        llm_instance: Your LLM instance from get_llm("langchain")
    
    Returns:
        Configured multi-database system ready for use
    """
    global _autosys_system
    
    try:
        # Create database manager
        db_manager = DatabaseManager()
        
        # Add all database instances
        for instance_name, config in database_configs.items():
            autosys_db = config["autosys_db"]
            description = config.get("description", "")
            db_manager.add_instance(instance_name, autosys_db, description)
            logger.info(f"Added instance {instance_name}: {description}")
        
        # Initialize the multi-database system
        _autosys_system = MultiDatabaseAutosysLangGraphSystem(db_manager, llm_instance)
        
        logger.info("Multi-database Autosys LangGraph system initialized successfully")
        
        return {
            "status": "ready",
            "instances": db_manager.list_instances(),
            "features": [
                "Smart message routing (conversation vs database)",
                "Multi-database instance selection", 
                "Parameter extraction with LLM",
                "Automatic clarification requests",
                "LLM-powered SQL generation",
                "Professional HTML formatting",
                "Session persistence with memory",
                "Comprehensive error handling"
            ]
        }
        
    except Exception as e:
        logger.error(f"Multi-database system setup failed: {e}")
        raise Exception(f"Failed to initialize multi-database system: {str(e)}")

def get_chat_response(message: str, session_id: str) -> str:
    """Main chat function - handles conversation, instance selection, and database queries"""
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
                <p>The multi-database system is not initialized. Please contact administrator.</p>
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

def get_database_instances() -> Dict[str, Any]:
    """Get information about available database instances"""
    global _autosys_system
    
    if not _autosys_system:
        return {"error": "System not initialized"}
    
    try:
        return {
            "instances": _autosys_system.db_manager.list_instances(),
            "detailed_info": _autosys_system.db_manager.get_instance_info(),
            "total_count": len(_autosys_system.db_manager.instances)
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# USAGE EXAMPLES AND INTEGRATION
# ============================================================================

def example_multi_database_setup():
    """Example showing how to set up multiple database instances"""
    
    # Example database configurations
    database_configs = {
        "PROD": {
            "autosys_db": "AutosysOracleDatabase('oracle+oracledb://user:pass@prod-host:1521/prod_service')",
            "description": "Production Autosys environment"
        },
        "DEV": {
            "autosys_db": "AutosysOracleDatabase('oracle+oracledb://user:pass@dev-host:1521/dev_service')",
            "description": "Development Autosys environment"  
        },
        "TEST": {
            "autosys_db": "AutosysOracleDatabase('oracle+oracledb://user:pass@test-host:1521/test_service')",
            "description": "Testing Autosys environment"
        },
        "UAT": {
            "autosys_db": "AutosysOracleDatabase('oracle+oracledb://user:pass@uat-host:1521/uat_service')",
            "description": "User Acceptance Testing environment"
        }
    }
    
    # Your existing LLM setup
    # llm = get_llm("langchain")
    
    # Initialize multi-database system
    # setup_autosys_multi_database_system(database_configs, llm)
    
    # Test queries - the system will handle instance selection automatically
    test_queries = [
        "Hello!",  # → General conversation
        "Show me failed jobs",  # → Will ask for instance clarification
        "List running jobs in PROD instance",  # → Direct to PROD
        "Check job status in DEV for job ABC123",  # → Direct to DEV with job filter
        "What instances are available?"  # → System information
    ]
    
    return "Setup example ready"

def main():
    """Example usage and testing"""
    
    print("Multi-Database Autosys LangGraph System")
    print("=====================================")
    print()
    print("Features:")
    print("- Multiple database instance support")
    print("- Smart parameter extraction")
    print("- Automatic instance selection")
    print("- Conversation and database query routing")
    print("- Session memory and context")
    print()
    
    # Example conversations:
    example_conversations = [
        {
            "user": "Hi there!",
            "expected": "General conversation response"
        },
        {
            "user": "Show me failed jobs",  
            "expected": "Request for instance clarification"
        },
        {
            "user": "List failed jobs in PROD instance",
            "expected": "SQL query against PROD database"
        },
        {
            "user": "Check job XYZ123 status in DEV",
            "expected": "Job-specific query in DEV instance" 
        }
    ]
    
    print("Example conversation flows:")
    for i, conv in enumerate(example_conversations, 1):
        print(f"{i}. User: '{conv['user']}'")
        print(f"   Expected: {conv['expected']}")
        print()

if __name__ == "__main__":
    main()

# ============================================================================
# COMPLETE INTEGRATION INSTRUCTIONS
# ============================================================================

"""
MULTI-DATABASE INTEGRATION GUIDE:

1. SETUP YOUR DATABASE CONFIGURATIONS:

database_configs = {
    "PROD": {
        "autosys_db": AutosysOracleDatabase("oracle+oracledb://user:pass@prod:1521/service"),
        "description": "Production environment"
    },
    "DEV": {
        "autosys_db": AutosysOracleDatabase("oracle+oracledb://user:pass@dev:1521/service"), 
        "description": "Development environment"
    },
    "TEST": {
        "autosys_db": AutosysOracleDatabase("oracle+oracledb://user:pass@test:1521/service"),
        "description": "Testing environment"
    }
}

llm = get_llm("langchain")

2. INITIALIZE THE SYSTEM:

setup_autosys_multi_database_system(database_configs, llm)

3. USE THE SAME INTERFACE:

response = get_chat_response(user_message, session_id)

4. QUERY EXAMPLES:

- "Hello!" → General conversation
- "Show me failed jobs" → System asks for instance clarification  
- "List failed jobs in PROD" → Queries PROD database
- "Check job ABC123 in DEV instance" → Specific job query in DEV
- "Show running jobs in TEST environment" → Queries TEST database

5. AUTOMATIC FEATURES:

✓ Instance detection from user messages
✓ Parameter extraction (job names, calendars)
✓ Clarification requests when needed
✓ Smart routing (conversation vs database)
✓ Session memory for context
✓ Professional HTML formatting
✓ Multi-database error handling

6. NO CHANGES NEEDED TO YOUR EXISTING:

✓ API endpoints
✓ Session handling  
✓ Client-side code
✓ get_chat_response() function signature

The system automatically handles all the complexity behind the scenes!
"""


#######@@@@@@@@@@@@@#######
        # ============================================================================
# MULTI-DATABASE AUTOSYS SYSTEM WITH INSTANCE SELECTION
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
# MULTI-DATABASE CONFIGURATION
# ============================================================================

class DatabaseInstance:
    """Represents a single database instance"""
    def __init__(self, instance_name: str, autosys_db, description: str = ""):
        self.instance_name = instance_name
        self.autosys_db = autosys_db
        self.description = description
        self.is_connected = self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test if database connection is available"""
        try:
            if hasattr(self.autosys_db, 'connection') and self.autosys_db.connection:
                return True
            return False
        except:
            return False

class DatabaseManager:
    """Manages multiple database instances"""
    def __init__(self):
        self.instances: Dict[str, DatabaseInstance] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_instance(self, instance_name: str, autosys_db, description: str = ""):
        """Add a database instance"""
        instance = DatabaseInstance(instance_name, autosys_db, description)
        self.instances[instance_name.upper()] = instance
        self.logger.info(f"Added database instance: {instance_name}")
        return instance
    
    def get_instance(self, instance_name: str) -> Optional[DatabaseInstance]:
        """Get database instance by name"""
        return self.instances.get(instance_name.upper())
    
    def list_instances(self) -> List[str]:
        """List all available instances"""
        return [name for name, instance in self.instances.items() if instance.is_connected]
    
    def get_instance_info(self) -> str:
        """Get formatted instance information"""
        if not self.instances:
            return "No database instances configured."
        
        info_lines = []
        for name, instance in self.instances.items():
            status = "✅ Connected" if instance.is_connected else "❌ Disconnected"
            desc = f" - {instance.description}" if instance.description else ""
            info_lines.append(f"• {name}: {status}{desc}")
        
        return "\n".join(info_lines)

# ============================================================================
# ENHANCED STATE FOR MULTI-DATABASE
# ============================================================================

class AutosysState(TypedDict):
    """Enhanced state for multi-database Autosys workflow"""
    messages: Annotated[List[Dict[str, str]], operator.add]
    user_question: str
    is_general_conversation: bool
    extracted_instance: str
    extracted_job_calendar: str
    needs_clarification: bool
    sql_query: str
    query_results: Dict[str, Any]
    formatted_output: str
    error: str
    session_id: str

# ============================================================================
# MESSAGE ROUTING AND EXTRACTION LOGIC
# ============================================================================

def is_autosys_related_query(message: str) -> bool:
    """Determine if the message requires Autosys database query"""
    
    # Check for casual inputs first
    casual_inputs = [
        "hi", "hello", "hey", "how are you", "good morning", "good evening",
        "what's up", "how's it going", "yo", "greetings", "thanks", "thank you"
    ]
    
    message_lower = message.lower().strip()
    
    if any(casual_word in message_lower for casual_word in casual_inputs):
        return False
    
    # Keywords that indicate database/job queries
    autosys_keywords = [
        'job', 'jobs', 'atsys', 'autosys', 'schedule', 'status', 'failed', 'failure',
        'running', 'success', 'database', 'query', 'select', 'show', 'list', 'find',
        'search', 'owner', 'machine', 'execution', 'sql', 'table', 'count', 'report',
        'calendar', 'instance'
    ]
    
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
# ENHANCED AUTOSYS TOOL WITH INSTANCE SUPPORT
# ============================================================================

class AutosysQueryInput(BaseModel):
    """Input schema for Autosys Query Tool"""
    question: str = Field(description="Natural language question about Autosys jobs")

class MultiDatabaseAutosysQueryTool(BaseTool):
    """Enhanced tool for querying multiple Autosys database instances"""
    
    name: str = "AutosysQuery"
    description: str = "Query Autosys job scheduler database across multiple instances"
    args_schema = AutosysQueryInput
    
    def __init__(self, db_manager: DatabaseManager, llm_instance, max_results: int = 50):
        super().__init__()
        self.db_manager = db_manager
        self.llm = llm_instance
        self.max_results = max_results
        self.logger = logging.getLogger(self.__class__.__name__)

    def _run(self, question: str, instance_name: str = "") -> Dict[str, Any]:
        """Execute database query with instance selection"""
        try:
            # If no instance specified, return available instances
            if not instance_name:
                return {
                    "success": False,
                    "needs_instance": True,
                    "available_instances": self.db_manager.list_instances(),
                    "error": "Please specify database instance"
                }
            
            # Get database instance
            instance = self.db_manager.get_instance(instance_name)
            if not instance:
                return {
                    "success": False,
                    "error": f"Database instance '{instance_name}' not found",
                    "available_instances": self.db_manager.list_instances()
                }
            
            # Generate SQL
            sql_query = self._generate_sql(question, instance_name)
            
            # Execute query  
            query_result = self._execute_query(sql_query, instance)
            
            return {
                "success": query_result["success"],
                "instance_used": instance_name,
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

    def _generate_sql(self, user_question: str, instance_name: str) -> str:
        """Generate SQL using LLM with intelligent parameter incorporation"""
        
        sql_prompt = f"""
Generate Oracle SQL for Autosys job scheduler database with intelligent parameter handling.

DATABASE INSTANCE: {instance_name}
USER QUERY: "{user_question}"

SCHEMA:
- aedbadmin.ujo_jobst: job_name, status (SU=Success, FA=Failure, RU=Running), last_start, last_end, joid
- aedbadmin.ujo_job: joid, owner, machine, job_type  
- aedbadmin.UJO_INTCODES: code, TEXT (status descriptions)

SQL PATTERNS:
- Time: TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss')
- Joins: js INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
- Status: LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
- Recent: WHERE js.last_start >= (SYSDATE - INTERVAL '1' DAY) * 86400

INTELLIGENT QUERY CONSTRUCTION:
- If user mentions specific job name, add: WHERE UPPER(js.job_name) LIKE UPPER('%job_name%')
- If user mentions calendar, search for calendar-related jobs: WHERE UPPER(js.job_name) LIKE UPPER('%calendar_name%')
- If user asks for failed jobs: WHERE js.status = 7 OR ic.TEXT = 'FAILURE'
- If user asks for running jobs: WHERE js.status = 8 OR ic.TEXT = 'RUNNING'  
- If user asks for successful jobs: WHERE js.status = 4 OR ic.TEXT = 'SUCCESS'
- If user mentions "today": WHERE js.last_start >= TRUNC(SYSDATE) * 86400
- If user mentions "yesterday": WHERE js.last_start >= (TRUNC(SYSDATE) - 1) * 86400 AND js.last_start < TRUNC(SYSDATE) * 86400

QUERY OPTIMIZATION:
- Always include execution time information when relevant
- Sort by relevance (status first, then time)
- Include owner information for job management context
- Add machine information for deployment details

Return only the optimized SQL query without explanations:
"""
        
        try:
            response = self.llm.invoke(sql_prompt)
            sql = response.content if hasattr(response, 'content') else str(response)
            return self._clean_sql(sql)
        except Exception as e:
            self.logger.error(f"Enhanced SQL generation failed: {e}")
            return self._fallback_sql_with_context(user_question)

    def _clean_sql(self, sql: str) -> str:
        """Clean and enhance SQL query"""
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        sql = ' '.join(sql.split()).strip()
        
        if 'ROWNUM' not in sql.upper():
            if 'WHERE' in sql.upper():
                sql = sql.replace('WHERE', f'WHERE ROWNUM <= {self.max_results} AND', 1)
            else:
                sql += f' WHERE ROWNUM <= {self.max_results}'
        
        return sql

    def _execute_query(self, sql_query: str, instance: DatabaseInstance) -> Dict[str, Any]:
        """Execute SQL query using specific database instance"""
        try:
            start_time = datetime.now()
            
            if hasattr(instance.autosys_db, 'run'):
                raw_results = instance.autosys_db.run(sql_query)
            elif hasattr(instance.autosys_db, 'execute_query'):
                raw_results = instance.autosys_db.execute_query(sql_query)
            else:
                raise Exception(f"Database connection method not found for instance {instance.instance_name}")
            
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

    def _fallback_sql_with_context(self, user_question: str) -> str:
        """Enhanced fallback SQL query with context awareness"""
        question_lower = user_question.lower()
        
        # Build WHERE conditions based on user intent
        conditions = []
        
        # Status-based conditions
        if any(word in question_lower for word in ['failed', 'failure', 'error']):
            conditions.append("(js.status = 7 OR ic.TEXT = 'FAILURE')")
        elif any(word in question_lower for word in ['running', 'active']):
            conditions.append("(js.status = 8 OR ic.TEXT = 'RUNNING')")
        elif any(word in question_lower for word in ['success', 'successful', 'completed']):
            conditions.append("(js.status = 4 OR ic.TEXT = 'SUCCESS')")
        
        # Time-based conditions
        if 'today' in question_lower:
            conditions.append("js.last_start >= TRUNC(SYSDATE) * 86400")
        elif 'yesterday' in question_lower:
            conditions.append("js.last_start >= (TRUNC(SYSDATE) - 1) * 86400 AND js.last_start < TRUNC(SYSDATE) * 86400")
        elif any(word in question_lower for word in ['recent', 'latest', 'last']):
            conditions.append("js.last_start >= (SYSDATE - INTERVAL '1' DAY) * 86400")
        
        # Default ATSYS filter if no specific job mentioned
        if not any(char in user_question for char in ['.', '_']) or 'atsys' in question_lower:
            conditions.append("UPPER(js.job_name) LIKE UPPER('%ATSYS%')")
        
        # Combine conditions
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        return f"""
        SELECT 
            js.job_name,
            TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (js.last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss') AS start_time,
            TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (js.last_end / 86400), 'MM/DD/YYYY HH24:Mi:Ss') AS end_time,
            NVL(ic.TEXT, 'UNKNOWN') AS status,
            j.owner,
            j.machine
        FROM aedbadmin.ujo_jobst js
        INNER JOIN aedbadmin.ujo_job j ON j.joid = js.joid
        LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
        WHERE {where_clause}
        AND ROWNUM <= {self.max_results}
        ORDER BY js.last_start DESC
        """

    def request_clarification_node(self, state: AutosysState) -> AutosysState:
        """Enhanced clarification request with intelligent suggestions"""
        
        available_instances = self.db_manager.list_instances()
        instance_info = self.db_manager.get_instance_info()
        
        # Analyze what information is missing and provide targeted guidance
        extracted_job = state.get("extracted_job_calendar", "")
        user_question = state.get("user_question", "")
        
        # Create contextual clarification message
        clarification_parts = []
        
        if not state.get("extracted_instance"):
            clarification_parts.append("database instance")
        
        missing_info = " and ".join(clarification_parts) if clarification_parts else "additional information"
        
        # Generate smart suggestions based on the user's original query
        suggestion_examples = []
        base_query = user_question.lower()
        
        if 'failed' in base_query:
            suggestion_examples = [
                f"Show failed jobs in PROD instance",
                f"List failed jobs in DEV environment", 
                f"Check failed job ABC123 in TEST"
            ]
        elif 'running' in base_query:
            suggestion_examples = [
                f"Show running jobs in PROD",
                f"List active jobs in DEV instance",
                f"Check running job XYZ456 in PROD"
            ]
        elif extracted_job:
            suggestion_examples = [
                f"Show job {extracted_job} status in PROD",
                f"Check {extracted_job} in DEV instance",
                f"Get details for {extracted_job} in TEST environment"
            ]
        else:
            suggestion_examples = [
                f"Show job status in PROD instance",
                f"List failed jobs in DEV environment",
                f"Check job ABC123 in TEST"
            ]
        
        suggestions_html = "<br>".join([f"• {example}" for example in suggestion_examples])
        
        clarification_html = f"""
        <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #856404;">Please Specify {missing_info.title()}</h4>
            <p style="margin: 0 0 10px 0; color: #856404;">
                I need to know which database instance to query for: <em>"{user_question}"</em>
            </p>
            
            <div style="background: #f8f9fa; border-radius: 4px; padding: 10px; margin: 10px 0;">
                <strong>Available Instances:</strong><br>
                <div style="font-family: monospace; font-size: 12px; margin: 5px 0; white-space: pre-line;">{instance_info}</div>
            </div>
            
            <div style="background: #e7f3ff; border-radius: 4px; padding: 10px; margin: 10px 0;">
                <strong>Example queries:</strong><br>
                <div style="font-size: 13px; margin: 5px 0; line-height: 1.4;">{suggestions_html}</div>
            </div>
            
            <p style="margin: 10px 0 0 0; color: #856404; font-size: 12px;">
                <em>Just mention the instance name (PROD, DEV, TEST, etc.) in your next message!</em>
            </p>
        </div>
        """
        
        state["formatted_output"] = clarification_html
        return state

# ============================================================================
# ENHANCED LANGGRAPH WORKFLOW SYSTEM
# ============================================================================

class MultiDatabaseAutosysLangGraphSystem:
    """Complete LangGraph-based system for multi-database Autosys queries"""
    
    def __init__(self, db_manager: DatabaseManager, llm_instance):
        self.db_manager = db_manager
        self.llm = llm_instance
        self.tool = MultiDatabaseAutosysQueryTool(db_manager, llm_instance)
        self.memory = MemorySaver()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Build workflow
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the enhanced LangGraph workflow"""
        
        workflow = StateGraph(AutosysState)
        
        # Add nodes
        workflow.add_node("route_message", self.route_message_node)
        workflow.add_node("handle_conversation", self.handle_conversation_node)
        workflow.add_node("extract_parameters", self.extract_parameters_node)
        workflow.add_node("request_clarification", self.request_clarification_node)
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
                "database": "extract_parameters"
            }
        )
        
        workflow.add_edge("handle_conversation", END)
        
        workflow.add_conditional_edges(
            "extract_parameters",
            self._needs_clarification,
            {
                "clarification": "request_clarification",
                "proceed": "generate_sql"
            }
        )
        
        workflow.add_edge("request_clarification", END)
        
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
        
        message = state["user_question"].strip().lower()
        
        # Prioritize casual greetings
        casual_inputs = [
            "hi", "hello", "hey", "how are you", "good morning", "good evening",
            "what's up", "how's it going", "yo", "greetings"
        ]
        
        if any(casual_word in message for casual_word in casual_inputs):
            is_conversation = True
        else:
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

You can handle friendly conversation and database queries across multiple instances.

Available database instances: {', '.join(self.db_manager.list_instances())}

User message: "{state['user_question']}"

Respond in a friendly, professional manner. If they ask about capabilities, mention you can query multiple Autosys database instances.
"""
            
            response = self.llm.invoke(conversation_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            state["formatted_output"] = f"""
            <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                <div style="max-width: 70%; background: #e9ecef; border-radius: 18px; padding: 12px 16px; color: #212529; font-size: 14px; line-height: 1.4; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                    {content}
                </div>
            </div>
            """
            
        except Exception as e:
            state["formatted_output"] = f"""
            <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                <div style="max-width: 70%; background: #e9ecef; border-radius: 18px; padding: 12px 16px; color: #212529; font-size: 14px; line-height: 1.4;">
                    Hi! I can help with general questions and query multiple Autosys database instances. How can I assist you?
                </div>
            </div>
            """
        
        return state

    def extract_parameters_node(self, state: AutosysState) -> AutosysState:
        """Extract instance, job, and calendar information from user query using LLM"""
        
        try:
            available_instances = self.db_manager.list_instances()
            
            extraction_prompt = f"""
You are an expert at extracting Autosys parameters from user queries.

Available database instances: {', '.join(available_instances)}
User query: "{state['user_question']}"

Analyze the query and extract:
1. Database instance name (PROD, DEV, TEST, etc.)
2. Job name (specific job identifier)
3. Calendar name (calendar identifier)
4. Query intent (status check, job details, calendar info, etc.)

Be intelligent in extraction:
- Look for instance keywords: "prod", "production", "dev", "development", "test", "uat"
- Identify job patterns: job names often contain periods, underscores, or specific formats
- Recognize calendar references: "calendar", "cal", specific calendar names
- Understand variations: "job ABC123", "calendar XYZ", "PROD environment"

Format response as JSON:
{{
    "instance": "detected_instance_or_null",
    "job_name": "specific_job_name_or_null", 
    "calendar_name": "calendar_name_or_null",
    "query_intent": "status/details/schedule/list/etc",
    "confidence": "high/medium/low",
    "missing_info": ["list", "of", "missing", "required", "info"]
}}

Examples:
- "Show job ABC.DEF.123 status in PROD" → {{"instance": "PROD", "job_name": "ABC.DEF.123", "calendar_name": null, "query_intent": "status", "confidence": "high"}}
- "List failed jobs" → {{"instance": null, "job_name": null, "calendar_name": null, "query_intent": "list", "confidence": "medium", "missing_info": ["instance"]}}

Return only the JSON:
"""
            
            response = self.llm.invoke(extraction_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse extraction results
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    extraction_data = json.loads(json_match.group())
                    
                    # Store extracted information
                    state["extracted_instance"] = extraction_data.get("instance", "").upper() if extraction_data.get("instance") else ""
                    state["extracted_job_calendar"] = extraction_data.get("job_name", "") or extraction_data.get("calendar_name", "")
                    
                    # Additional extracted info for context
                    query_intent = extraction_data.get("query_intent", "")
                    confidence = extraction_data.get("confidence", "low")
                    missing_info = extraction_data.get("missing_info", [])
                    
                    # Log extraction details
                    self.logger.info(f"Extracted - Instance: {state['extracted_instance']}, Job/Calendar: {state['extracted_job_calendar']}, Intent: {query_intent}, Confidence: {confidence}")
                    
                    # Enhanced logic for clarification needs
                    needs_clarification = False
                    
                    # Check if instance is missing and we have multiple instances
                    if not state["extracted_instance"]:
                        if len(available_instances) > 1:
                            needs_clarification = True
                        elif len(available_instances) == 1:
                            # Auto-select single instance
                            state["extracted_instance"] = available_instances[0]
                            self.logger.info(f"Auto-selected single available instance: {state['extracted_instance']}")
                    
                    # Validate extracted instance exists
                    if state["extracted_instance"] and state["extracted_instance"] not in available_instances:
                        # Try fuzzy matching
                        matched_instance = self._fuzzy_match_instance(state["extracted_instance"], available_instances)
                        if matched_instance:
                            state["extracted_instance"] = matched_instance
                            self.logger.info(f"Fuzzy matched instance: {matched_instance}")
                        else:
                            needs_clarification = True
                            self.logger.warning(f"Instance {state['extracted_instance']} not found in available instances")
                    
                    state["needs_clarification"] = needs_clarification
                    
                else:
                    self.logger.warning("Could not parse JSON from extraction response")
                    state["extracted_instance"] = ""
                    state["extracted_job_calendar"] = ""
                    state["needs_clarification"] = len(available_instances) > 1
                    
            except Exception as parse_error:
                self.logger.error(f"JSON parsing failed: {parse_error}")
                state["extracted_instance"] = ""
                state["extracted_job_calendar"] = ""
                state["needs_clarification"] = len(available_instances) > 1
            
        except Exception as e:
            self.logger.error(f"Parameter extraction failed: {str(e)}")
            state["error"] = f"Parameter extraction failed: {str(e)}"
            state["needs_clarification"] = True
        
        return state
    
    def _fuzzy_match_instance(self, user_instance: str, available_instances: List[str]) -> Optional[str]:
        """Fuzzy match user input to available instances"""
        user_lower = user_instance.lower()
        
        # Direct mapping for common variations
        instance_mapping = {
            'prod': 'PROD', 'production': 'PROD', 'prd': 'PROD',
            'dev': 'DEV', 'development': 'DEV', 'devel': 'DEV',
            'test': 'TEST', 'testing': 'TEST', 'tst': 'TEST',
            'uat': 'UAT', 'user acceptance': 'UAT', 'acceptance': 'UAT',
            'stage': 'STAGE', 'staging': 'STAGE', 'stg': 'STAGE'
        }
        
        # Check direct mapping
        if user_lower in instance_mapping:
            mapped = instance_mapping[user_lower]
            if mapped in available_instances:
                return mapped
        
        # Check if user input is contained in or contains available instance
        for instance in available_instances:
            if user_lower in instance.lower() or instance.lower() in user_lower:
                return instance
        
        return None

    def request_clarification_node(self, state: AutosysState) -> AutosysState:
        """Request clarification for missing parameters"""
        
        available_instances = self.db_manager.list_instances()
        instance_info = self.db_manager.get_instance_info()
        
        clarification_html = f"""
        <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #856404;">Please Specify Database Instance</h4>
            <p style="margin: 0 0 10px 0; color: #856404;">I need to know which database instance to query.</p>
            
            <div style="background: #f8f9fa; border-radius: 4px; padding: 10px; margin: 10px 0;">
                <strong>Available Instances:</strong><br>
                <pre style="margin: 5px 0; font-size: 12px;">{instance_info}</pre>
            </div>
            
            <p style="margin: 10px 0 0 0; color: #856404; font-size: 13px;">
                <em>Please specify the instance name in your question (e.g., "Show failed jobs in PROD instance")</em>
            </p>
        </div>
        """
        
        state["formatted_output"] = clarification_html
        return state

    def generate_sql_node(self, state: AutosysState) -> AutosysState:
        """Generate SQL query using tool"""
        
        try:
            tool_result = self.tool.run(state["user_question"], state["extracted_instance"])
            
            if tool_result["success"]:
                state["sql_query"] = tool_result["sql_query"]
            elif tool_result.get("needs_instance"):
                state["needs_clarification"] = True
            else:
                state["error"] = f"SQL generation failed: {tool_result.get('error', 'Unknown error')}"
                
        except Exception as e:
            state["error"] = f"SQL generation exception: {str(e)}"
        
        return state

    def execute_query_node(self, state: AutosysState) -> AutosysState:
        """Execute database query"""
        
        try:
            tool_result = self.tool.run(state["user_question"], state["extracted_instance"])
            
            if tool_result["success"]:
                state["query_results"] = {
                    "success": True,
                    "results": tool_result["results"],
                    "row_count": tool_result["row_count"],
                    "execution_time": tool_result["execution_time"],
                    "instance_used": tool_result["instance_used"]
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
            instance_used = state["query_results"].get("instance_used", "Unknown")
            
            if not results:
                state["formatted_output"] = f"""
                <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px;">
                    <h4 style="margin: 0 0 10px 0; color: #856404;">No Results Found</h4>
                    <p style="margin: 0; color: #856404;">No Autosys jobs match your query criteria in instance: <strong>{instance_used}</strong></p>
                </div>
                """
                return state
            
            # Format using LLM
            format_prompt = f"""
Create professional HTML for these Autosys query results:

Question: {state['user_question']}
Database Instance: {instance_used}
Results: {len(results)} jobs found
Data: {json.dumps(results[:10], indent=2, default=str)}

Requirements:
- Responsive HTML with inline CSS
- Color-coded status badges (SUCCESS=green, FAILURE=red, RUNNING=blue, INACTIVE=gray)
- Professional styling with good contrast
- Include instance name prominently
- Mobile-friendly design
- Include summary statistics

Return only HTML:
"""
            
            response = self.llm.invoke(format_prompt)
            formatted_html = response.content if hasattr(response, 'content') else str(response)
            
            # Add metadata with instance info
            metadata = f"""
            <div style="margin-top: 15px; padding: 8px 12px; background: #f8f9fa; border-radius: 4px; font-size: 11px; color: #666; border-left: 3px solid #007bff;">
                <strong>AutosysQuery</strong> • Instance: <strong>{instance_used}</strong> • {state['query_results']['row_count']} results • {state['query_results']['execution_time']:.2f}s
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
        
        # Add available instances info
        instances = self.db_manager.list_instances()
        if instances:
            error_html += f"""
            <div style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 3px;">
                <strong>Available Instances:</strong> {', '.join(instances)}
            </div>
            """
        
        error_html += "</div>"
        state["formatted_output"] = error_html
        
        return state

    # Conditional edge functions
    def _should_handle_conversation(self, state: AutosysState) -> str:
        return "conversation" if state.get("is_general_conversation") else "database"

    def _needs_clarification(self, state: AutosysState) -> str:
        return "clarification" if state.get("needs_clarification") else "proceed"

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
            "extracted_instance": "",
            "extracted_job_calendar": "",
            "needs_clarification": False,
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
                "needs_clarification": final_state.get("needs_clarification", False),
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

def setup_autosys_multi_database_system(database_configs: Dict[str, Any], llm_instance):
    """
    Setup the complete multi-database Autosys LangGraph system
    
    Args:
        database_configs: Dictionary of database configurations
        {
            "PROD": {
                "autosys_db": AutosysOracleDatabase(prod_uri),
                "description": "Production environment"
            },
            "DEV": {
                "autosys_db": AutosysOracleDatabase(dev_uri), 
                "description": "Development environment"
            },
            "TEST": {
                "autosys_db": AutosysOracleDatabase(test_uri),
                "description": "Testing environment"
            }
        }
        llm_instance: Your LLM instance from get_llm("langchain")
    
    Returns:
        Configured multi-database system ready for use
    """
    global _autosys_system
    
    try:
        # Create database manager
        db_manager = DatabaseManager()
        
        # Add all database instances
        for instance_name, config in database_configs.items():
            autosys_db = config["autosys_db"]
            description = config.get("description", "")
            db_manager.add_instance(instance_name, autosys_db, description)
            logger.info(f"Added instance {instance_name}: {description}")
        
        # Initialize the multi-database system
        _autosys_system = MultiDatabaseAutosysLangGraphSystem(db_manager, llm_instance)
        
        logger.info("Multi-database Autosys LangGraph system initialized successfully")
        
        return {
            "status": "ready",
            "instances": db_manager.list_instances(),
            "features": [
                "Smart message routing (conversation vs database)",
                "Multi-database instance selection", 
                "Parameter extraction with LLM",
                "Automatic clarification requests",
                "LLM-powered SQL generation",
                "Professional HTML formatting",
                "Session persistence with memory",
                "Comprehensive error handling"
            ]
        }
        
    except Exception as e:
        logger.error(f"Multi-database system setup failed: {e}")
        raise Exception(f"Failed to initialize multi-database system: {str(e)}")

def get_chat_response(message: str, session_id: str) -> str:
    """Main chat function - handles conversation, instance selection, and database queries"""
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
                <p>The multi-database system is not initialized. Please contact administrator.</p>
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

def get_database_instances() -> Dict[str, Any]:
    """Get information about available database instances"""
    global _autosys_system
    
    if not _autosys_system:
        return {"error": "System not initialized"}
    
    try:
        return {
            "instances": _autosys_system.db_manager.list_instances(),
            "detailed_info": _autosys_system.db_manager.get_instance_info(),
            "total_count": len(_autosys_system.db_manager.instances)
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# USAGE EXAMPLES AND INTEGRATION
# ============================================================================

def example_multi_database_setup():
    """Example showing how to set up multiple database instances"""
    
    # Example database configurations
    database_configs = {
        "PROD": {
            "autosys_db": "AutosysOracleDatabase('oracle+oracledb://user:pass@prod-host:1521/prod_service')",
            "description": "Production Autosys environment"
        },
        "DEV": {
            "autosys_db": "AutosysOracleDatabase('oracle+oracledb://user:pass@dev-host:1521/dev_service')",
            "description": "Development Autosys environment"  
        },
        "TEST": {
            "autosys_db": "AutosysOracleDatabase('oracle+oracledb://user:pass@test-host:1521/test_service')",
            "description": "Testing Autosys environment"
        },
        "UAT": {
            "autosys_db": "AutosysOracleDatabase('oracle+oracledb://user:pass@uat-host:1521/uat_service')",
            "description": "User Acceptance Testing environment"
        }
    }
    
    # Your existing LLM setup
    # llm = get_llm("langchain")
    
    # Initialize multi-database system
    # setup_autosys_multi_database_system(database_configs, llm)
    
    # Test queries - the system will handle instance selection automatically
    test_queries = [
        "Hello!",  # → General conversation
        "Show me failed jobs",  # → Will ask for instance clarification
        "List running jobs in PROD instance",  # → Direct to PROD
        "Check job status in DEV for job ABC123",  # → Direct to DEV with job filter
        "What instances are available?"  # → System information
    ]
    
    return "Setup example ready"

def main():
    """Example usage and testing"""
    
    print("Multi-Database Autosys LangGraph System")
    print("=====================================")
    print()
    print("Features:")
    print("- Multiple database instance support")
    print("- Smart parameter extraction")
    print("- Automatic instance selection")
    print("- Conversation and database query routing")
    print("- Session memory and context")
    print()
    
    # Example conversations:
    example_conversations = [
        {
            "user": "Hi there!",
            "expected": "General conversation response"
        },
        {
            "user": "Show me failed jobs",  
            "expected": "Request for instance clarification"
        },
        {
            "user": "List failed jobs in PROD instance",
            "expected": "SQL query against PROD database"
        },
        {
            "user": "Check job XYZ123 status in DEV",
            "expected": "Job-specific query in DEV instance" 
        }
    ]
    
    print("Example conversation flows:")
    for i, conv in enumerate(example_conversations, 1):
        print(f"{i}. User: '{conv['user']}'")
        print(f"   Expected: {conv['expected']}")
        print()

if __name__ == "__main__":
    main()

# ============================================================================
# COMPLETE INTEGRATION INSTRUCTIONS
# ============================================================================

"""
MULTI-DATABASE INTEGRATION GUIDE:

1. SETUP YOUR DATABASE CONFIGURATIONS:

database_configs = {
    "PROD": {
        "autosys_db": AutosysOracleDatabase("oracle+oracledb://user:pass@prod:1521/service"),
        "description": "Production environment"
    },
    "DEV": {
        "autosys_db": AutosysOracleDatabase("oracle+oracledb://user:pass@dev:1521/service"), 
        "description": "Development environment"
    },
    "TEST": {
        "autosys_db": AutosysOracleDatabase("oracle+oracledb://user:pass@test:1521/service"),
        "description": "Testing environment"
    }
}

llm = get_llm("langchain")

2. INITIALIZE THE SYSTEM:

setup_autosys_multi_database_system(database_configs, llm)

3. USE THE SAME INTERFACE:

response = get_chat_response(user_message, session_id)

4. QUERY EXAMPLES:

- "Hello!" → General conversation
- "Show me failed jobs" → System asks for instance clarification  
- "List failed jobs in PROD" → Queries PROD database
- "Check job ABC123 in DEV instance" → Specific job query in DEV
- "Show running jobs in TEST environment" → Queries TEST database

5. AUTOMATIC FEATURES:

✓ Instance detection from user messages
✓ Parameter extraction (job names, calendars)
✓ Clarification requests when needed
✓ Smart routing (conversation vs database)
✓ Session memory for context
✓ Professional HTML formatting
✓ Multi-database error handling

6. NO CHANGES NEEDED TO YOUR EXISTING:

✓ API endpoints
✓ Session handling  
✓ Client-side code
✓ get_chat_response() function signature

The system automatically handles all the complexity behind the scenes!
"""
