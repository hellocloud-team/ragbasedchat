
import re
import sqlparse
import pandas as pd
import oracledb
from flask import Flask, request, jsonify
from collections import defaultdict
import yaml
import vanna as vn
from llm_utils import get_llm   # ✅ use your helper

# ---------- Load Config ----------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DB_CONFIG = config["databases"]

ALLOWED_TABLES = ["employees", "orders", "customers"]
MAX_ROWS_RETURN = 50
SESSIONS = defaultdict(dict)

# ---------- Initialize LLM via your helper ----------
llm = get_llm("gemini")
vn.set_llm(llm)

# (Optional) Train Vanna with allowed tables. 
# Later we can fetch table/column metadata dynamically using DB owners.
vn.train(ALLOWED_TABLES)

# ---------- Safety ----------
def safe_sql_check(sql: str, allowed_tables: list) -> bool:
    stmts = sqlparse.split(sql)
    if len(stmts) != 1:
        return False
    if not sql.lower().strip().startswith("select"):
        return False
    forbidden = re.compile(r"\b(insert|update|delete|drop|alter|truncate|create)\b", re.I)
    if forbidden.search(sql):
        return False
    return any(t in sql.lower() for t in allowed_tables)

# ---------- Run Oracle ----------
def run_query(instance: str, sql: str):
    if instance not in DB_CONFIG:
        raise ValueError(f"Unknown DB instance: {instance}")

    cfg = DB_CONFIG[instance]
    connection = oracledb.connect(
        user=cfg["user"], password=cfg["password"], dsn=cfg["dsn"]
    )
    with connection.cursor() as cursor:
        cursor.execute(sql)
        cols = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
    connection.close()
    return pd.DataFrame(rows, columns=cols)

# ---------- Flask App ----------
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    session_id = data.get("session_id", "default")
    nl_query = data.get("nl_query")
    instance = data.get("instance")
    state = SESSIONS[session_id]

    if nl_query:
        state["nl_query"] = nl_query

    if not instance:
        if "nl_query" not in state:
            return jsonify({"message": "Please provide a question."})
        return jsonify({
            "message": "Which DB instance should I run this on?",
            "options": list(DB_CONFIG.keys())
        })

    if "nl_query" not in state:
        return jsonify({"message": "No question found in session, please ask again."})

    # ✅ Use Vanna to generate SQL
    sql = vn.ask(state["nl_query"])

    if not safe_sql_check(sql, ALLOWED_TABLES):
        return jsonify({"error": "Unsafe or unapproved SQL generated", "sql": sql})

    try:
        df = run_query(instance, sql)
    except Exception as e:
        return jsonify({"error": str(e), "sql": sql})

    return jsonify({
        "sql": sql,
        "columns": list(df.columns),
        "rows": df.head(MAX_ROWS_RETURN).to_dict(orient="records"),
        "total_rows": len(df),
        "message": "Here are your results."
    })

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)





databases:
  db1:
    user: "scott"
    password: "tiger"
    dsn: "localhost:1521/ORCLPDB1"
    owner: "SCOTT"
  db2:
    user: "scott"
    password: "tiger"
    dsn: "otherhost:1521/ORCLPDB2"
    owner: "SCOTT"
  db3:
    user: "hr"
    password: "hrpass"
    dsn: "db3host:1521/ORCLPDB3"
    owner: "HR"
  db4:
    user: "hr"
    password: "hrpass"
    dsn: "db4host:1521/ORCLPDB4"
    owner: "HR"

llm:
  provider: "gemini"
  api_key: "your_gemini_api_key_here"

agent:
  max_rows_return: 50
  allowed_tables: ["employees", "orders", "customers"]


  


import os
import re
import sqlparse
import pandas as pd
import oracledb
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
app = Flask(__name__)

# --- Oracle DB connection configs ---
DB_CONFIG = {
    "db1": {"user": os.getenv("DB1_USER"), "password": os.getenv("DB1_PASS"), "dsn": os.getenv("DB1_DSN")},
    "db2": {"user": os.getenv("DB2_USER"), "password": os.getenv("DB2_PASS"), "dsn": os.getenv("DB2_DSN")},
    "db3": {"user": os.getenv("DB3_USER"), "password": os.getenv("DB3_PASS"), "dsn": os.getenv("DB3_DSN")},
    "db4": {"user": os.getenv("DB4_USER"), "password": os.getenv("DB4_PASS"), "dsn": os.getenv("DB4_DSN")},
}

# Restrict which tables can be queried
ALLOWED_TABLES = ["employees", "orders", "customers"]
MAX_ROWS_RETURN = 50

# --- Session memory ---
SESSIONS = defaultdict(dict)

# ---------- Gemini Stub ----------
def call_gemini_generate_sql(nl_query: str, allowed_tables: list) -> str:
    """
    In production: call Gemini API with a strong prompt.
    For now: simple stub that switches based on keywords.
    """
    q = nl_query.lower()

    if "order" in q:
        return f"SELECT * FROM orders FETCH FIRST {MAX_ROWS_RETURN+1} ROWS ONLY"
    elif "customer" in q:
        return f"SELECT * FROM customers FETCH FIRST {MAX_ROWS_RETURN+1} ROWS ONLY"
    else:
        return f"SELECT * FROM employees FETCH FIRST {MAX_ROWS_RETURN+1} ROWS ONLY"

# ---------- Safety ----------
def safe_sql_check(sql: str, allowed_tables: list) -> bool:
    stmts = sqlparse.split(sql)
    if len(stmts) != 1:
        return False
    if not sql.lower().strip().startswith("select"):
        return False
    forbidden = re.compile(r"\b(insert|update|delete|drop|alter|truncate|create)\b", re.I)
    if forbidden.search(sql):
        return False

    # check table names roughly
    for table in allowed_tables:
        if table in sql.lower():
            return True
    return False

# ---------- Run Oracle query ----------
def run_query(instance: str, sql: str):
    if instance not in DB_CONFIG:
        raise ValueError(f"Unknown DB instance {instance}")
    cfg = DB_CONFIG[instance]
    connection = oracledb.connect(user=cfg["user"], password=cfg["password"], dsn=cfg["dsn"])
    with connection.cursor() as cursor:
        cursor.execute(sql)
        cols = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
    connection.close()
    return pd.DataFrame(rows, columns=cols)

# ---------- Conversational endpoint ----------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    session_id = data.get("session_id", "default")
    nl_query = data.get("nl_query")
    instance = data.get("instance")

    state = SESSIONS[session_id]

    # If query provided, save it
    if nl_query:
        state["nl_query"] = nl_query

    # If no instance yet, ask for it
    if not instance:
        if "nl_query" not in state:
            return jsonify({"message": "Please provide a question."}), 200
        return jsonify({
            "message": "Which DB instance should I run this on?",
            "options": list(DB_CONFIG.keys())
        }), 200

    # Need a stored question
    if "nl_query" not in state:
        return jsonify({"message": "No question found in session, please ask again."}), 400

    # Generate SQL from Gemini
    sql = call_gemini_generate_sql(state["nl_query"], ALLOWED_TABLES)

    if not safe_sql_check(sql, ALLOWED_TABLES):
        return jsonify({"error": "Unsafe SQL generated", "sql": sql}), 400

    try:
        df = run_query(instance, sql)
    except Exception as e:
        return jsonify({"error": str(e), "sql": sql}), 500

    rows = df.head(MAX_ROWS_RETURN).to_dict(orient="records")
    return jsonify({
        "sql": sql,
        "columns": list(df.columns),
        "rows": rows,
        "total_rows": len(df),
        "message": "Here are your results."
    }), 200

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)




def extract_parameters_llm_node(self, state: AutosysState) -> AutosysState:
    """Extract parameters while preserving ALL previous values."""
    try:
        session_id = state["session_id"]
        
        # Get existing parameters from session
        session_context = session_manager.get_session_context(session_id)
        existing_params = session_context.get("extracted_params", {})
        
        # CRITICAL: Check what we already have
        self.logger.info(f"Existing params from session: {existing_params}")
        
        # Build the extraction prompt with explicit context preservation
        extraction_prompt = f"""
You MUST preserve previously extracted parameters. Analyze this query in context:

CURRENT QUERY: "{state['user_question']}"

ALREADY EXTRACTED (DO NOT LOSE THESE):
- Instance: {existing_params.get('instance', 'NOT SET')}
- Job Name: {existing_params.get('job_name', 'NOT SET')}
- Calendar Name: {existing_params.get('calendar_name', 'NOT SET')}

AVAILABLE INSTANCES: {', '.join(self.db_manager.list_instances())}

RULES:
1. If a parameter is ALREADY SET above, you MUST include it in your response
2. Only extract NEW parameters from the current query
3. Combine previous + new parameters in your response
4. If current query only provides instance, keep the previous job_name/calendar_name

Return JSON:
{{
    "extracted_instance": "current_or_previous_instance",
    "extracted_job_name": "current_or_previous_job_name",
    "extracted_calendar_name": "current_or_previous_calendar_name",
    "user_intent": "what_user_wants"
}}

EXAMPLE: If previous had job_name="job123" and current query is "DA3", respond:
{{
    "extracted_instance": "DA3",
    "extracted_job_name": "job123",
    "extracted_calendar_name": null,
    "user_intent": "provide_missing_instance"
}}
"""
        
        response = self.cached_llm_invoke(extraction_prompt)
        extraction_result = self._safe_parse_llm_json(response)
        
        # DEFENSIVE MERGING: Never lose existing values
        merged_params = {}
        
        # For each parameter, use this priority: new_value > existing_value > None
        merged_params["instance"] = (
            extraction_result.get("extracted_instance") 
            or existing_params.get("instance") 
            or ""
        )
        
        merged_params["job_name"] = (
            extraction_result.get("extracted_job_name") 
            or existing_params.get("job_name") 
            or ""
        )
        
        merged_params["calendar_name"] = (
            extraction_result.get("extracted_calendar_name") 
            or existing_params.get("calendar_name") 
            or ""
        )
        
        # Clean up null/none strings
        for key in merged_params:
            if merged_params[key] and merged_params[key].lower() in ["null", "none"]:
                merged_params[key] = ""
        
        # Determine what's STILL missing after merge
        missing = []
        if not merged_params.get("instance"):
            missing.append("instance name")
        if not (merged_params.get("job_name") or merged_params.get("calendar_name")):
            missing.append("job name or calendar name")
        
        # Update state with merged values
        state["extracted_instance"] = merged_params["instance"]
        state["extracted_job_name"] = merged_params["job_name"]
        state["extracted_calendar_name"] = merged_params["calendar_name"]
        state["missing_parameters"] = missing
        
        # CRITICAL: Save merged params back to session
        session_manager.update_session_context(session_id, {
            "extracted_params": merged_params,
            "user_intent": extraction_result.get("user_intent", ""),
            "pending_clarification": bool(missing)
        })
        
        self.logger.info(
            f"Session {session_id} - "
            f"Previous: {existing_params} | "
            f"Extracted: {extraction_result} | "
            f"Merged: {merged_params} | "
            f"Missing: {missing}"
        )
        
    except Exception as e:
        self.logger.error(f"Parameter extraction failed: {str(e)}")
        state["missing_parameters"] = ["extraction_error"]
    
    return state





def extract_parameters_llm_node(self, state: AutosysState) -> AutosysState:
    """Extract parameters while preserving session context and avoiding redundant prompts."""
    try:
        session_id = state["session_id"]
        
        # Get existing session context
        session_context = session_manager.get_session_context(session_id)
        existing_params = session_context.get("extracted_params", {})
        
        # CRITICAL: Only extract if we're missing required parameters
        current_missing = []
        if not existing_params.get("instance"):
            current_missing.append("instance name")
        if not (existing_params.get("job_name") or existing_params.get("calendar_name")):
            current_missing.append("job name or calendar name")
        
        # If we already have all required parameters, skip extraction
        if not current_missing:
            self.logger.info(f"All parameters available from context: {existing_params}")
            return state
        
        # Create extraction prompt that explicitly preserves existing values
        extraction_prompt = f"""
Extract parameters from this query, considering previous context:

CURRENT USER QUERY: "{state['user_question']}"

PREVIOUS EXTRACTED PARAMETERS: {json.dumps(existing_params, indent=2)}
STILL MISSING: {', '.join(current_missing)}

AVAILABLE INSTANCES: {', '.join(self.db_manager.list_instances())}

Instructions:
1. Extract any NEW parameters from the current query
2. PRESERVE all parameters from PREVIOUS EXTRACTED PARAMETERS
3. Only mark as null if explicitly stated as unknown in current query
4. If user provides missing information, use it to fill gaps

Return ONLY JSON:
{{
    "extracted_instance": "instance_name_or_use_previous_or_null",
    "extracted_job_name": "job_name_or_use_previous_or_null",
    "extracted_calendar_name": "calendar_name_or_use_previous_or_null",
    "user_intent": "what_user_wants_to_do",
    "context_used": true_if_previous_context_was_helpful,
    "missing_still": ["list", "of", "still_missing", "params"]
}}
"""
        
        response = self.cached_llm_invoke(extraction_prompt)
        extraction_result = self._safe_parse_llm_json(response)
        
        if not extraction_result:
            extraction_result = {"missing_still": current_missing}
        
        # IMPROVED MERGING: Prioritize existing values, only update with new non-null values
        merged_params = existing_params.copy()  # Start with what we have
        
        # Only update if new value is provided and not null
        new_instance = extraction_result.get("extracted_instance")
        if new_instance and new_instance.lower() not in ["null", "none", ""]:
            merged_params["instance"] = new_instance
        
        new_job = extraction_result.get("extracted_job_name")
        if new_job and new_job.lower() not in ["null", "none", ""]:
            merged_params["job_name"] = new_job
        
        new_calendar = extraction_result.get("extracted_calendar_name")
        if new_calendar and new_calendar.lower() not in ["null", "none", ""]:
            merged_params["calendar_name"] = new_calendar
        
        # Determine what's still missing AFTER merge
        missing = []
        if not merged_params.get("instance"):
            missing.append("instance name")
        if not (merged_params.get("job_name") or merged_params.get("calendar_name")):
            missing.append("job name or calendar name")
        
        # Update state
        state["extracted_instance"] = merged_params.get("instance", "")
        state["extracted_job_name"] = merged_params.get("job_name", "")
        state["extracted_calendar_name"] = merged_params.get("calendar_name", "")
        state["missing_parameters"] = missing
        
        # Update session context with merged parameters
        session_manager.update_session_context(session_id, {
            "extracted_params": merged_params,
            "user_intent": extraction_result.get("user_intent", ""),
            "pending_clarification": bool(missing)
        })
        
        # Store extraction for debugging
        if "llm_analysis" not in state:
            state["llm_analysis"] = {}
        state["llm_analysis"]["extraction_with_memory"] = {
            "previous_params": existing_params,
            "new_extraction": extraction_result,
            "merged_params": merged_params,
            "still_missing": missing
        }
        
        self.logger.info(f"Session {session_id} - Merged params: {merged_params}, Missing: {missing}")
        
    except Exception as e:
        self.logger.error(f"Parameter extraction with memory failed: {str(e)}")
        state["missing_parameters"] = ["extraction_error"]
    
    return state










vcvv"""''''''''''''''’""''
# SESSION-AWARE MULTI-AGENT ROUTER WITH CONTEXT FORWARDING

from typing import Dict, Any, Optional, List
import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime
import requests

@dataclass
class SessionContext:
    """Session context to store extracted information across conversations"""
    session_id: str
    job_name: Optional[str] = None
    instance_name: Optional[str] = None
    last_agent_used: Optional[str] = None
    conversation_history: List[Dict] = None
    extracted_entities: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.extracted_entities is None:
            self.extracted_entities = {}
    
    def add_conversation(self, user_query: str, agent_used: str, response: str, extracted_data: Dict = None):
        """Add conversation to history with extracted data"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "agent_used": agent_used,
            "response": response[:200],  # Truncate for storage
            "extracted_data": extracted_data or {}
        }
        self.conversation_history.append(entry)
        
        # Keep only last 10 conversations
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def update_entities(self, new_entities: Dict[str, Any]):
        """Update extracted entities"""
        self.extracted_entities.update(new_entities)
    
    def get_context_summary(self) -> str:
        """Get summary of current context for LLM"""
        context_parts = []
        
        if self.job_name:
            context_parts.append(f"Job Name: {self.job_name}")
        if self.instance_name:
            context_parts.append(f"Instance: {self.instance_name}")
        if self.last_agent_used:
            context_parts.append(f"Last Agent: {self.last_agent_used}")
        
        if self.extracted_entities:
            for key, value in self.extracted_entities.items():
                if value:
                    context_parts.append(f"{key}: {value}")
        
        return " | ".join(context_parts) if context_parts else "No context available"

class EntityExtractor:
    """Extract entities from user queries and agent responses using LLM"""
    
    def __init__(self, get_llm_function):
        self.llm = get_llm_function("langchain")
    
    def extract_from_query(self, query: str) -> Dict[str, Any]:
        """Extract job name, instance name, and other entities from user query"""
        
        extraction_prompt = f"""Extract AutoSys-related entities from this user query:

Query: "{query}"

Extract the following if present:
1. Job Name: Any job identifier, name, or ID
2. Instance Name: AutoSys instance (DA3, DB3, DC3, DG3, LS3)
3. Time References: Dates, time periods, "last 24 hours", etc.
4. Status Types: running, failed, success, pending, etc.
5. Other Entities: calendar names, application names, etc.

Respond in JSON format:
{{
    "job_name": "job_name_if_found_or_null",
    "instance_name": "instance_if_found_or_null",
    "time_reference": "time_reference_if_found_or_null",
    "status_type": "status_if_found_or_null",
    "other_entities": {{"key": "value"}}
}}

Only include entities that are explicitly mentioned. Use null for missing entities."""
        
        try:
            if hasattr(self.llm, 'predict'):
                response = self.llm.predict(extraction_prompt)
            elif hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(extraction_prompt).content
            else:
                response = str(self.llm(extraction_prompt))
            
            # Try to parse JSON response
            try:
                entities = json.loads(response.strip())
                return entities
            except json.JSONDecodeError:
                # Fallback to regex extraction
                return self._regex_fallback_extraction(query)
                
        except Exception as e:
            logging.error(f"Entity extraction failed: {e}")
            return self._regex_fallback_extraction(query)
    
    def _regex_fallback_extraction(self, query: str) -> Dict[str, Any]:
        """Fallback regex-based entity extraction"""
        
        entities = {
            "job_name": None,
            "instance_name": None,
            "time_reference": None,
            "status_type": None,
            "other_entities": {}
        }
        
        query_upper = query.upper()
        
        # Extract instance name
        instances = ["DA3", "DB3", "DC3", "DG3", "LS3"]
        for instance in instances:
            if instance in query_upper:
                entities["instance_name"] = instance
                break
        
        # Extract job name patterns
        job_patterns = [
            r'job\s+([A-Z0-9_]+)',
            r'([A-Z0-9_]{3,})\s+job',
            r'job\s*:\s*([A-Z0-9_]+)',
            r'([A-Z0-9_]+)\s+status'
        ]
        
        for pattern in job_patterns:
            match = re.search(pattern, query_upper)
            if match:
                entities["job_name"] = match.group(1)
                break
        
        # Extract status types
        status_patterns = ['RUNNING', 'FAILED', 'SUCCESS', 'PENDING', 'HOLD']
        for status in status_patterns:
            if status in query_upper:
                entities["status_type"] = status
                break
        
        return entities
    
    def extract_from_response(self, response_text: str, agent_type: str) -> Dict[str, Any]:
        """Extract entities from agent responses"""
        
        extraction_prompt = f"""Extract information provided by the {agent_type} agent:

Agent Response: "{response_text}"

Extract any mentioned:
1. Job Names/IDs
2. Instance Names
3. Dates/Times
4. Status Information
5. Other relevant data

Respond in JSON format with extracted information."""
        
        try:
            if hasattr(self.llm, 'predict'):
                response = self.llm.predict(extraction_prompt)
            else:
                response = str(self.llm(extraction_prompt))
            
            return json.loads(response.strip())
        except:
            return {}

class SessionAwareRouter:
    """Router with session memory and intelligent context forwarding"""
    
    def __init__(self, config, get_llm_function):
        self.config = config
        self.get_llm_function = get_llm_function
        self.llm = get_llm_function("langchain")
        
        # Session storage (in production, use Redis or database)
        self.sessions: Dict[str, SessionContext] = {}
        
        # Entity extractor
        self.entity_extractor = EntityExtractor(get_llm_function)
        
        # Tools
        self.jil_tool = self._create_jil_tool()
        self.job_tool = self._create_job_tool()
    
    def _get_or_create_session(self, session_id: str) -> SessionContext:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionContext(session_id=session_id)
        return self.sessions[session_id]
    
    def _create_jil_tool(self):
        def call_jil_api_with_context(query: str, session_id: str, context: Dict = None) -> Dict[str, Any]:
            try:
                payload = {
                    "message": query,
                    "session_id": session_id,
                    "context": context or {}
                }
                
                response = requests.post(
                    self.config.jil_agent_url,
                    json=payload,
                    headers=self.config.headers,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                return {
                    "success": True,
                    "data": response.json(),
                    "agent": "jil_agent"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "agent": "jil_agent"
                }
        
        return call_jil_api_with_context
    
    def _create_job_tool(self):
        def call_job_api_with_context(query: str, session_id: str, context: Dict = None) -> Dict[str, Any]:
            try:
                payload = {
                    "message": query,
                    "session_id": session_id,
                    "context": context or {}
                }
                
                response = requests.post(
                    self.config.job_agent_url,
                    json=payload,
                    headers=self.config.headers,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                return {
                    "success": True,
                    "data": response.json(),
                    "agent": "job_agent"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "agent": "job_agent"
                }
        
        return call_job_api_with_context
    
    def _determine_routing_with_context(self, query: str, session: SessionContext) -> str:
        """Determine routing considering session context"""
        
        routing_prompt = f"""You are an intelligent AutoSys query router with session awareness.

CURRENT SESSION CONTEXT:
{session.get_context_summary()}

RECENT CONVERSATION:
{json.dumps(session.conversation_history[-3:], indent=2) if session.conversation_history else "No previous conversation"}

AGENT CAPABILITIES:
🔧 JIL_AGENT: JIL syntax, onboarding, connection profiles, configuration, how-to guides
📊 JOB_AGENT: Job status, execution details, creation dates, job failures, scheduling, performance

CONTEXT-AWARE ROUTING RULES:
1. If job name and instance are already known in session, use that context
2. For follow-up questions about the same job, route based on the question type
3. Creation date, execution details, job history → JOB_AGENT
4. JIL syntax, configuration, setup → JIL_AGENT

USER QUERY: "{query}"

ROUTING DECISION: Respond with "JIL_AGENT" or "JOB_AGENT" considering the context."""
        
        try:
            if hasattr(self.llm, 'predict'):
                response = self.llm.predict(routing_prompt)
            else:
                response = str(self.llm(routing_prompt))
            
            decision = response.strip().upper()
            return "JIL_AGENT" if "JIL_AGENT" in decision else "JOB_AGENT"
            
        except Exception as e:
            logging.error(f"Routing decision failed: {e}")
            return "JOB_AGENT"  # Default
    
    def _enhance_query_with_context(self, query: str, session: SessionContext, target_agent: str) -> str:
        """Enhance query with session context before sending to agent"""
        
        enhancement_prompt = f"""Enhance this user query with available session context for the {target_agent}.

ORIGINAL QUERY: "{query}"

AVAILABLE SESSION CONTEXT:
- Job Name: {session.job_name or 'Not available'}
- Instance Name: {session.instance_name or 'Not available'}
- Previous Context: {session.get_context_summary()}

ENHANCEMENT RULES:
1. If job name and instance are available in context but not in query, add them
2. If this is a follow-up question, make the context explicit
3. Keep the original intent but add necessary context
4. Don't duplicate information already in the query

ENHANCED QUERY: Provide the enhanced version that includes necessary context."""
        
        try:
            if hasattr(self.llm, 'predict'):
                enhanced = self.llm.predict(enhancement_prompt)
            else:
                enhanced = str(self.llm(enhancement_prompt))
            
            # Extract just the enhanced query part
            enhanced_query = enhanced.strip()
            if "ENHANCED QUERY:" in enhanced_query:
                enhanced_query = enhanced_query.split("ENHANCED QUERY:")[-1].strip()
            
            return enhanced_query
            
        except Exception as e:
            logging.error(f"Query enhancement failed: {e}")
            # Fallback: manually add context
            context_parts = []
            if session.job_name and session.job_name not in query:
                context_parts.append(f"job name {session.job_name}")
            if session.instance_name and session.instance_name not in query:
                context_parts.append(f"instance {session.instance_name}")
            
            if context_parts:
                return f"{query} (using {', '.join(context_parts)} from session context)"
            else:
                return query
    
    def route_and_call(self, query: str, session_id: str, context: Dict = None) -> Dict[str, Any]:
        """Main routing function with session awareness"""
        
        try:
            # Get or create session
            session = self._get_or_create_session(session_id)
            
            logging.info(f"🔄 Session-aware routing for: '{query}'")
            logging.info(f"📊 Current context: {session.get_context_summary()}")
            
            # Extract entities from current query
            query_entities = self.entity_extractor.extract_from_query(query)
            logging.info(f"🔍 Extracted entities: {query_entities}")
            
            # Update session with new entities
            if query_entities.get("job_name"):
                session.job_name = query_entities["job_name"]
            if query_entities.get("instance_name"):
                session.instance_name = query_entities["instance_name"]
            
            session.update_entities(query_entities)
            
            # Determine routing with session context
            routing_decision = self._determine_routing_with_context(query, session)
            
            # Enhance query with session context
            enhanced_query = self._enhance_query_with_context(query, session, routing_decision)
            
            logging.info(f"🎯 Routing to: {routing_decision}")
            logging.info(f"🔧 Enhanced query: '{enhanced_query}'")
            
            # Prepare context for agent
            agent_context = {
                "session_context": asdict(session),
                "original_query": query,
                "enhanced_query": enhanced_query,
                "routing_reason": f"Routed to {routing_decision} based on context analysis"
            }
            
            # Call appropriate agent
            if routing_decision == "JIL_AGENT":
                result = self.jil_tool(enhanced_query, session_id, agent_context)
                agent_used = "jil_agent"
            else:
                result = self.job_tool(enhanced_query, session_id, agent_context)
                agent_used = "job_agent"
            
            # Extract entities from agent response
            if result["success"]:
                response_entities = self.entity_extractor.extract_from_response(
                    str(result["data"]), agent_used
                )
                session.update_entities(response_entities)
            
            # Update session history
            session.last_agent_used = agent_used
            session.add_conversation(
                query, 
                agent_used, 
                str(result.get("data", result.get("error", ""))),
                {**query_entities, **response_entities} if result["success"] else query_entities
            )
            
            return {
                "success": result["success"],
                "response": result.get("data", result.get("error")),
                "agent_used": agent_used,
                "routing_decision": routing_decision,
                "original_query": query,
                "enhanced_query": enhanced_query,
                "session_context": session.get_context_summary(),
                "extracted_entities": query_entities,
                "session_id": session_id,
                "context_forwarded": enhanced_query != query
            }
            
        except Exception as e:
            logging.error(f"Session-aware routing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "original_query": query
            }
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information for debugging"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            return {
                "session_id": session_id,
                "context_summary": session.get_context_summary(),
                "conversation_count": len(session.conversation_history),
                "last_agent": session.last_agent_used,
                "entities": session.extracted_entities
            }
        else:
            return {"session_id": session_id, "status": "not_found"}
    
    def clear_session(self, session_id: str) -> Dict[str, Any]:
        """Clear session data"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return {"session_id": session_id, "status": "cleared"}
        else:
            return {"session_id": session_id, "status": "not_found"}

# TESTING SCENARIO
def test_session_aware_routing(get_llm_function):
    """Test the session-aware routing with context forwarding"""
    
    print("🧪 TESTING SESSION-AWARE ROUTING")
    print("=" * 60)
    
    # Configuration
    config = APIConfig(
        jil_agent_url="http://localhost:8081/chat-atsys",
        job_agent_url="http://localhost:8080/chat"
    )
    
    # Create session-aware router
    router = SessionAwareRouter(config, get_llm_function)
    
    # Test scenario: JIL query followed by job query
    session_id = "test_session_123"
    
    test_conversation = [
        {
            "query": "I need the JIL of job ATSYS_DAILY_BACKUP on instance DA3",
            "expected_agent": "jil_agent",
            "description": "Initial JIL request - should extract job name and instance"
        },
        {
            "query": "What is the creation date of this job?",
            "expected_agent": "job_agent",
            "description": "Follow-up about creation date - should use context from previous query"
        },
        {
            "query": "Show me the job dependencies",
            "expected_agent": "job_agent", 
            "description": "Another follow-up - should continue using same context"
        },
        {
            "query": "How do I modify the JIL for a different schedule?",
            "expected_agent": "jil_agent",
            "description": "New JIL question - should still use job context if relevant"
        }
    ]
    
    for i, test in enumerate(test_conversation, 1):
        print(f"\n{i}. {test['description']}")
        print(f"   Query: '{test['query']}'")
        print(f"   Expected Agent: {test['expected_agent']}")
        
        try:
            # Show session before query
            session_info = router.get_session_info(session_id)
            print(f"   Session Before: {session_info.get('context_summary', 'Empty')}")
            
            # Execute query
            result = router.route_and_call(test['query'], session_id)
            
            print(f"   ✅ Success: {result['success']}")
            print(f"   🎯 Agent Used: {result['agent_used']}")
            print(f"   🔄 Context Forwarded: {result['context_forwarded']}")
            print(f"   🔧 Enhanced Query: '{result['enhanced_query']}'")
            print(f"   📊 Session Context: {result['session_context']}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        print("-" * 50)
    
    # Show final session state
    print(f"\n📊 FINAL SESSION STATE:")
    final_session = router.get_session_info(session_id)
    print(json.dumps(final_session, indent=2))

# FASTAPI INTEGRATION
def create_session_aware_fastapi(get_llm_function):
    """Create FastAPI app with session-aware routing"""
    
    @asynccontextmanager 
    async def lifespan(app: FastAPI):
        logging.info("Starting Session-Aware Router API...")
        yield
        logging.info("Shutting down Session-Aware Router API...")
    
    # Configuration
    config = APIConfig(
        jil_agent_url="http://localhost:8081/chat-atsys",
        job_agent_url="http://localhost:8080/chat"
    )
    
    # Create session-aware router
    router = SessionAwareRouter(config, get_llm_function)
    
    app = FastAPI(title="Session-Aware Multi-Agent Router", lifespan=lifespan)
    
    @app.post("/chat-atsys-route")
    def chat_with_session_awareness(request: ChatRequest):
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="No message provided")
        
        try:
            result = router.route_and_call(
                request.message, 
                request.session_id, 
                request.context
            )
            return result
            
        except Exception as e:
            logging.error(f"Session-aware chat error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/session/{session_id}")
    def get_session(session_id: str):
        """Get session information"""
        return router.get_session_info(session_id)
    
    @app.delete("/session/{session_id}")
    def clear_session(session_id: str):
        """Clear session data"""
        return router.clear_session(session_id)
    
    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "router_type": "session_aware_multi_agent",
            "features": ["context_forwarding", "entity_extraction", "session_memory"]
        }
    
    return app

if __name__ == "__main__":
    # Mock get_llm function for testing
    def mock_get_llm(framework):
        return "Mock Gemini LLM"
    
    # Test the session-aware routing
    test_session_aware_routing(mock_get_llm)
    
    print("\n" + "=" * 60)
    print("🎯 SESSION-AWARE ROUTING READY")
    print("=" * 60)
    
    print("""
✅ Key Features Implemented:

1. 🧠 Session Memory: Remembers job names, instances, and context
2. 🔄 Context Forwarding: Passes previous info to new queries  
3. 🎯 Intelligent Routing: Routes based on question type + context
4. 🔍 Entity Extraction: Extracts job names, instances from queries
5. 🔧 Query Enhancement: Adds context to queries automatically

✅ Example Flow:
1. "JIL of job ATSYS_BACKUP on DA3" → JIL Agent (stores job + instance)
2. "What's the creation date?" → Job Agent + context forwarded
3. "Show dependencies" → Job Agent + same context
    
✅ This eliminates re-asking for job name and instance!
    """)
££££££££££££¢£££¢



# FIXED ROUTER WITH PROPER TOOL CALLING

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from enum import Enum
import requests
import json
import logging
from contextlib import asynccontextmanager

# Your existing classes
class AgentAPI(Enum):
    JIL_AGENT = "jil_agent"
    JOB_AGENT = "job_agent"
    UNKNOWN = "unknown"

class ChatRequest(BaseModel):
    message: str
    session_id: str
    context: Optional[Dict[str, Any]] = # CREATE ROUTER WITH YOUR get_llm FUNCTION
def create_router_with_your_llm(get_llm_function):
    """Create router using your existing get_llm function"""
    
    # Configuration
    config = APIConfig(
        jil_agent_url="http://localhost:8081/chat-atsys",
        job_agent_url="http://localhost:8080/chat"
    )
    
    # Option 1: Use ToolBasedRouter with Gemini
    gemini_config = {
        "framework": "langchain", 
        "model": "gemini-1.5-pro-002",
        "temperature": 0.1,
        "top_p": 1
    }
    
    router = ToolBasedRouter(config, gemini_config)
    
    # Option 2: Use GeminiOptimizedRouter
    # router = GeminiOptimizedRouter(config, gemini_config)
    
    # Option 3: Use custom router with your get_llm function
    # CustomRouter = integrate_with_existing_get_llm(get_llm_function)
    # router = CustomRouter(config)
    
    return router

# TESTING YOUR GEMINI ROUTER
def test_gemini_router(get_llm_function):
    """Test the Gemini router with actual tool calls"""
    
    print("🧪 TESTING GEMINI ROUTER")
    print("=" * 50)
    
    router = create_router_with_your_llm(get_llm_function)
    
    test_queries = [
        ("how to onboard a new application in autosys", "Should call JIL agent"),
        ("what is the connection profile for DA3", "Should call JIL agent"),
        ("list all the jobs starting with ATSYS in DA3", "Should call JOB agent"),
        ("show job failures in last 24 hours", "Should call JOB agent"),
        ("provide next start time of the job", "Should call JOB agent")
    ]
    
    for query, expected in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected: {expected}")
        
        try:
            result = router.route_and_call(query, "test_session")
            print(f"Success: {result['success']}")
            print(f"Tools Used: {result.get('tools_used', [])}")
            print(f"LLM Model: {result.get('llm_model', 'Unknown')}")
            print(f"Response: {result.

class APIConfig:
    def __init__(self, jil_agent_url: str, job_agent_url: str, timeout: int = 30):
        self.jil_agent_url = jil_agent_url
        self.job_agent_url = job_agent_url
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}

# SOLUTION 1: Replace DummyLLM with Real LLM Tool Calling
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate

class ToolBasedRouter:
    """Router that uses actual LLM tools instead of string classification"""
    
    def __init__(self, config: APIConfig, get_llm_function):
        self.config = config
        self.get_llm_function = get_llm_function
        
        # Use your existing get_llm function
        self.llm = self.get_llm_function("langchain")
        
        # Create tools for each agent
        self.tools = [
            self._create_jil_agent_tool(),
            self._create_job_agent_tool()
        ]
        
        # Create agent with tools
        self.agent = self._create_routing_agent()
    
    def _create_jil_agent_tool(self):
        """Create tool that calls JIL agent API"""
        
        @tool
        def call_jil_agent(query: str) -> str:
            """
            Call JIL Agent for queries about JIL, Confluence, Autoping, Connection Profile, 
            onboarding, how-to guides, and general system usage.
            
            Use for:
            - How to onboard applications
            - Connection profiles
            - JIL configuration
            - System setup guides
            - General AutoSys questions
            
            Args:
                query: User's question about JIL/system configuration
            """
            try:
                payload = {"message": query, "session_id": "router_session"}
                response = requests.post(
                    self.config.jil_agent_url,
                    json=payload,
                    headers=self.config.headers,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                return f"JIL Agent Response: {result.get('data', result)}"
                
            except Exception as e:
                return f"JIL Agent Error: {str(e)}"
        
        return call_jil_agent
    
    def _create_job_agent_tool(self):
        """Create tool that calls JOB agent API"""
        
        @tool  
        def call_job_agent(query: str) -> str:
            """
            Call Job Agent for queries about job status, job names, job failures, 
            calendars, next start time, and job-related troubleshooting.
            
            Use for:
            - Job status queries
            - Job failures and errors
            - Schedule and calendar information
            - Job execution details
            - Performance monitoring
            
            Args:
                query: User's question about jobs and scheduling
            """
            try:
                payload = {"message": query, "session_id": "router_session"}
                response = requests.post(
                    self.config.job_agent_url,
                    json=payload,
                    headers=self.config.headers,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                return f"Job Agent Response: {result.get('data', result)}"
                
            except Exception as e:
                return f"Job Agent Error: {str(e)}"
        
        return call_job_agent
    
    def _create_routing_agent(self):
        """Create agent that uses tools for routing"""
        
        system_prompt = """You are an intelligent AutoSys router. You have access to two specialized agents:

1. JIL AGENT (call_jil_agent): For JIL, Confluence, Autoping, Connection Profile, onboarding, how-to guides, and general system usage
2. JOB AGENT (call_job_agent): For job status, job names, job failures, calendars, next start time, and job-related troubleshooting

MANDATORY RULES:
- You MUST use one of the available tools for every user query
- NEVER provide direct answers without calling a tool
- Choose the most appropriate tool based on the query content
- If unsure, prefer the JOB agent for job-related queries

Examples of routing:
- "how to onboard a new application in autosys" → call_jil_agent
- "what is the connection profile for DA3" → call_jil_agent  
- "list all the jobs starting with ATSYS in DA3" → call_job_agent
- "show job failures in last 24 hours" → call_job_agent
- "provide next start time of the job" → call_job_agent

Always call the appropriate tool first, then provide the response."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create tool-calling agent
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        
        # Create executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
        
        return agent_executor
    
    def route_and_call(self, query: str, session_id: str, context: Dict = # USAGE WITH YOUR get_llm FUNCTION - UPDATED
def create_router_with_your_get_llm(get_llm_function):
    """Create router using your existing get_llm function"""
    
    # Configuration
    config = APIConfig(
        jil_agent_url="http) -> Dict[str, Any]:
        """Route query and call appropriate agent using tools"""
        
        try:
            logging.info(f"Routing query: {query}")
            
            # Execute agent with tools
            result = self.agent.invoke({"input": query})
            
            # Extract tool usage information
            intermediate_steps = result.get("intermediate_steps", [])
            tools_used = [step[0].tool for step in intermediate_steps]
            
            logging.info(f"Tools used: {tools_used}")
            
            return {
                "success": True,
                "response": result["output"],
                "tools_used": tools_used,
                "agent_type": "tool_based_router",
                "session_id": session_id
            }
            
        except Exception as e:
            logging.error(f"Router error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "agent_type": "tool_based_router",
                "session_id": session_id
            }

# SOLUTION 2: Alternative - Direct Tool Selection Router
class DirectToolRouter:
    """LLM-based router that uses Gemini to decide which tool to call"""
    
    def __init__(self, config: APIConfig, get_llm_function):
        self.config = config
        self.get_llm_function = get_llm_function
        
        # Use your get_llm function for LLM-based routing decisions
        self.llm = self.get_llm_function("langchain")
        
        # Create API calling tools
        self.jil_tool = self._create_jil_tool()
        self.job_tool = self._create_job_tool()
        
        # LLM routing prompt template
        self.routing_prompt_template = self._create_routing_prompt_template()
    
    def _create_routing_prompt_template(self) -> str:
        """Create LLM prompt template for routing decisions"""
        
        return """You are an intelligent AutoSys query router. Analyze the user query and determine which agent should handle it.

AGENT DESCRIPTIONS:

🔧 JIL_AGENT: Handles queries about:
- JIL (Job Information Language) configuration and syntax
- Confluence documentation and guides
- Autoping setup and configuration  
- Connection profiles and database connections
- Onboarding new applications to AutoSys
- How-to guides and tutorials
- General system usage and setup
- System configuration and administration

📊 JOB_AGENT: Handles queries about:
- Job status and execution details
- Job names, IDs, and identification
- Job failures, errors, and troubleshooting
- Calendar and scheduling information
- Next start times and job timing
- Job performance and monitoring
- Running, failed, success status queries
- Job-related operational issues

ROUTING EXAMPLES:
"how to onboard a new application in autosys" → JIL_AGENT
"what is the connection profile for DA3" → JIL_AGENT
"show me JIL syntax for creating jobs" → JIL_AGENT
"configure autoping for new environment" → JIL_AGENT

"list all the jobs starting with ATSYS in DA3" → JOB_AGENT
"show job failures in last 24 hours" → JOB_AGENT
"what is the status of job123" → JOB_AGENT
"provide next start time of job456" → JOB_AGENT
"find all running jobs on DB3" → JOB_AGENT

ANALYSIS INSTRUCTIONS:
1. Read the user query carefully
2. Identify key terms and context
3. Match the query type to agent capabilities
4. Consider the primary intent of the user

USER QUERY: "{query}"

ROUTING DECISION: Respond with exactly "JIL_AGENT" or "JOB_AGENT" based on your analysis.
"""
    
    def _create_jil_tool(self):
        """Create function to call JIL agent API"""
        
        def call_jil_api(query: str, session_id: str) -> Dict[str, Any]:
            """Call JIL Agent API for configuration and onboarding queries"""
            try:
                payload = {
                    "message": query, 
                    "session_id": session_id,
                    "agent_type": "jil_agent"
                }
                
                logging.info(f"🔧 Calling JIL Agent API: {self.config.jil_agent_url}")
                
                response = requests.post(
                    self.config.jil_agent_url,
                    json=payload,
                    headers=self.config.headers,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                api_result = response.json()
                
                return {
                    "success": True,
                    "data": api_result,
                    "agent": "jil_agent",
                    "api_url": self.config.jil_agent_url,
                    "status_code": response.status_code
                }
                
            except requests.exceptions.Timeout:
                return {
                    "success": False,
                    "error": f"JIL Agent API timeout after {self.config.timeout}s",
                    "agent": "jil_agent"
                }
            except requests.exceptions.ConnectionError:
                return {
                    "success": False,
                    "error": "Failed to connect to JIL Agent API",
                    "agent": "jil_agent"
                }
            except requests.exceptions.HTTPError as e:
                return {
                    "success": False,
                    "error": f"JIL Agent API error: {e.response.status_code}",
                    "agent": "jil_agent"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"JIL Agent API call failed: {str(e)}",
                    "agent": "jil_agent"
                }
        
        return call_jil_api
    
    def _create_job_tool(self):
        """Create function to call Job agent API"""
        
        def call_job_api(query: str, session_id: str) -> Dict[str, Any]:
            """Call Job Agent API for job status and operational queries"""
            try:
                payload = {
                    "message": query,
                    "session_id": session_id, 
                    "agent_type": "job_agent"
                }
                
                logging.info(f"📊 Calling Job Agent API: {self.config.job_agent_url}")
                
                response = requests.post(
                    self.config.job_agent_url,
                    json=payload,
                    headers=self.config.headers,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                api_result = response.json()
                
                return {
                    "success": True,
                    "data": api_result,
                    "agent": "job_agent",
                    "api_url": self.config.job_agent_url,
                    "status_code": response.status_code
                }
                
            except requests.exceptions.Timeout:
                return {
                    "success": False,
                    "error": f"Job Agent API timeout after {self.config.timeout}s",
                    "agent": "job_agent"
                }
            except requests.exceptions.ConnectionError:
                return {
                    "success": False,
                    "error": "Failed to connect to Job Agent API",
                    "agent": "job_agent"
                }
            except requests.exceptions.HTTPError as e:
                return {
                    "success": False,
                    "error": f"Job Agent API error: {e.response.status_code}",
                    "agent": "job_agent"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Job Agent API call failed: {str(e)}",
                    "agent": "job_agent"
                }
        
        return call_job_api
    
    def _get_llm_routing_decision(self, query: str) -> str:
        """Use LLM to make routing decision"""
        
        try:
            # Create routing prompt with user query
            routing_prompt = self.routing_prompt_template.format(query=query)
            
            logging.info(f"🧠 Getting LLM routing decision for: '{query}'")
            
            # Get LLM decision using your get_llm function
            if hasattr(self.llm, 'predict'):
                # LangChain LLM
                llm_response = self.llm.predict(routing_prompt)
            elif hasattr(self.llm, 'invoke'):
                # LangChain chat model
                llm_response = self.llm.invoke(routing_prompt).content
            elif hasattr(self.llm, 'generate_content'):
                # Direct Gemini API
                llm_response = self.llm.generate_content(routing_prompt).text
            else:
                # Fallback - try calling as function
                llm_response = str(self.llm(routing_prompt))
            
            # Extract decision from LLM response
            decision = llm_response.strip().upper()
            logging.info(f"🤖 LLM routing decision: '{decision}'")
            
            # Validate decision
            if "JIL_AGENT" in decision:
                return "JIL_AGENT"
            elif "JOB_AGENT" in decision:
                return "JOB_AGENT"
            else:
                logging.warning(f"⚠️ LLM returned unclear decision: '{decision}', defaulting to JOB_AGENT")
                return "JOB_AGENT"  # Default to job agent for operational queries
                
        except Exception as e:
            logging.error(f"❌ LLM routing decision failed: {str(e)}")
            # Fallback to simple keyword-based routing
            return self._fallback_keyword_routing(query)
    
    def _fallback_keyword_routing(self, query: str) -> str:
        """Fallback keyword-based routing if LLM fails"""
        
        query_lower = query.lower()
        
        # JIL-related keywords (configuration, setup, how-to)
        jil_keywords = [
            'onboard', 'connection profile', 'confluence', 'autoping',
            'how to', 'setup', 'configure', 'jil', 'guide', 'tutorial',
            'documentation', 'install', 'create application', 'syntax'
        ]
        
        # Job-related keywords (operational, status, monitoring)
        job_keywords = [
            'job', 'status', 'failure', 'calendar', 'schedule', 'start time',
            'running', 'failed', 'success', 'pending', 'da3', 'db3', 'dc3',
            'dg3', 'ls3', 'execution', 'monitor', 'performance', 'error'
        ]
        
        # Count keyword matches
        jil_score = sum(1 for keyword in jil_keywords if keyword in query_lower)
        job_score = sum(1 for keyword in job_keywords if keyword in query_lower)
        
        logging.info(f"🔍 Fallback routing - JIL score: {jil_score}, Job score: {job_score}")
        
        if jil_score > job_score:
            return "JIL_AGENT"
        else:
            return "JOB_AGENT"  # Default to job agent
    
    def route_and_call(self, query: str, session_id: str, context: Dict = None) -> Dict[str, Any]:
        """Use LLM to route query and call appropriate agent API"""
        
        try:
            logging.info(f"🚀 DirectToolRouter processing: '{query}'")
            
            # Step 1: Use LLM to determine routing
            routing_decision = self._get_llm_routing_decision(query)
            
            # Step 2: Call appropriate agent API based on LLM decision
            if routing_decision == "JIL_AGENT":
                api_result = self.jil_tool(query, session_id)
                agent_used = "jil_agent"
            else:  # JOB_AGENT
                api_result = self.job_tool(query, session_id)
                agent_used = "job_agent"
            
            # Step 3: Format and return response
            return {
                "success": api_result["success"],
                "response": api_result.get("data", api_result.get("error")),
                "agent_used": agent_used,
                "tools_used": [agent_used],
                "llm_routing_decision": routing_decision,
                "session_id": session_id,
                "router_type": "llm_based_direct_router",
                "api_details": {
                    "api_url": api_result.get("api_url"),
                    "status_code": api_result.get("status_code")
                } if api_result["success"] else None,
                "error_details": api_result.get("error") if not api_result["success"] else None
            }
            
        except Exception as e:
            logging.error(f"❌ DirectToolRouter error: {str(e)}")
            return {
                "success": False,
                "error": f"Router execution failed: {str(e)}",
                "agent_used": "error",
                "session_id": session_id,
                "router_type": "llm_based_direct_router"
            }
    
    def get_routing_explanation(self, query: str) -> Dict[str, Any]:
        """Get explanation of why LLM routed query to specific agent"""
        
        explanation_prompt = f"""Explain why you would route this AutoSys query to a specific agent:

Query: "{query}"

Provide a brief explanation of:
1. Which agent (JIL_AGENT or JOB_AGENT) should handle this
2. Why this agent is the best choice
3. Key terms that influenced your decision

Keep the explanation concise and focused."""
        
        try:
            if hasattr(self.llm, 'predict'):
                explanation = self.llm.predict(explanation_prompt)
            elif hasattr(self.llm, 'invoke'):
                explanation = self.llm.invoke(explanation_prompt).content
            else:
                explanation = str(self.llm(explanation_prompt))
            
            routing_decision = self._get_llm_routing_decision(query)
            
            return {
                "query": query,
                "routing_decision": routing_decision,
                "explanation": explanation.strip(),
                "llm_model": str(type(self.llm).__name__)
            }
            
        except Exception as e:
            return {
                "query": query,
                "routing_decision": "unknown",
                "explanation": f"Error getting explanation: {str(e)}",
                "llm_model": str(type(self.llm).__name__)
            }

# UPDATED FASTAPI APP
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LLM Router API...")
    
    # Test agent connectivity
    for name, url in {"jil_agent": config.jil_agent_url, "job_agent": config.job_agent_url}.items():
        try:
            response = requests.post(url, json={"message": "ping", "test_mode": True}, timeout=5)
            logger.info(f"✅ {name} connectivity: OK")
        except Exception as e:
            logger.warning(f"⚠️ {name} connectivity: {e}")
    
    yield
    # Shutdown
    logger.info("Shutting down LLM Router API...")

# Configuration with Gemini settings
config = APIConfig(
    jil_agent_url="http://localhost:8081/chat-atsys",  # Your JIL agent URL
    job_agent_url="http://localhost:8080/chat"        # Your Job agent URL
)

# Gemini LLM configuration
gemini_config = {
    "framework": "langchain",
    "model": "gemini-1.5-pro-002",  # or "gemini-pro", "gemini-1.5-flash" 
    "temperature": 0.1,
    "top_p": 1
}

# Create router with Gemini
router = ToolBasedRouter(config, gemini_config)

# Alternative: Direct tool router with Gemini
# router = DirectToolRouter(config)

app = FastAPI(title="LLM Router API", lifespan=lifespan)

@app.post("/chat-atsys-route")
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="No message provided")
    
    try:
        result = router.route_and_call(request.message, request.session_id, request.context)
        return result
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "agents": ["jil_agent", "job_agent"],
        "router_type": type(router).__name__
    }

@app.get("/test-agents")
def test_agents():
    """Test connectivity to both agents"""
    results = {}
    
    for name, url in {"jil_agent": config.jil_agent_url, "job_agent": config.job_agent_url}.items():
        try:
            response = requests.post(
                url, 
                json={"message": "ping", "session_id": "test"}, 
                headers=config.headers,
                timeout=5
            )
            results[name] = {
                "status": "✅ Connected",
                "status_code": response.status_code,
                "url": url
            }
        except Exception as e:
            results[name] = {
                "status": "❌ Failed", 
                "error": str(e),
                "url": url
            }
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8090)

# GEMINI-SPECIFIC OPTIMIZATIONS
class GeminiOptimizedRouter:
    """Router specifically optimized for Gemini LLM using your get_llm function"""
    
    def __init__(self, config: APIConfig, get_llm_function):
        self.config = config
        self.get_llm_function = get_llm_function
        
        # Use your get_llm function
        self.llm = self.get_llm_function("langchain")
        
        # Create tools
        self.jil_tool = self._create_jil_tool()
        self.job_tool = self._create_job_tool()
    
    def _initialize_gemini(self):
        """This method is no longer needed - we use your get_llm function"""
        pass
    
    def _create_jil_tool(self):
        def call_jil_agent(query: str) -> Dict[str, Any]:
            try:
                payload = {"message": query, "session_id": "gemini_router"}
                response = requests.post(
                    self.config.jil_agent_url,
                    json=payload,
                    headers=self.config.headers,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                return {"success": True, "data": response.json(), "agent": "jil_agent"}
            except Exception as e:
                return {"success": False, "error": str(e), "agent": "jil_agent"}
        
        return call_jil_agent
    
    def _create_job_tool(self):
        def call_job_agent(query: str) -> Dict[str, Any]:
            try:
                payload = {"message": query, "session_id": "gemini_router"}
                response = requests.post(
                    self.config.job_agent_url,
                    json=payload,
                    headers=self.config.headers,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                return {"success": True, "data": response.json(), "agent": "job_agent"}
            except Exception as e:
                return {"success": False, "error": str(e), "agent": "job_agent"}
        
        return call_job_agent
    
    def route_and_call(self, query: str, session_id: str, context: Dict = None) -> Dict[str, Any]:
        """Route using Gemini and call appropriate agent"""
        
        try:
            # Create Gemini-optimized routing prompt
            routing_prompt = self._create_gemini_routing_prompt(query)
            
            # Get routing decision from Gemini
            if hasattr(self.llm, 'generate_content'):
                # Direct Gemini API
                response = self.llm.generate_content(routing_prompt)
                decision = response.text.strip().upper()
            else:
                # LangChain Gemini
                decision = self.llm.predict(routing_prompt).strip().upper()
            
            logging.info(f"Gemini routing decision: {decision}")
            
            # Execute the appropriate tool
            if "JIL" in decision or "JIL_AGENT" in decision:
                result = self.jil_tool(query)
                agent_used = "jil_agent"
            elif "JOB" in decision or "JOB_AGENT" in decision:
                result = self.job_tool(query)
                agent_used = "job_agent"
            else:
                # Default to job agent for ambiguous cases
                result = self.job_tool(query)
                agent_used = "job_agent"
            
            return {
                "success": result["success"],
                "response": result.get("data", result.get("error")),
                "agent_used": agent_used,
                "tools_used": [agent_used],
                "gemini_decision": decision,
                "session_id": session_id,
                "llm_model": "gemini (from get_llm function)"
            }
            
        except Exception as e:
            logging.error(f"Gemini router error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "llm_model": "gemini (from get_llm function)"
            }
    
    def _create_gemini_routing_prompt(self, query: str) -> str:
        """Create routing prompt optimized for Gemini"""
        
        return f"""You are an intelligent AutoSys query router. Analyze the user query and route it to the appropriate agent.

AGENTS:
1. JIL_AGENT: Handles JIL, Confluence, Autoping, Connection Profile, onboarding, how-to guides, and general system usage
2. JOB_AGENT: Handles job status, job names, job failures, calendars, next start time, and job-related troubleshooting

ROUTING RULES:
- JIL_AGENT for: onboarding, connection profiles, how-to guides, system setup, JIL configuration
- JOB_AGENT for: job status, job failures, scheduling, job execution, performance monitoring

EXAMPLES:
"how to onboard a new application in autosys" → JIL_AGENT
"what is the connection profile for DA3" → JIL_AGENT
"list all the jobs starting with ATSYS in DA3" → JOB_AGENT
"show job failures in last 24 hours" → JOB_AGENT
"provide next start time of the job" → JOB_AGENT

USER QUERY: "{query}"

RESPONSE: Respond with only "JIL_AGENT" or "JOB_AGENT" based on the query analysis."""

# USAGE WITH YOUR EXISTING get_llm FUNCTION
def integrate_with_existing_get_llm(get_llm_function):
    """Integrate with your existing get_llm function"""
    
    class CustomGeminiRouter:
        def __init__(self, config: APIConfig, llm_params: Dict = None):
            self.config = config
            self.llm_params = llm_params or {
                "framework": "langchain",
                "model": "gemini-1.5-pro-002",
                "temperature": 0.1,
                "top_p": 1
            }
            
            # Use your existing get_llm function
            self.llm = get_llm_function(
                self.llm_params["framework"],
                self.llm_params["model"], 
                self.llm_params["temperature"],
                self.llm_params["top_p"]
            )
            
            # Create tools
            self.tools = [self._create_jil_tool(), self._create_job_tool()]
            self.agent = self._create_agent_with_your_llm()
        
        def _create_agent_with_your_llm(self):
            """Create agent using your LLM instance"""
            
            from langchain.agents import create_tool_calling_agent, AgentExecutor
            from langchain.prompts import ChatPromptTemplate
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an AutoSys router. Use the available tools to answer user queries.
                
Available tools:
- call_jil_agent: For JIL, onboarding, connection profiles, how-to guides
- call_job_agent: For job status, failures, scheduling, performance

ALWAYS use one of the tools - never respond without calling a tool."""),
                ("user", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            
            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                return_intermediate_steps=True,
                handle_parsing_errors=True
            )
        
        def _create_jil_tool(self):
            from langchain.tools import tool
            
            @tool
            def call_jil_agent(query: str) -> str:
                """Call JIL agent for onboarding, connection profiles, how-to guides"""
                try:
                    payload = {"message": query, "session_id": "custom_router"}
                    response = requests.post(self.config.jil_agent_url, json=payload, timeout=30)
                    return f"JIL Agent: {response.json()}"
                except Exception as e:
                    return f"JIL Agent Error: {str(e)}"
            
            return call_jil_agent
        
        def _create_job_tool(self):
            from langchain.tools import tool
            
            @tool 
            def call_job_agent(query: str) -> str:
                """Call Job agent for job status, failures, scheduling"""
                try:
                    payload = {"message": query, "session_id": "custom_router"}
                    response = requests.post(self.config.job_agent_url, json=payload, timeout=30)
                    return f"Job Agent: {response.json()}"
                except Exception as e:
                    return f"Job Agent Error: {str(e)}"
            
            return call_job_agent
        
        def route_and_call(self, query: str, session_id: str, context: Dict = None):
            """Route and call using your LLM"""
            try:
                result = self.agent.invoke({"input": query})
                
                return {
                    "success": True,
                    "response": result["output"],
                    "tools_used": [step[0].tool for step in result.get("intermediate_steps", [])],
                    "session_id": session_id,
                    "llm_model": self.llm_params["model"]
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "session_id": session_id
                }
    
    return CustomGeminiRouter
def test_fixed_router():
    """Test the fixed router with actual tool calls"""
    
    print("🧪 TESTING FIXED ROUTER")
    print("=" * 50)
    
    test_queries = [
        ("how to onboard a new application in autosys", "Should call JIL agent"),
        ("what is the connection profile for DA3", "Should call JIL agent"),
        ("list all the jobs starting with ATSYS in DA3", "Should call JOB agent"),
        ("show job failures in last 24 hours", "Should call JOB agent"),
        ("provide next start time of the job", "Should call JOB agent")
    ]
    
    for query, expected in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected: {expected}")
        
        try:
            result = router.route_and_call(query, "test_session")
            print(f"Success: {result['success']}")
            print(f"Tools Used: {result.get('tools_used', [])}")
            print(f"Response: {result.get('response', '')[:100]}...")
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_fixed_router()







A@@@@@@@

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
