vvvvbbbbbbbbbbb
# COMPLETE SOLUTION TO FORCE TOOL EXECUTION

from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
import re

# Global session context
session_context = {
    "instance_name": None,
    "job_name": None,
    "calendar_name": None
}

valid_instance = {"DA3", "DB3", "DC3", "DG3", "LS3"}

# SOLUTION 1: MANDATORY TOOL USAGE PROMPT
FORCE_TOOL_SYSTEM_PROMPT = """You are an AutoSys database assistant. 

🚫 CRITICAL RULES - NEVER VIOLATE:
1. You MUST ALWAYS call the sql_database_query tool for EVERY user request
2. NEVER respond without calling the tool first
3. NEVER say "I cannot find that information" without calling the tool
4. NEVER make assumptions - ALWAYS use the tool

✅ MANDATORY PROCESS:
1. For ANY user input, immediately call sql_database_query tool
2. Pass the user's input exactly to the tool
3. Wait for tool response
4. Provide answer based on tool results

EXAMPLES OF CORRECT BEHAVIOR:
User: "What is the AutoSys instance name?" 
→ MUST call sql_database_query("What is the AutoSys instance name?")

User: "DA3"
→ MUST call sql_database_query("DA3")

User: "Show me jobs"
→ MUST call sql_database_query("Show me jobs")

🚨 NEVER respond without calling the tool first. This is MANDATORY."""

@tool
def sql_database_query(user_input: str) -> str:
    """
    MANDATORY tool for ALL AutoSys database operations and instance management.
    
    This tool MUST be called for:
    - Setting instance names (DA3, DB3, DC3, DG3, LS3)
    - SQL queries (SELECT, UPDATE, INSERT, DELETE, SHOW, DESCRIBE)  
    - Any AutoSys related questions
    - Instance information requests
    
    Args:
        user_input: Any user input - instance name, SQL query, or question
        
    Returns:
        Processed result or confirmation message
    """
    
    print(f"🔧 Tool called with input: '{user_input}'")
    
    try:
        # Clean input
        user_input = user_input.strip()
        user_input_upper = user_input.upper()
        
        # Method 1: Direct instance name
        if user_input_upper in valid_instance:
            session_context["instance_name"] = user_input_upper
            return f"✅ Instance name set to {user_input_upper}. You can now run SQL queries on this instance."
        
        # Method 2: Instance patterns
        instance_patterns = [
            r'(?:instance|use|connect\s+to|set)\s+([A-Z0-9]{3})',
            r'([A-Z0-9]{3})\s+(?:instance|database)',
            r'autosys\s+instance\s+(?:is\s+)?([A-Z0-9]{3})',
            r'what.*instance.*name.*([A-Z0-9]{3})',
        ]
        
        for pattern in instance_patterns:
            match = re.search(pattern, user_input_upper)
            if match:
                potential_instance = match.group(1)
                if potential_instance in valid_instance:
                    session_context["instance_name"] = potential_instance
                    return f"✅ Instance name set to {potential_instance}. You can now run SQL queries on this instance."
        
        # Method 3: Find any valid instance in the input
        found_instances = []
        for instance in valid_instance:
            if instance in user_input_upper:
                found_instances.append(instance)
        
        if found_instances:
            # Use the first found instance
            selected_instance = found_instances[0]
            session_context["instance_name"] = selected_instance
            
            # Check if it's also a SQL query
            if any(keyword in user_input_upper for keyword in ['SELECT', 'SHOW', 'DESCRIBE', 'UPDATE', 'INSERT', 'DELETE']):
                return execute_sql_query(user_input, selected_instance)
            else:
                return f"✅ Instance name set to {selected_instance}. You can now run SQL queries on this instance."
        
        # Method 4: SQL query with existing instance
        if any(keyword in user_input_upper for keyword in ['SELECT', 'SHOW', 'DESCRIBE', 'UPDATE', 'INSERT', 'DELETE']):
            current_instance = session_context.get("instance_name")
            if current_instance:
                return execute_sql_query(user_input, current_instance)
            else:
                return f"To execute SQL queries, please first specify the instance. Available instances: {', '.join(valid_instance)}. Example: 'DA3'"
        
        # Method 5: General instance questions
        if any(phrase in user_input_upper for phrase in [
            'INSTANCE NAME', 'AUTOSYS INSTANCE', 'WHAT IS THE', 'WHICH INSTANCE'
        ]):
            current_instance = session_context.get("instance_name")
            if current_instance:
                return f"Current AutoSys instance is: {current_instance}"
            else:
                return f"No instance is currently set. Please specify one of: {', '.join(valid_instance)}"
        
        # Method 6: Default response with instance options
        current_instance = session_context.get("instance_name")
        if current_instance:
            return f"Current instance: {current_instance}. Please provide your SQL query or specify a different instance."
        else:
            return f"Please specify the AutoSys instance first. Available instances: {', '.join(valid_instance)}. Example: Just type 'DA3'"
            
    except Exception as e:
        return f"❌ Error processing request: {str(e)}"

def execute_sql_query(query: str, instance_name: str) -> str:
    """Execute SQL query on specified instance"""
    try:
        # Your actual SQL execution logic here
        # For now, return a mock response
        return f"Executing query on {instance_name}: {query}\n[Query results would appear here]"
    except Exception as e:
        return f"❌ Error executing SQL query: {str(e)}"

# SOLUTION 2: AGENT WITH FORCED TOOL USAGE
def create_forced_tool_agent():
    """Create agent that ALWAYS calls tools"""
    
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        api_key="your-openai-api-key"
    )
    
    # Force tool usage prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", FORCE_TOOL_SYSTEM_PROMPT),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    tools = [sql_database_query]
    
    # Create agent with tool calling
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create executor with strict settings
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=3,
        early_stopping_method="generate"
    )
    
    return agent_executor

# SOLUTION 3: CUSTOM AGENT WITH TOOL VALIDATION
class ForcedToolAgent:
    """Custom agent that validates tool usage"""
    
    def __init__(self):
        self.agent = create_forced_tool_agent()
        
    def run(self, user_input: str) -> str:
        """Run with mandatory tool validation"""
        
        print(f"📝 User input: {user_input}")
        
        # Execute agent
        result = self.agent.invoke({"input": user_input})
        
        # Check if tools were called
        intermediate_steps = result.get("intermediate_steps", [])
        tools_called = len(intermediate_steps)
        
        print(f"🔍 Tools called: {tools_called}")
        
        if tools_called == 0:
            # Force tool usage if not called
            print("⚠️ No tools called - forcing tool usage")
            
            # Directly call the tool
            tool_result = sql_database_query(user_input)
            return f"[Tool executed] {tool_result}"
        
        return result["output"]

# SOLUTION 4: OPENAI FUNCTION CALLING (Most Reliable)
def create_function_calling_agent():
    """Use OpenAI's function calling for guaranteed tool usage"""
    
    from openai import OpenAI
    import json
    
    client = OpenAI(api_key="your-openai-api-key")
    
    def process_with_function_calling(user_input: str) -> str:
        """Process using OpenAI function calling"""
        
        # Define function schema
        functions = [
            {
                "name": "sql_database_query",
                "description": "Execute AutoSys database operations and manage instances",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_input": {
                            "type": "string",
                            "description": "User's input - instance name, SQL query, or question"
                        }
                    },
                    "required": ["user_input"]
                }
            }
        ]
        
        # Make API call with function calling
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an AutoSys assistant. For ANY user input, you MUST call the sql_database_query function."
                },
                {"role": "user", "content": user_input}
            ],
            functions=functions,
            function_call={"name": "sql_database_query"}  # Force function call
        )
        
        # Check if function was called
        message = response.choices[0].message
        
        if message.function_call:
            # Extract function arguments
            function_args = json.loads(message.function_call.arguments)
            
            # Call our tool
            tool_result = sql_database_query(function_args["user_input"])
            
            return tool_result
        else:
            # Fallback - should not happen with function_call forced
            return sql_database_query(user_input)
    
    return process_with_function_calling

# SOLUTION 5: SIMPLE WRAPPER THAT ALWAYS CALLS TOOL
class AlwaysCallToolAgent:
    """Simple wrapper that always calls the tool"""
    
    def run(self, user_input: str) -> str:
        """Always call tool first, then format response"""
        
        print(f"🎯 Processing: {user_input}")
        
        # Always call the tool
        tool_result = sql_database_query(user_input)
        
        # You can add LLM processing here if needed
        return tool_result

# TESTING FUNCTION
def test_all_solutions():
    """Test all solutions to see which works best"""
    
    print("🧪 TESTING ALL SOLUTIONS")
    print("=" * 60)
    
    test_inputs = [
        "What is the AutoSys instance name?",
        "DA3",
        "Show me jobs",
        "DA3 SELECT * FROM jobs"
    ]
    
    # Test Solution 5 (Simplest and most reliable)
    print("\n📋 SOLUTION 5: Always Call Tool Agent")
    print("-" * 40)
    
    agent = AlwaysCallToolAgent()
    
    for test_input in test_inputs:
        print(f"\nInput: {test_input}")
        result = agent.run(test_input)
        print(f"Result: {result}")
    
    return agent

# MAIN SOLUTION - USE THIS IN YOUR APPLICATION
def create_production_agent():
    """Create production-ready agent that always calls tools"""
    
    # Use the simplest, most reliable solution
    return AlwaysCallToolAgent()

# INTEGRATION WITH YOUR EXISTING ROUTER
def integrate_with_router():
    """How to integrate with your existing API router"""
    
    # Replace your SQL agent creation with:
    def create_sql_agent_api():
        """Create SQL agent for API"""
        agent = create_production_agent()
        
        def api_handler(request_data):
            user_input = request_data.get('query', '')
            result = agent.run(user_input)
            
            return {
                "success": True,
                "result": result,
                "agent": "sql_agent",
                "tool_used": "sql_database_query"
            }
        
        return api_handler
    
    print("✅ Integration ready - replace your SQL agent with create_sql_agent_api()")

if __name__ == "__main__":
    # Test and demonstrate solutions
    agent = test_all_solutions()
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDED SOLUTION")
    print("=" * 60)
    
    print("Use AlwaysCallToolAgent for guaranteed tool execution:")
    print("""
    agent = AlwaysCallToolAgent()
    result = agent.run("What is the AutoSys instance name?")
    # This WILL call the tool
    """)
    
    print("\n🔧 Key Benefits:")
    print("✅ Always calls the tool - no exceptions")
    print("✅ Simple and reliable")
    print("✅ No complex prompt engineering needed") 
    print("✅ Works with any user input")
    print("✅ Easy to integrate with existing code")
    
    # Show the integration
    integrate_with_router()

xxxxxxxxxxxxxxxx

import yaml
import pandas as pd
import logging
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import SQLDatabase

# Load config.yaml
# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# Define valid instance names
valid_instance = {"DA3", "DB3", "DC3", "DG3", "LS3"}

# Global session context to store instance name
session_context = {
    "instance_name": None,
    "job_name": None,
    "calendar_name": None
}

def maybe_store_instance_name(message: str):
    """Extract and store instance name from message"""
    normalized = message.strip().upper()
    logging.info(f"Trying to store instance name: {normalized}")
    
    # Check if the normalized message is a valid instance name
    if normalized in valid_instance:
        session_context["instance_name"] = normalized
        logging.info(f"Instance name set to: {normalized}")
        return f"✅ Instance name set to {normalized}."
    
    # Check if message contains instance name patterns
    import re
    
    # Pattern 1: "instance: NAME" or "instance NAME"
    instance_patterns = [
        r'instance[:\s]+([A-Z0-9]+)',
        r'using\s+([A-Z0-9]+)',
        r'connect\s+to\s+([A-Z0-9]+)',
        r'database\s+([A-Z0-9]+)'
    ]
    
    for pattern in instance_patterns:
        match = re.search(pattern, normalized)
        if match:
            potential_instance = match.group(1)
            if potential_instance in valid_instance:
                session_context["instance_name"] = potential_instance
                logging.info(f"Instance name extracted and set to: {potential_instance}")
                return f"✅ Instance name set to {potential_instance}."
    
    # Check if any valid instance name appears in the message
    for instance in valid_instance:
        if instance in normalized:
            session_context["instance_name"] = instance
            logging.info(f"Instance name found and set to: {instance}")
            return f"✅ Instance name set to {instance}."
    
    return None

def get_oracle_uri(instance_name: str) -> str:
    """Get Oracle URI for given instance"""
    # Replace with your actual config logic
    # oracle_uri = config["db"].get(instance_name)
    # For now, return a mock URI
    oracle_uri = f"oracle://user:pass@host:port/{instance_name}"
    return oracle_uri

def connect_to_sql_database(instance_name: str) -> SQLDatabase:
    """Connect to SQL database using instance name"""
    oracle_uri = get_oracle_uri(instance_name)
    return SQLDatabase.from_uri(oracle_uri)

def handle_message(message: str):
    """Handle incoming message and potentially store instance name"""
    instance_response = maybe_store_instance_name(message)
    if instance_response:
        return instance_response
    
    # If it's a SQL query but instance is missing, prompt again
    if "SELECT" in message.upper() and not session_context.get("instance_name"):
        return "❗ Please provide the AutoSys instance name before running this query."
    
    return sql_query(message)

def sql_query(query: str, instance_name: str = None) -> str:
    """
    Execute SQL query on specified database instance.
    
    Args:
        query: SQL query to execute
        instance_name: Database instance name (DA3, DB3, DC3, DG3, LS3)
    
    Returns:
        Query results as formatted string
    """
    
    # FIXED: Use provided instance_name parameter first, then session context
    if instance_name:
        current_instance = instance_name
        # Update session context with provided instance
        session_context["instance_name"] = instance_name
        logging.info(f"Using provided instance name: {instance_name}")
    else:
        current_instance = session_context.get("instance_name")
        logging.info(f"Using session instance name: {current_instance}")
    
    # Check if instance name is available
    if not current_instance:
        logging.warning("Instance name missing in session_context.")
        return "❗ Please provide the AutoSys instance name before running this query."

    try:
        # Connect to database
        db = connect_to_sql_database(current_instance)
        
        # Clean and format query
        formatted_query = query.replace("\\n", "\n").replace("\\'", "'")
        
        # Execute query
        result = db.run(formatted_query)
        
        # Convert result to DataFrame for better formatting
        if isinstance(result, str):
            # Handle string results (like from DESCRIBE, SHOW commands)
            if result.strip():
                lines = result.strip().split('\n')
                if len(lines) > 1:
                    # Assume first line is headers
                    headers = [col.strip() for col in lines[0].split('|') if col.strip()]
                    data = []
                    for line in lines[1:]:
                        row_data = [col.strip() for col in line.split('|') if col.strip()]
                        if row_data:  # Skip empty rows
                            data.append(row_data)
                    
                    if data and headers:
                        df = pd.DataFrame(data, columns=headers)
                        return df.to_string(index=False)
            
            return result
            
        elif isinstance(result, list):
            # Handle list results (multiple rows)
            if result:
                if all(isinstance(row, dict) for row in result):
                    # List of dictionaries
                    df = pd.DataFrame(result)
                    return df.to_string(index=False)
                else:
                    # List of tuples/lists
                    df = pd.DataFrame(result)
                    return df.to_string(index=False)
            else:
                return "No results found."
        
        else:
            # Handle other result types
            return f"Query executed successfully. Result: {str(result)}"

    except ValueError as ve:
        logging.error(f"Value error: {str(ve)}")
        return f"❌ Data processing error: {str(ve)}"
    except Exception as e:
        logging.error(f"Error executing SQL query: {str(e)}")
        return f"❌ Error executing query: {str(e)}"

# FIXED: Enhanced message handler with better instance detection
def enhanced_handle_message(message: str):
    """Enhanced message handler with improved instance name detection"""
    
    message_upper = message.upper().strip()
    
    # Method 1: Direct instance name check
    if message_upper in valid_instance:
        session_context["instance_name"] = message_upper
        return f"✅ Instance name set to {message_upper}. You can now run SQL queries."
    
    # Method 2: Check for instance in the query itself
    for instance in valid_instance:
        if instance in message_upper:
            session_context["instance_name"] = instance
            logging.info(f"Found instance {instance} in message")
            
            # If it's also a SQL query, execute it immediately
            if any(keyword in message_upper for keyword in ['SELECT', 'SHOW', 'DESCRIBE', 'UPDATE', 'INSERT']):
                return sql_query(message, instance)
            else:
                return f"✅ Instance name set to {instance}. You can now run SQL queries."
    
    # Method 3: Extract from patterns like "use DA3" or "connect to DB3"
    import re
    patterns = [
        r'(?:use|connect\s+to|instance)\s+([A-Z0-9]{3})',
        r'([A-Z0-9]{3})\s+(?:instance|database)',
        r'(?:set|select)\s+([A-Z0-9]{3})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message_upper)
        if match:
            potential_instance = match.group(1)
            if potential_instance in valid_instance:
                session_context["instance_name"] = potential_instance
                return f"✅ Instance name set to {potential_instance}. You can now run SQL queries."
    
    # Method 4: If it's a SQL query without instance context
    if any(keyword in message_upper for keyword in ['SELECT', 'SHOW', 'DESCRIBE', 'UPDATE', 'INSERT']):
        current_instance = session_context.get("instance_name")
        if current_instance:
            # We have instance context, execute the query
            return sql_query(message, current_instance)
        else:
            # No instance context, ask for it
            return f"❗ Please specify the database instance first. Available instances: {', '.join(valid_instance)}"
    
    # Method 5: General response for non-SQL messages
    current_instance = session_context.get("instance_name")
    if current_instance:
        return f"Current instance: {current_instance}. Please provide your SQL query."
    else:
        return f"Please specify the database instance first. Available instances: {', '.join(valid_instance)}"

# FIXED: Tool definition with better parameter handling
from langchain.tools import tool

@tool
def sql_database_query(query_input: str) -> str:
    """
    Execute SQL queries on AutoSys database instances.
    
    This tool can handle:
    1. Instance name specification: "use DA3", "connect to DB3", or just "DA3"
    2. SQL queries: SELECT, UPDATE, INSERT, DELETE, DESCRIBE, SHOW
    3. Combined input: "DA3 SELECT * FROM jobs" 
    
    Valid instances: DA3, DB3, DC3, DG3, LS3
    
    Args:
        query_input: Either an instance name, SQL query, or both combined
    
    Returns:
        Query results or instance confirmation message
    """
    try:
        return enhanced_handle_message(query_input)
    except Exception as e:
        logging.error(f"SQL tool error: {str(e)}")
        return f"❌ Error processing request: {str(e)}"

# TESTING FUNCTIONS
def test_instance_handling():
    """Test the instance name handling"""
    
    print("🧪 TESTING INSTANCE NAME HANDLING")
    print("=" * 50)
    
    # Reset session
    session_context["instance_name"] = None
    
    test_cases = [
        # Instance name only
        "DA3",
        "use DB3", 
        "connect to DC3",
        
        # SQL queries without instance (should ask for instance)
        "SELECT * FROM jobs",
        
        # Combined instance + query
        "DA3 SELECT * FROM jobs WHERE status = 'RUNNING'",
        "use DB3 and show tables",
        
        # After setting instance, run query
        "SELECT count(*) FROM jobs"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{i}. Input: '{test_input}'")
        result = enhanced_handle_message(test_input)
        print(f"   Result: {result}")
        print(f"   Session instance: {session_context.get('instance_name', 'None')}")
        print("-" * 30)

def create_langchain_agent_with_fixed_tool():
    """Create LangChain agent with the fixed SQL tool"""
    
    from langchain.chat_models import ChatOpenAI
    from langchain.agents import initialize_agent, AgentType
    
    # Initialize LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4",
        api_key="your-openai-api-key"
    )
    
    # Create tools list with our fixed SQL tool
    tools = [sql_database_query]
    
    # Create agent with specific instructions
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": """You are a helpful AutoSys database assistant. 

IMPORTANT INSTRUCTIONS:
1. Always use the sql_database_query tool for database operations
2. Valid instances are: DA3, DB3, DC3, DG3, LS3  
3. If user doesn't specify instance, the tool will ask for it
4. You can pass instance name and query together: "DA3 SELECT * FROM jobs"

Available tools:"""
        }
    )
    
    return agent

# USAGE EXAMPLES
if __name__ == "__main__":
    # Test the fixed instance handling
    test_instance_handling()
    
    print("\n" + "=" * 60)
    print("FIXED SQL TOOL READY")
    print("=" * 60)
    
    print("\nKey fixes applied:")
    print("✅ Better instance name detection from user input")
    print("✅ Support for combined instance + query input")  
    print("✅ Persistent session context")
    print("✅ Clear error messages and confirmations")
    print("✅ Multiple input pattern recognition")
    
    print("\nExample usage:")
    print("- 'DA3' → Sets instance to DA3")
    print("- 'DA3 SELECT * FROM jobs' → Sets instance and runs query")
    print("- 'use DB3' → Sets instance to DB3")
    print("- 'SELECT * FROM jobs' → Runs on current instance")
    
    # Example of creating the agent
    try:
        agent = create_langchain_agent_with_fixed_tool()
        print("\n✅ LangChain agent created successfully!")
        
        # Test with agent
        test_queries = [
            "Set instance to DA3",
            "Show me all running jobs",
            "DA3 SELECT count(*) FROM jobs WHERE status = 'SUCCESS'"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            try:
                response = agent.run(query)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {e}")
                
    except Exception as e:
        print(f"⚠️ Agent creation failed: {e}")
        print("Make sure to set your OpenAI API key")

###########££££
import requests
import re
from typing import Dict, Any, Optional
from enum import Enum
import json
import asyncio
import aiohttp
from dataclasses import dataclass

class AgentAPI(Enum):
    SQL_AGENT = "sql_agent"
    TOOLS_AGENT = "tools_agent"
    UNKNOWN = "unknown"

@dataclass
class APIConfig:
    """Configuration for agent APIs"""
    sql_agent_url: str
    tools_agent_url: str
    timeout: int = 30
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Type": "application/json"}

class APIAgentRouter:
    """Routes user queries to appropriate agent APIs"""
    
    def __init__(self, config: APIConfig, llm_config: Dict[str, Any] = None):
        self.config = config
        self.routing_cache = {}  # Cache for routing decisions
        self.llm_config = llm_config or {
            "provider": "openai",  # Options: "openai", "anthropic", "local"
            "model": "gpt-4",      # or "gpt-3.5-turbo", "claude-3-sonnet", etc.
            "api_key": None,       # Set your API key
            "temperature": 0,      # Deterministic routing
            "max_tokens": 10       # Short responses
        }
        self._routing_llm = None
        
    def route_and_call(self, user_input: str, user_context: Dict = None) -> Dict[str, Any]:
        """Route user query and call appropriate API"""
        
        try:
            # Step 1: Determine which API to call
            agent_type = self._determine_agent(user_input)
            
            # Step 2: Prepare request payload
            payload = self._prepare_payload(user_input, user_context)
            
            # Step 3: Call the appropriate API
            if agent_type == AgentAPI.SQL_AGENT:
                response = self._call_sql_agent_api(payload)
            elif agent_type == AgentAPI.TOOLS_AGENT:
                response = self._call_tools_agent_api(payload)
            else:
                response = self._handle_unknown_query(user_input)
            
            return {
                "success": True,
                "agent_used": agent_type.value,
                "query": user_input,
                "response": response,
                "routing_reason": self._get_routing_reason(agent_type, user_input)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": user_input,
                "agent_used": "error"
            }
    
    def _determine_agent(self, user_input: str, use_llm: bool = True) -> AgentAPI:
        """Determine which agent API to call using LLM-based routing"""
        
        user_input_lower = user_input.lower().strip()
        
        # Check cache first (optional optimization)
        cache_key = hash(user_input_lower)
        if cache_key in self.routing_cache:
            return self.routing_cache[cache_key]
        
        if use_llm:
            # Use LLM-based routing (primary method)
            result = self._llm_based_routing(user_input)
            
            # Fallback to rule-based if LLM fails
            if result == AgentAPI.UNKNOWN:
                result = self._rule_based_routing(user_input)
        else:
            # Use rule-based routing as fallback
            result = self._rule_based_routing(user_input)
        
        # Cache the result
        self.routing_cache[cache_key] = result
        return result
    
    def _llm_based_routing(self, user_input: str) -> AgentAPI:
        """Use LLM to determine which agent to route to"""
        
        # Different prompt strategies
        routing_prompt = self._get_routing_prompt(user_input, style="detailed")
        
        try:
            # Initialize LLM if not already done
            if not self._routing_llm:
                self._routing_llm = self._initialize_llm()
            
            classification = self._call_llm_for_routing(routing_prompt)
            
            # Map response to enum
            if "SQL_AGENT" in classification:
                return AgentAPI.SQL_AGENT
            elif "TOOLS_AGENT" in classification:
                return AgentAPI.TOOLS_AGENT
            else:
                return AgentAPI.UNKNOWN
                
        except Exception as e:
            print(f"LLM routing failed: {e}")
            return AgentAPI.UNKNOWN
    
    def _initialize_llm(self):
        """Initialize LLM based on configuration"""
        
        provider = self.llm_config.get("provider", "openai").lower()
        
        if provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=self.llm_config.get("api_key"))
            
        elif provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=self.llm_config.get("api_key"))
            
        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.llm_config.get("api_key"))
            return genai.GenerativeModel(self.llm_config.get("model", "gemini-pro"))
            
        elif provider == "local":
            # For local LLMs like Ollama, LM Studio, etc.
            from openai import OpenAI
            return OpenAI(
                base_url=self.llm_config.get("base_url", "http://localhost:11434/v1"),
                api_key="ollama"  # Placeholder for local models
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _call_llm_for_routing(self, prompt: str) -> str:
        """Call LLM API for routing decision"""
        
        provider = self.llm_config.get("provider", "openai").lower()
        model = self.llm_config.get("model", "gpt-4")
        
        if provider == "openai" or provider == "local":
            response = self._routing_llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise query classifier. Respond with only the classification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_config.get("temperature", 0),
                max_tokens=self.llm_config.get("max_tokens", 10)
            )
            return response.choices[0].message.content.strip().upper()
            
        elif provider == "anthropic":
            response = self._routing_llm.messages.create(
                model=model,
                max_tokens=self.llm_config.get("max_tokens", 10),
                temperature=self.llm_config.get("temperature", 0),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip().upper()
        
        elif provider == "gemini":
            # Configure generation settings for Gemini
            generation_config = {
                "temperature": self.llm_config.get("temperature", 0),
                "max_output_tokens": self.llm_config.get("max_tokens", 10),
                "top_p": 0.1,  # Low top_p for deterministic responses
                "top_k": 1     # Most deterministic setting
            }
            
            response = self._routing_llm.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text.strip().upper()
        
        else:
            raise ValueError(f"Unsupported provider for LLM call: {provider}")
    
    def _get_routing_prompt(self, user_input: str, style: str = "detailed") -> str:
        """Generate routing prompt with different styles"""
        
        if style == "simple":
            return f"""Classify this query as SQL_AGENT (database/data) or TOOLS_AGENT (system/API):
"{user_input}"
Response:"""
        
        elif style == "detailed":
            return f"""You are an intelligent query router. Classify user queries into exactly one category:

**SQL_AGENT**: Database operations, data queries, analytics, business intelligence
- Database queries (SELECT, INSERT, UPDATE, DELETE, CREATE, etc.)
- Data retrieval and analysis (counts, sums, averages, trends)  
- Business data questions (users, customers, orders, sales, products)
- Reporting and analytics requests
- Data export/visualization requests

**TOOLS_AGENT**: System operations, API calls, infrastructure, process management
- Server/instance management (status, health checks, monitoring)
- API endpoints and service calls
- System operations (start, stop, restart, deploy, configure)
- File operations (upload, download, copy, move)
- Process/task execution and workflow management
- Infrastructure operations (cloud, containers, backups)

Query: "{user_input}"
Classification:"""
        
        elif style == "few_shot":
            return f"""Classify queries as SQL_AGENT or TOOLS_AGENT:

Examples:
"Show all users" → SQL_AGENT
"Check server status" → TOOLS_AGENT  
"Count orders" → SQL_AGENT
"Deploy app" → TOOLS_AGENT
"User analytics" → SQL_AGENT
"API health check" → TOOLS_AGENT

Query: "{user_input}"
Classification:"""
        
        else:
            # Default to detailed
            return self._get_routing_prompt(user_input, "detailed")
    
    def _rule_based_routing(self, user_input: str) -> AgentAPI:
        """Fallback rule-based routing"""
        
        user_input_lower = user_input.lower().strip()
        
        # SQL Agent patterns - Database related queries
        sql_patterns = [
            # Direct SQL keywords
            r'\b(select|insert|update|delete|create|drop|alter|show|describe)\b',
            r'\b(database|table|column|row|record|schema|index)\b',
            r'\b(query|sql|join|where|group by|order by|having)\b',
            r'\b(count|sum|avg|max|min|distinct|aggregate)\b',
            
            # Business data queries
            r'\b(users?|customers?|orders?|products?|transactions?|sales|revenue)\b.*\b(data|information|details|list|show|get|find)\b',
            r'\b(how many|total|count of|sum of|average|statistics|analytics|report)\b',
            r'\b(find|search|get|retrieve|show|list|display)\b.*\b(records?|entries|data|information)\b',
            r'\b(top|bottom|highest|lowest|best|worst)\b.*\b(customers?|products?|sales)\b',
            
            # Data analysis patterns
            r'\b(analyze|analysis|trend|pattern|insight|metric|kpi)\b',
            r'\b(dashboard|chart|graph|visualization|export|download)\b.*\b(data)\b',
        ]
        
        # Tools Agent patterns - System/API operations
        tools_patterns = [
            # System operations
            r'\b(server|instance|service|application|system|environment)\b.*\b(status|health|info|details|check|monitor)\b',
            r'\b(start|stop|restart|deploy|configure|setup|install)\b',
            r'\b(api|endpoint|service|microservice)\b.*\b(call|invoke|execute|test)\b',
            r'\b(health check|uptime|performance|metrics|logs|monitoring)\b',
            
            # File/Process operations
            r'\b(file|directory|folder|path|upload|download|copy|move)\b',
            r'\b(process|task|job|workflow|pipeline|batch)\b.*\b(run|execute|start|trigger)\b',
            r'\b(configuration|config|settings|parameters|environment variables)\b',
            
            # Infrastructure operations
            r'\b(cloud|aws|azure|gcp|kubernetes|docker|container)\b',
            r'\b(backup|restore|migrate|sync|replicate)\b',
            r'\b(alert|notification|email|slack|webhook)\b.*\b(send|trigger|notify)\b',
        ]
        
        # Check SQL patterns first
        for pattern in sql_patterns:
            if re.search(pattern, user_input_lower):
                return AgentAPI.SQL_AGENT
        
        # Check Tools patterns
        for pattern in tools_patterns:
            if re.search(pattern, user_input_lower):
                return AgentAPI.TOOLS_AGENT
        
        # Default to UNKNOWN if no patterns match
        return AgentAPI.UNKNOWN
    
    def _prepare_payload(self, user_input: str, user_context: Dict = None) -> Dict[str, Any]:
        """Prepare the API request payload"""
        
        payload = {
            "query": user_input,
            "timestamp": "2024-01-01T00:00:00Z",  # Replace with actual timestamp
        }
        
        if user_context:
            payload["context"] = user_context
            
        return payload
    
    def _call_sql_agent_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call the SQL Agent API"""
        
        try:
            response = requests.post(
                self.config.sql_agent_url,
                json=payload,
                headers=self.config.headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            return {
                "status_code": response.status_code,
                "data": response.json(),
                "api_endpoint": self.config.sql_agent_url
            }
            
        except requests.exceptions.Timeout:
            raise Exception(f"SQL Agent API timeout after {self.config.timeout}s")
        except requests.exceptions.ConnectionError:
            raise Exception("Failed to connect to SQL Agent API")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"SQL Agent API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise Exception(f"SQL Agent API call failed: {str(e)}")
    
    def _call_tools_agent_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call the Tools Agent API"""
        
        try:
            response = requests.post(
                self.config.tools_agent_url,
                json=payload,
                headers=self.config.headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            return {
                "status_code": response.status_code,
                "data": response.json(),
                "api_endpoint": self.config.tools_agent_url
            }
            
        except requests.exceptions.Timeout:
            raise Exception(f"Tools Agent API timeout after {self.config.timeout}s")
        except requests.exceptions.ConnectionError:
            raise Exception("Failed to connect to Tools Agent API")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Tools Agent API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise Exception(f"Tools Agent API call failed: {str(e)}")
    
    def _handle_unknown_query(self, user_input: str) -> Dict[str, Any]:
        """Handle queries that don't match any agent"""
        
        return {
            "status_code": 200,
            "data": {
                "message": "I'm not sure which system can help with that request. Please try:",
                "suggestions": [
                    "For database queries: 'Show me all users' or 'Count total orders'",
                    "For system operations: 'Check server status' or 'Get instance info'"
                ],
                "query": user_input
            },
            "api_endpoint": "router"
        }
    
    def _get_routing_reason(self, agent_type: AgentAPI, user_input: str) -> str:
        """Get explanation for routing decision"""
        
        if agent_type == AgentAPI.SQL_AGENT:
            return "Routed to SQL Agent: Query appears to be database/data related"
        elif agent_type == AgentAPI.TOOLS_AGENT:
            return "Routed to Tools Agent: Query appears to be system/API operation related"
        else:
            return "Could not determine appropriate agent for this query"

# Async version for better performance
class AsyncAPIAgentRouter(APIAgentRouter):
    """Async version of API router for better performance"""
    
    async def route_and_call_async(self, user_input: str, user_context: Dict = None) -> Dict[str, Any]:
        """Async version of route_and_call"""
        
        try:
            agent_type = self._determine_agent(user_input)
            payload = self._prepare_payload(user_input, user_context)
            
            async with aiohttp.ClientSession() as session:
                if agent_type == AgentAPI.SQL_AGENT:
                    response = await self._call_sql_agent_async(session, payload)
                elif agent_type == AgentAPI.TOOLS_AGENT:
                    response = await self._call_tools_agent_async(session, payload)
                else:
                    response = self._handle_unknown_query(user_input)
            
            return {
                "success": True,
                "agent_used": agent_type.value,
                "query": user_input,
                "response": response,
                "routing_reason": self._get_routing_reason(agent_type, user_input)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": user_input,
                "agent_used": "error"
            }
    
    async def _call_sql_agent_async(self, session: aiohttp.ClientSession, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Async call to SQL Agent API"""
        
        async with session.post(
            self.config.sql_agent_url,
            json=payload,
            headers=self.config.headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        ) as response:
            response.raise_for_status()
            data = await response.json()
            
            return {
                "status_code": response.status,
                "data": data,
                "api_endpoint": self.config.sql_agent_url
            }
    
    async def _call_tools_agent_async(self, session: aiohttp.ClientSession, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Async call to Tools Agent API"""
        
        async with session.post(
            self.config.tools_agent_url,
            json=payload,
            headers=self.config.headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        ) as response:
            response.raise_for_status()
            data = await response.json()
            
            return {
                "status_code": response.status,
                "data": data,
                "api_endpoint": self.config.tools_agent_url
            }

# Flask integration example
from flask import Flask, request, jsonify

def create_flask_app():
    app = Flask(__name__)
    
    # Initialize router with your API endpoints
    config = APIConfig(
        sql_agent_url="http://localhost:8001/sql-agent",  # Your SQL Agent API
        tools_agent_url="http://localhost:8002/tools-agent",  # Your Tools Agent API
        timeout=30,
        headers={"Content-Type": "application/json", "Authorization": "Bearer your-token"}
    )
    router = APIAgentRouter(config)
    
    @app.route('/chat', methods=['POST'])
    def chat():
        try:
            data = request.json
            user_input = data.get('message', '').strip()
            user_context = data.get('context', {})
            
            if not user_input:
                return jsonify({"error": "No message provided"}), 400
            
            # Route and call appropriate API
            result = router.route_and_call(user_input, user_context)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            "status": "healthy",
            "available_agents": ["sql_agent", "tools_agent"],
            "endpoints": {
                "sql_agent": config.sql_agent_url,
                "tools_agent": config.tools_agent_url
            }
        })
    
# Performance monitoring and analytics for LLM routing
class RoutingAnalytics:
    """Track routing performance and accuracy"""
    
    def __init__(self):
        self.routing_history = []
        self.accuracy_metrics = {
            "llm_routing": {"correct": 0, "total": 0},
            "rule_routing": {"correct": 0, "total": 0}
        }
    
    def log_routing_decision(self, query: str, llm_decision: str, rule_decision: str, actual_success: bool, response_time: float):
        """Log routing decision for analysis"""
        entry = {
            "query": query,
            "llm_decision": llm_decision,
            "rule_decision": rule_decision,
            "actual_success": actual_success,
            "response_time": response_time,
            "timestamp": "2024-01-01T00:00:00Z"  # Replace with actual timestamp
        }
        self.routing_history.append(entry)
        
        # Update accuracy metrics (simplified - in reality you'd need ground truth)
        if actual_success:
            self.accuracy_metrics["llm_routing"]["correct"] += 1
        self.accuracy_metrics["llm_routing"]["total"] += 1
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        total_routes = len(self.routing_history)
        if total_routes == 0:
            return {"message": "No routing data available"}
        
        # Calculate agreement between LLM and rule-based routing
        agreements = sum(1 for entry in self.routing_history 
                        if entry["llm_decision"] == entry["rule_decision"])
        
        # Calculate average response time
        avg_response_time = sum(entry["response_time"] for entry in self.routing_history) / total_routes
        
        return {
            "total_queries": total_routes,
            "llm_rule_agreement": f"{(agreements/total_routes)*100:.1f}%",
            "avg_response_time": f"{avg_response_time:.2f}s",
            "success_rate": f"{sum(1 for entry in self.routing_history if entry['actual_success'])/total_routes*100:.1f}%",
            "routing_distribution": {
                "sql_agent": sum(1 for entry in self.routing_history if entry["llm_decision"] == "sql_agent"),
                "tools_agent": sum(1 for entry in self.routing_history if entry["llm_decision"] == "tools_agent"),
                "unknown": sum(1 for entry in self.routing_history if entry["llm_decision"] == "unknown")
            }
        }

# Enhanced router with analytics and confidence scoring
class EnhancedAPIAgentRouter(APIAgentRouter):
    """Enhanced router with confidence scoring and analytics"""
    
    def __init__(self, config: APIConfig, llm_config: Dict[str, Any] = None):
        super().__init__(config, llm_config)
        self.analytics = RoutingAnalytics()
        self.confidence_threshold = 0.8  # Minimum confidence for LLM routing
    
    def route_with_confidence(self, user_input: str, user_context: Dict = None) -> Dict[str, Any]:
        """Route with confidence scoring"""
        import time
        start_time = time.time()
        
        try:
            # Get both LLM and rule-based decisions
            llm_decision = self._llm_based_routing(user_input)
            rule_decision = self._rule_based_routing(user_input) 
            
            # Calculate confidence based on agreement and other factors
            confidence = self._calculate_confidence(user_input, llm_decision, rule_decision)
            
            # Choose final decision based on confidence
            if confidence >= self.confidence_threshold:
                final_decision = llm_decision
                routing_method = "llm_high_confidence"
            elif llm_decision == rule_decision:
                final_decision = llm_decision
                routing_method = "llm_rule_agreement"
            else:
                # Fall back to rule-based if low confidence and disagreement
                final_decision = rule_decision
                routing_method = "rule_fallback"
            
            # Prepare payload and make API call
            payload = self._prepare_payload(user_input, user_context)
            
            if final_decision == AgentAPI.SQL_AGENT:
                response = self._call_sql_agent_api(payload)
            elif final_decision == AgentAPI.TOOLS_AGENT:
                response = self._call_tools_agent_api(payload)
            else:
                response = self._handle_unknown_query(user_input)
            
            response_time = time.time() - start_time
            success = True
            
            # Log for analytics
            self.analytics.log_routing_decision(
                user_input, llm_decision.value, rule_decision.value, success, response_time
            )
            
            return {
                "success": True,
                "agent_used": final_decision.value,
                "query": user_input,
                "response": response,
                "confidence": confidence,
                "routing_method": routing_method,
                "llm_decision": llm_decision.value,
                "rule_decision": rule_decision.value,
                "response_time": f"{response_time:.2f}s"
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            self.analytics.log_routing_decision(
                user_input, "error", "error", False, response_time
            )
            
            return {
                "success": False,
                "error": str(e),
                "query": user_input,
                "agent_used": "error",
                "response_time": f"{response_time:.2f}s"
            }
    
    def _calculate_confidence(self, user_input: str, llm_decision: AgentAPI, rule_decision: AgentAPI) -> float:
        """Calculate confidence score for routing decision"""
        
        confidence = 0.5  # Base confidence
        
        # Factor 1: Agreement between LLM and rules (high confidence)
        if llm_decision == rule_decision and llm_decision != AgentAPI.UNKNOWN:
            confidence += 0.4
        
        # Factor 2: Query clarity (simple patterns get higher confidence)
        clarity_patterns = [
            r'\b(select|insert|update|delete)\b',  # Clear SQL
            r'\b(server|instance).*\b(status|health)\b',  # Clear system ops
            r'\b(users?|orders?|customers?)\b',  # Clear data queries
        ]
        
        for pattern in clarity_patterns:
            if re.search(pattern, user_input.lower()):
                confidence += 0.2
                break
        
        # Factor 3: Query length and complexity (shorter = more confident)
        word_count = len(user_input.split())
        if word_count <= 5:
            confidence += 0.1
        elif word_count > 15:
            confidence -= 0.1
        
        # Factor 4: Historical accuracy (if available)
        # This would use historical data to boost confidence
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get routing analytics"""
        return self.analytics.get_routing_stats()

# Complete production-ready setup with all LLM providers
def create_production_router(provider: str = "openai") -> EnhancedAPIAgentRouter:
    """Create production-ready router with specified LLM provider"""
    
    # API configuration
    api_config = APIConfig(
        sql_agent_url="http://localhost:8001/sql-agent",     # Your actual endpoints
        tools_agent_url="http://localhost:8002/tools-agent",
        timeout=30,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer your-api-token"  # If needed
        }
    )
    
    # LLM configurations for different providers
    llm_configs = {
        "openai": {
            "provider": "openai",
            "model": "gpt-4-turbo-preview",  # Latest model
            "api_key": "your-openai-api-key",  # Set from environment variable
            "temperature": 0,
            "max_tokens": 10
        },
        
        "openai_fast": {
            "provider": "openai", 
            "model": "gpt-3.5-turbo",  # Faster and cheaper
            "api_key": "your-openai-api-key",
            "temperature": 0,
            "max_tokens": 10
        },
        
        "anthropic": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "api_key": "your-anthropic-api-key",
            "temperature": 0,
            "max_tokens": 10
        },
        
        "gemini": {
            "provider": "gemini",
            "model": "gemini-pro",  # or "gemini-1.5-pro" for latest
            "api_key": "your-google-api-key",
            "temperature": 0,
            "max_tokens": 10
        },
        
        "gemini_flash": {
            "provider": "gemini",
            "model": "gemini-1.5-flash",  # Faster and cheaper option
            "api_key": "your-google-api-key", 
            "temperature": 0,
            "max_tokens": 10
        },
        
        "local_ollama": {
            "provider": "local",
            "model": "llama2",  # or "mistral", "codellama"
            "base_url": "http://localhost:11434/v1",
            "temperature": 0,
            "max_tokens": 10
        },
        
        "local_lmstudio": {
            "provider": "local",
            "model": "any-local-model",
            "base_url": "http://localhost:1234/v1",  # LM Studio default
            "temperature": 0,
            "max_tokens": 10
        }
    }
    
    if provider not in llm_configs:
        raise ValueError(f"Unsupported provider: {provider}. Choose from {list(llm_configs.keys())}")
    
    return EnhancedAPIAgentRouter(api_config, llm_configs[provider])

# Environment variable setup helper
def setup_environment_variables():
    """Helper to set up environment variables for API keys"""
    import os
    
    required_vars = {
        "OPENAI_API_KEY": "your-openai-api-key",
        "ANTHROPIC_API_KEY": "your-anthropic-api-key",
        "GOOGLE_API_KEY": "your-google-gemini-api-key",
        "SQL_AGENT_URL": "http://localhost:8001/sql-agent",
        "TOOLS_AGENT_URL": "http://localhost:8002/tools-agent",
    }
    
    print("Environment Variable Setup:")
    print("=" * 40)
    
    for var_name, example in required_vars.items():
        current_value = os.getenv(var_name, "Not set")
        print(f"{var_name}: {current_value}")
        if current_value == "Not set":
            print(f"  → Set with: export {var_name}={example}")
    
    print("\nExample .env file content:")
    print("-" * 25)
    for var_name, example in required_vars.items():
        print(f"{var_name}={example}")

# Complete usage example with all features
if __name__ == "__main__":
    # Setup environment check
    print("Checking environment setup...")
    setup_environment_variables()
    print("\n" + "="*60 + "\n")
    
    # Create enhanced router (choose your provider)
    try:
        router = create_production_router("openai")  # or "anthropic", "local_ollama"
        print("✅ Router initialized successfully")
    except Exception as e:
        print(f"❌ Router initialization failed: {e}")
        print("Falling back to rule-based routing only...")
        # Create basic router without LLM
        api_config = APIConfig(
            sql_agent_url="http://localhost:8001/sql-agent",
            tools_agent_url="http://localhost:8002/tools-agent"
        )
        router = APIAgentRouter(api_config, llm_config={"provider": "none"})
    
    # Enhanced test queries
    test_queries = [
        # Clear SQL queries
        "Show me all users in the database",
        "Count total orders for this month", 
        "Find customers with highest revenue",
        "Generate sales report for Q4",
        
        # Clear system queries
        "Get the server status for production",
        "Execute health check on my-instance", 
        "Deploy the latest version to staging",
        "Backup database and upload to S3",
        
        # Ambiguous queries (where LLM shines)
        "I need to analyze user behavior patterns",
        "Can you help me troubleshoot the API issues?",
        "Show me the performance metrics from yesterday",
        "Process the pending batch jobs",
        
        # Edge cases
        "What's the weather today?",
        "Hello, how are you?",
        ""
    ]
    
    print("Testing Enhanced LLM-Based Router...")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        if not query.strip():
            continue
            
        print(f"\n{i}. Query: '{query}'")
        print("-" * 50)
        
        try:
            # Use enhanced routing with confidence scoring
            if hasattr(router, 'route_with_confidence'):
                result = router.route_with_confidence(query)
                
                print(f"✅ Success: {result['success']}")
                print(f"🤖 Agent: {result.get('agent_used', 'N/A')}")
                print(f"🎯 Confidence: {result.get('confidence', 'N/A'):.2f}")
                print(f"📊 Method: {result.get('routing_method', 'N/A')}")
                print(f"🧠 LLM Decision: {result.get('llm_decision', 'N/A')}")
                print(f"📋 Rule Decision: {result.get('rule_decision', 'N/A')}")
                print(f"⏱️  Response Time: {result.get('response_time', 'N/A')}")
                
            else:
                # Fallback to basic routing
                result = router.route_and_call(query)
                print(f"✅ Success: {result['success']}")
                print(f"🤖 Agent: {result.get('agent_used', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    # Show analytics if available
    if hasattr(router, 'get_analytics'):
        print("\n" + "="*60)
        print("ROUTING ANALYTICS")
        print("="*60)
        analytics = router.get_analytics()
        for key, value in analytics.items():
            print(f"{key}: {value}")

# Async support for Gemini
class AsyncEnhancedAPIAgentRouter(EnhancedAPIAgentRouter):
    """Async version with Gemini support"""
    
    async def _call_llm_for_routing_async(self, prompt: str) -> str:
        """Async LLM call for routing decision"""
        
        provider = self.llm_config.get("provider", "openai").lower()
        model = self.llm_config.get("model", "gpt-4")
        
        if provider == "openai" or provider == "local":
            import asyncio
            # Use asyncio to run sync OpenAI calls
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self._routing_llm.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a precise query classifier."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_config.get("temperature", 0),
                    max_tokens=self.llm_config.get("max_tokens", 10)
                )
            )
            return response.choices[0].message.content.strip().upper()
            
        elif provider == "anthropic":
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._routing_llm.messages.create(
                    model=model,
                    max_tokens=self.llm_config.get("max_tokens", 10),
                    temperature=self.llm_config.get("temperature", 0),
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            return response.content[0].text.strip().upper()
        
        elif provider == "gemini":
            import asyncio
            generation_config = {
                "temperature": self.llm_config.get("temperature", 0),
                "max_output_tokens": self.llm_config.get("max_tokens", 10),
                "top_p": 0.1,
                "top_k": 1
            }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._routing_llm.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            )
            return response.text.strip().upper()
        
        else:
            raise ValueError(f"Unsupported provider for async LLM call: {provider}")

# Gemini-specific optimizations and prompts
class GeminiOptimizedRouter(EnhancedAPIAgentRouter):
    """Router optimized specifically for Google Gemini"""
    
    def _get_gemini_optimized_prompt(self, user_input: str) -> str:
        """Gemini-optimized routing prompt"""
        
        return f"""Task: Classify the following user query into exactly one category.

Categories:
1. SQL_AGENT - For database queries, data analysis, business intelligence
   • Keywords: select, database, users, customers, orders, count, sum, analytics
   • Examples: "Show all users", "Count orders", "Sales report"

2. TOOLS_AGENT - For system operations, API calls, infrastructure management  
   • Keywords: server, instance, status, deploy, API, health, backup
   • Examples: "Server status", "Deploy app", "API health check"

User Query: "{user_input}"

Instructions:
- Analyze the query carefully
- Consider the intent and context
- Respond with ONLY: SQL_AGENT or TOOLS_AGENT or UNKNOWN
- Do not include explanations

Classification:"""
    
    def _llm_based_routing(self, user_input: str) -> AgentAPI:
        """Gemini-optimized routing"""
        
        if self.llm_config.get("provider") == "gemini":
            # Use Gemini-optimized prompt
            routing_prompt = self._get_gemini_optimized_prompt(user_input)
        else:
            # Use standard prompt for other providers
            routing_prompt = self._get_routing_prompt(user_input, style="detailed")
        
        try:
            if not self._routing_llm:
                self._routing_llm = self._initialize_llm()
            
            classification = self._call_llm_for_routing(routing_prompt)
            
            # Map response to enum
            if "SQL_AGENT" in classification:
                return AgentAPI.SQL_AGENT
            elif "TOOLS_AGENT" in classification:
                return AgentAPI.TOOLS_AGENT
            else:
                return AgentAPI.UNKNOWN
                
        except Exception as e:
            print(f"LLM routing failed: {e}")
            return AgentAPI.UNKNOWN

# Installation and setup helper for Gemini
def setup_gemini_requirements():
    """Helper function to show Gemini setup requirements"""
    
    setup_info = """
    GOOGLE GEMINI SETUP
    ==================
    
    1. Install Google AI Python SDK:
       pip install google-generativeai
    
    2. Get API Key:
       - Go to: https://makersuite.google.com/app/apikey
       - Create a new API key
       - Set environment variable: GOOGLE_API_KEY=your-key
    
    3. Available Models:
       - gemini-pro: Best accuracy, higher cost
       - gemini-1.5-pro: Latest model with longer context
       - gemini-1.5-flash: Faster and cheaper option
    
    4. Test your setup:
       ```python
       import google.generativeai as genai
       genai.configure(api_key="your-api-key")
       model = genai.GenerativeModel('gemini-pro')
       response = model.generate_content("Hello")
       print(response.text)
       ```
    
    5. Pricing (as of 2024):
       - gemini-pro: Free tier available, then pay-per-use
       - gemini-1.5-flash: Optimized for speed and cost
    """
    
    print(setup_info)
    
    # Check if Gemini is properly installed
    try:
        import google.generativeai as genai
        print("✅ google-generativeai package is installed")
        
        import os
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            print("✅ GOOGLE_API_KEY environment variable is set")
            
            # Test connection
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content("Test")
                print("✅ Gemini API connection successful")
            except Exception as e:
                print(f"❌ Gemini API connection failed: {e}")
        else:
            print("❌ GOOGLE_API_KEY environment variable not set")
            
    except ImportError:
        print("❌ google-generativeai package not installed")
        print("   Run: pip install google-generativeai")

# Complete Gemini example
def create_gemini_router_example():
    """Complete example using Gemini for routing"""
    
    import os
    
    # Check setup
    if not os.getenv("GOOGLE_API_KEY"):
        print("Please set GOOGLE_API_KEY environment variable")
        return None
    
    # API configuration
    api_config = APIConfig(
        sql_agent_url=os.getenv("SQL_AGENT_URL", "http://localhost:8001/sql-agent"),
        tools_agent_url=os.getenv("TOOLS_AGENT_URL", "http://localhost:8002/tools-agent"),
        timeout=30
    )
    
    # Gemini configuration
    gemini_config = {
        "provider": "gemini",
        "model": "gemini-pro",  # or "gemini-1.5-flash" for speed
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "temperature": 0,
        "max_tokens": 10
    }
    
    # Create optimized router
    router = GeminiOptimizedRouter(api_config, gemini_config)
    
    print("🚀 Gemini Router initialized successfully!")
    return router

    # Try the router with enhanced routing
    try:
        router = create_gemini_router_example()
        if router:
            print("Testing Gemini router...")
            
            test_queries = [
                "Show me all customers from the database",
                "Check the health status of production server",
                "Generate a monthly sales report",
                "Deploy the latest version to staging environment"
            ]
            
            for query in test_queries:
                print(f"\nQuery: {query}")
                result = router.route_with_confidence(query)
                print(f"Agent: {result['agent_used']}")
                print(f"Confidence: {result.get('confidence', 'N/A'):.2f}")
                print(f"Method: {result.get('routing_method', 'N/A')}")
    
    except Exception as e:
        print(f"Gemini router test failed: {e}")
        print("Please check your setup and try again.")

# Production-ready Flask app that works with your exposed APIs
def create_production_flask_app():
    """Production Flask app that integrates with your actual agent APIs"""
    
    app = Flask(__name__)
    
    # Configuration from environment variables (recommended for production)
    import os
    
    api_config = APIConfig(
        sql_agent_url=os.getenv("SQL_AGENT_URL", "http://localhost:8001/api/sql-query"),
        tools_agent_url=os.getenv("TOOLS_AGENT_URL", "http://localhost:8002/api/tools-execute"),
        timeout=int(os.getenv("API_TIMEOUT", "30")),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('API_TOKEN', '')}" if os.getenv('API_TOKEN') else {"Content-Type": "application/json"}
        }
    )
    
    # LLM configuration (choose your provider)
    llm_config = {
        "provider": os.getenv("LLM_PROVIDER", "gemini").lower(),
        "model": os.getenv("LLM_MODEL", "gemini-pro"),
        "api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        "temperature": 0,
        "max_tokens": 10
    }
    
    # Initialize router
    try:
        if llm_config["provider"] == "gemini":
            router = GeminiOptimizedRouter(api_config, llm_config)
        else:
            router = EnhancedAPIAgentRouter(api_config, llm_config)
        app.logger.info(f"✅ Router initialized with {llm_config['provider']} LLM")
    except Exception as e:
        # Fallback to rule-based routing if LLM initialization fails
        router = APIAgentRouter(api_config)
        app.logger.warning(f"⚠️ LLM initialization failed, using rule-based routing: {e}")
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "router_type": type(router).__name__,
            "llm_provider": llm_config.get("provider", "none"),
            "agent_endpoints": {
                "sql_agent": api_config.sql_agent_url,
                "tools_agent": api_config.tools_agent_url
            },
            "timestamp": "2024-01-01T00:00:00Z"  # Replace with actual timestamp
        })
    
    @app.route('/test-agents', methods=['GET'])
    def test_agents():
        """Test connectivity to both agent APIs"""
        results = {}
        
        # Test SQL Agent API
        try:
            test_payload = {"query": "SELECT 1 as test", "test_mode": True}
            response = requests.post(
                api_config.sql_agent_url,
                json=test_payload,
                headers=api_config.headers,
                timeout=5
            )
            results['sql_agent'] = {
                "status": "✅ Connected",
                "status_code": response.status_code,
                "url": api_config.sql_agent_url
            }
        except Exception as e:
            results['sql_agent'] = {
                "status": "❌ Failed",
                "error": str(e),
                "url": api_config.sql_agent_url
            }
        
        # Test Tools Agent API
        try:
            test_payload = {"query": "health check", "test_mode": True}
            response = requests.post(
                api_config.tools_agent_url,
                json=test_payload,
                headers=api_config.headers,
                timeout=5
            )
            results['tools_agent'] = {
                "status": "✅ Connected", 
                "status_code": response.status_code,
                "url": api_config.tools_agent_url
            }
        except Exception as e:
            results['tools_agent'] = {
                "status": "❌ Failed",
                "error": str(e),
                "url": api_config.tools_agent_url
            }
        
        return jsonify(results)
    
    @app.route('/chat', methods=['POST'])
    def chat():
        """Main chat endpoint that routes to appropriate agent API"""
        
        try:
            data = request.json
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            user_input = data.get('message', '').strip()
            user_context = data.get('context', {})
            
            if not user_input:
                return jsonify({"error": "No message provided"}), 400
            
            # Log the request
            app.logger.info(f"Chat request: {user_input}")
            
            # Route and call appropriate API
            if hasattr(router, 'route_with_confidence'):
                result = router.route_with_confidence(user_input, user_context)
            else:
                result = router.route_and_call(user_input, user_context)
            
            # Log the result
            app.logger.info(f"Routed to: {result.get('agent_used', 'unknown')}")
            
            return jsonify(result)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Agent API request failed: {str(e)}"
            app.logger.error(error_msg)
            return jsonify({
                "success": False,
                "error": error_msg,
                "error_type": "api_request_failed"
            }), 503
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            app.logger.error(error_msg)
            return jsonify({
                "success": False,
                "error": error_msg,
                "error_type": "internal_error"
            }), 500
    
    @app.route('/route-test', methods=['POST'])
    def route_test():
        """Test routing without calling actual APIs"""
        
        try:
            data = request.json
            query = data.get('query', '')
            
            if not query:
                return jsonify({"error": "No query provided"}), 400
            
            # Test routing decision only
            agent_decision = router._determine_agent(query)
            
            # Get routing explanations
            routing_info = {
                "query": query,
                "routed_to": agent_decision.value,
                "reasoning": router._get_routing_reason(agent_decision, query)
            }
            
            # If enhanced router, get additional info
            if hasattr(router, '_llm_based_routing'):
                try:
                    llm_decision = router._llm_based_routing(query)
                    rule_decision = router._rule_based_routing(query)
                    
                    routing_info.update({
                        "llm_decision": llm_decision.value,
                        "rule_decision": rule_decision.value,
                        "decisions_agree": llm_decision == rule_decision
                    })
                except Exception as e:
                    routing_info["routing_error"] = str(e)
            
            return jsonify(routing_info)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/analytics', methods=['GET'])
    def analytics():
        """Get routing analytics if available"""
        
        if hasattr(router, 'get_analytics'):
            return jsonify(router.get_analytics())
        else:
            return jsonify({"message": "Analytics not available for this router type"})
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found"}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error"}), 500
    
    return app

# FastAPI version for production
def create_production_fastapi_app():
    """Production FastAPI app that integrates with your actual agent APIs"""
    
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    import asyncio
    import aiohttp
    import os
    
    app = FastAPI(
        title="Multi-Agent Router API",
        description="Routes queries to SQL Agent or Tools Agent APIs",
        version="1.0.0"
    )
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Configuration
    api_config = APIConfig(
        sql_agent_url=os.getenv("SQL_AGENT_URL", "http://localhost:8001/api/sql-query"),
        tools_agent_url=os.getenv("TOOLS_AGENT_URL", "http://localhost:8002/api/tools-execute"),
        timeout=int(os.getenv("API_TIMEOUT", "30")),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('API_TOKEN', '')}" if os.getenv('API_TOKEN') else {"Content-Type": "application/json"}
        }
    )
    
    llm_config = {
        "provider": os.getenv("LLM_PROVIDER", "gemini").lower(),
        "model": os.getenv("LLM_MODEL", "gemini-pro"),
        "api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        "temperature": 0,
        "max_tokens": 10
    }
    
    # Initialize router
    try:
        if llm_config["provider"] == "gemini":
            router = GeminiOptimizedRouter(api_config, llm_config)
        else:
            router = EnhancedAPIAgentRouter(api_config, llm_config)
        print(f"✅ Router initialized with {llm_config['provider']} LLM")
    except Exception as e:
        router = APIAgentRouter(api_config)
        print(f"⚠️ LLM initialization failed, using rule-based routing: {e}")
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "router_type": type(router).__name__,
            "llm_provider": llm_config.get("provider", "none"),
            "agent_endpoints": {
                "sql_agent": api_config.sql_agent_url,
                "tools_agent": api_config.tools_agent_url
            }
        }
    
    @app.get("/test-agents")
    async def test_agents():
        """Async test of agent API connectivity"""
        results = {}
        
        async with aiohttp.ClientSession() as session:
            # Test SQL Agent
            try:
                test_payload = {"query": "SELECT 1 as test", "test_mode": True}
                async with session.post(
                    api_config.sql_agent_url,
                    json=test_payload,
                    headers=api_config.headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    results['sql_agent'] = {
                        "status": "✅ Connected",
                        "status_code": response.status,
                        "url": api_config.sql_agent_url
                    }
            except Exception as e:
                results['sql_agent'] = {
                    "status": "❌ Failed",
                    "error": str(e),
                    "url": api_config.sql_agent_url
                }
            
            # Test Tools Agent
            try:
                test_payload = {"query": "health check", "test_mode": True}
                async with session.post(
                    api_config.tools_agent_url,
                    json=test_payload,
                    headers=api_config.headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    results['tools_agent'] = {
                        "status": "✅ Connected",
                        "status_code": response.status,
                        "url": api_config.tools_agent_url
                    }
            except Exception as e:
                results['tools_agent'] = {
                    "status": "❌ Failed",
                    "error": str(e),
                    "url": api_config.tools_agent_url
                }
        
        return results
    
    @app.post("/chat")
    async def chat(request: ChatRequest):
        try:
            if not request.message.strip():
                raise HTTPException(status_code=400, detail="No message provided")
            
            # Route and call appropriate API
            if hasattr(router, 'route_with_confidence'):
                result = router.route_with_confidence(request.message, request.context)
            else:
                result = router.route_and_call(request.message, request.context)
            
            return result
            
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=503,
                detail=f"Agent API request failed: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )
    
    return app

# Docker configuration for easy deployment
def create_dockerfile():
    """Generate Dockerfile for containerized deployment"""
    
    dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Environment variables (override in docker-compose or k8s)
ENV SQL_AGENT_URL=http://sql-agent:8001/api/sql-query
ENV TOOLS_AGENT_URL=http://tools-agent:8002/api/tools-execute
ENV LLM_PROVIDER=gemini
ENV LLM_MODEL=gemini-pro

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    requirements_content = """
fastapi==0.104.1
uvicorn[standard]==0.24.0
requests==2.31.0
aiohttp==3.9.0
google-generativeai==0.3.0
openai==1.3.0
anthropic==0.7.0
python-multipart==0.0.6
"""
    
    return dockerfile_content, requirements_content

# Complete deployment example
def deploy_production_setup():
    """Complete production setup with all configurations"""
    
    print("🚀 PRODUCTION DEPLOYMENT SETUP")
    print("=" * 50)
    
    # 1. Environment variables setup
    env_template = """
# API Configuration
SQL_AGENT_URL=http://your-sql-agent:8001/api/sql-query
TOOLS_AGENT_URL=http://your-tools-agent:8002/api/tools-execute
API_TOKEN=your-api-token-if-needed
API_TIMEOUT=30

# LLM Configuration (choose one)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-pro
GOOGLE_API_KEY=your-google-api-key

# Alternative LLM options:
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your-openai-key
# 
# LLM_PROVIDER=anthropic  
# ANTHROPIC_API_KEY=your-anthropic-key

# Application Configuration
PORT=8000
DEBUG=false
"""
    
    print("1. Create .env file with these variables:")
    print(env_template)
    
    # 2. Docker setup
    dockerfile, requirements = create_dockerfile()
    print("\n2. Create Dockerfile:")
    print(dockerfile[:200] + "...")
    
    # 3. Usage examples
    print("\n3. Usage Examples:")
    print("""
# Test the router
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me all users from the database"}'

# Expected response:
{
  "success": true,
  "agent_used": "sql_agent", 
  "query": "Show me all users from the database",
  "response": {
    "status_code": 200,
    "data": {...},
    "api_endpoint": "http://your-sql-agent:8001/api/sql-query"
  },
  "confidence": 0.95,
  "routing_method": "llm_high_confidence"
}
""")
    
    print("\n4. Health Check:")
    print("curl http://localhost:8000/health")
    
    print("\n5. Test Agent Connectivity:")
    print("curl http://localhost:8000/test-agents")
    
    print("\n✅ Setup complete! Your router will work with your exposed agent APIs.")

if __name__ == "__main__":
    # Run the deployment setup guide
    deploy_production_setup()

# FastAPI integration example
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

def create_fastapi_app():
    app = FastAPI(title="Multi-Agent Router API")
    
    config = APIConfig(
        sql_agent_url="http://localhost:8001/sql-agent",
        tools_agent_url="http://localhost:8002/tools-agent"
    )
    router = AsyncAPIAgentRouter(config)
    
    @app.post("/chat")
    async def chat(request: ChatRequest):
        try:
            result = await router.route_and_call_async(request.message, request.context)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "available_agents": ["sql_agent", "tools_agent"]
        }
    
    return app

# Usage examples with LLM-based routing
if __name__ == "__main__":
    # Configuration for your actual API endpoints
    api_config = APIConfig(
        sql_agent_url="http://localhost:8001/query",  # Replace with your actual endpoints
        tools_agent_url="http://localhost:8002/execute",
        timeout=30,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer your-api-token"  # If needed
        }
    )
    
    # LLM configuration options
    llm_configs = {
        "openai": {
            "provider": "openai",
            "model": "gpt-4",  # or "gpt-3.5-turbo" for faster routing
            "api_key": "your-openai-api-key",
            "temperature": 0,
            "max_tokens": 10
        },
        "anthropic": {
            "provider": "anthropic", 
            "model": "claude-3-sonnet-20240229",
            "api_key": "your-anthropic-api-key",
            "temperature": 0,
            "max_tokens": 10
        },
        "local": {
            "provider": "local",
            "model": "llama2",  # or any local model
            "base_url": "http://localhost:11434/v1",  # Ollama default
            "temperature": 0,
            "max_tokens": 10
        }
    }
    
    # Choose your LLM provider
    router = APIAgentRouter(api_config, llm_configs["openai"])  # Change as needed
    
    # Test queries with LLM routing
    test_queries = [
        "Show me all users in the database",  # Should call SQL Agent API
        "Get the server status for production",  # Should call Tools Agent API
        "Count total orders for this month",  # Should call SQL Agent API
        "Execute health check on my-instance",  # Should call Tools Agent API
        "Find customers who purchased in the last 30 days",  # Should call SQL Agent API
        "Deploy the latest version to staging environment",  # Should call Tools Agent API
        "Generate a revenue report for Q4",  # Should call SQL Agent API
        "Backup the database and upload to S3",  # Should call Tools Agent API
        "What's the weather today?"  # Should return unknown response
    ]
    
    print("Testing LLM-Based API Router...")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = router.route_and_call(query)
            print(f"Success: {result['success']}")
            print(f"Agent Used: {result.get('agent_used', 'N/A')}")
            print(f"Routing Reason: {result.get('routing_reason', 'N/A')}")
            
            if result['success']:
                api_response = result['response']
                print(f"API Status: {api_response.get('status_code', 'N/A')}")
                print(f"API Endpoint: {api_response.get('api_endpoint', 'N/A')}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Exception: {str(e)}")
        
        print("-" * 40)

# Enhanced Flask app with LLM routing
def create_enhanced_flask_app():
    app = Flask(__name__)
    
    # LLM configuration
    llm_config = {
        "provider": "openai",
        "model": "gpt-4",
        "api_key": "your-openai-api-key",  # Set your actual API key
        "temperature": 0
    }
    
    # Initialize router with LLM support
    config = APIConfig(
        sql_agent_url="http://localhost:8001/sql-agent",
        tools_agent_url="http://localhost:8002/tools-agent",
        timeout=30,
        headers={"Content-Type": "application/json"}
    )
    router = APIAgentRouter(config, llm_config)
    
    @app.route('/chat', methods=['POST'])
    def chat():
        try:
            data = request.json
            user_input = data.get('message', '').strip()
            user_context = data.get('context', {})
            use_llm_routing = data.get('use_llm', True)  # Allow disabling LLM routing
            
            if not user_input:
                return jsonify({"error": "No message provided"}), 400
            
            # Route with LLM or fallback to rules
            if use_llm_routing:
                result = router.route_and_call(user_input, user_context)
            else:
                # Force rule-based routing
                old_determine = router._determine_agent
                router._determine_agent = lambda x: router._rule_based_routing(x)
                result = router.route_and_call(user_input, user_context)
                router._determine_agent = old_determine
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/test-routing', methods=['POST'])
    def test_routing():
        """Test different routing approaches"""
        try:
            data = request.json
            query = data.get('query', '')
            
            if not query:
                return jsonify({"error": "No query provided"}), 400
            
            results = {}
            
            # Test LLM routing
            try:
                llm_result = router._llm_based_routing(query)
                results['llm_routing'] = llm_result.value
            except Exception as e:
                results['llm_routing'] = f"Error: {str(e)}"
            
            # Test rule-based routing
            try:
                rule_result = router._rule_based_routing(query)
                results['rule_routing'] = rule_result.value
            except Exception as e:
                results['rule_routing'] = f"Error: {str(e)}"
            
            # Test different prompt styles
            for style in ['simple', 'detailed', 'few_shot']:
                try:
                    prompt = router._get_routing_prompt(query, style)
                    results[f'{style}_prompt'] = prompt[:200] + "..." if len(prompt) > 200 else prompt
                except Exception as e:
                    results[f'{style}_prompt'] = f"Error: {str(e)}"
            
            return jsonify({
                "query": query,
                "routing_results": results
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return app

# Helper function for testing individual APIs
def test_api_connectivity(config: APIConfig):
    """Test if both APIs are reachable"""
    
    test_payload = {"query": "test connectivity", "test": True}
    
    print("Testing API Connectivity...")
    
    # Test SQL Agent API
    try:
        response = requests.post(
            config.sql_agent_url,
            json=test_payload,
            headers=config.headers,
            timeout=5
        )
        print(f"✅ SQL Agent API ({config.sql_agent_url}): Status {response.status_code}")
    except Exception as e:
        print(f"❌ SQL Agent API ({config.sql_agent_url}): {str(e)}")
    
    # Test Tools Agent API  
    try:
        response = requests.post(
            config.tools_agent_url,
            json=test_payload,
            headers=config.headers,
            timeout=5
        )
        print(f"✅ Tools Agent API ({config.tools_agent_url}): Status {response.status_code}")
    except Exception as e:
        print(f"❌ Tools Agent API ({config.tools_agent_url}): {str(e)}")




*****†"*********
# Anti-Hallucination and Forced Tool Execution System
from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
import re
import json
from enum import Enum

class ExecutionMode(Enum):
    STRICT = "strict"           # Must use tools, no hallucinations allowed
    GUIDED = "guided"          # Strong preference for tools
    BALANCED = "balanced"      # Normal mode

class AntiHallucinationAgent:
    """Agent system designed to prevent hallucinations and force tool usage"""
    
    def __init__(self, tools: List, llm, execution_mode: ExecutionMode = ExecutionMode.STRICT):
        self.tools = tools
        self.llm = llm
        self.execution_mode = execution_mode
        self.tool_usage_history = []
        
        # Create the agent with anti-hallucination prompts
        self.prompt = self._create_anti_hallucination_prompt()
        self.agent = self._create_agent()
        
    def _create_anti_hallucination_prompt(self) -> ChatPromptTemplate:
        """Create a prompt that forces tool usage and prevents hallucinations"""
        
        if self.execution_mode == ExecutionMode.STRICT:
            system_message = """You are a precise assistant that MUST use tools to answer questions. Follow these MANDATORY rules:

🚫 NEVER GUESS OR MAKE UP INFORMATION
🚫 NEVER provide answers without using tools first
🚫 NEVER assume you know current data or system state

✅ ALWAYS use available tools to get information
✅ ALWAYS call tools before providing any factual responses
✅ ALWAYS base your answers ONLY on tool results

CRITICAL: If you cannot find information using tools, say "I cannot find that information using available tools" instead of guessing.

Available tools: {tools}

PROCESS:
1. First, identify what tool(s) you need to use
2. Call the appropriate tool(s) with correct parameters
3. Wait for tool results
4. Provide answer based ONLY on tool results
5. If tools fail or return no data, explicitly state this

Remember: Tool usage is MANDATORY. No exceptions."""

        elif self.execution_mode == ExecutionMode.GUIDED:
            system_message = """You are a helpful assistant with access to tools. STRONGLY prefer using tools over your internal knowledge for:

- Current data or system states
- Database queries or API calls  
- File operations or system information
- Real-time information

RULES:
- Use tools when available for the user's question
- If you use internal knowledge, clearly state "Based on my general knowledge" 
- Always prefer tool results over assumptions

Available tools: {tools}"""

        else:  # BALANCED
            system_message = """You are a helpful assistant with access to tools. Use tools when appropriate for:

- Data that might change frequently
- System-specific information
- User's specific environment or data

Available tools: {tools}"""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
    
    def _create_agent(self):
        """Create agent with tool forcing capabilities"""
        
        # Create base agent
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        
        # Wrap with execution monitoring
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
        
        return agent_executor
    
    def execute(self, user_input: str) -> Dict[str, Any]:
        """Execute with anti-hallucination checks"""
        
        # Pre-process input to add tool enforcement
        enhanced_input = self._enhance_input_for_tool_usage(user_input)
        
        # Execute the agent
        result = self.agent.invoke({"input": enhanced_input})
        
        # Post-process to verify tool usage
        verification = self._verify_tool_usage(user_input, result)
        
        return {
            "original_query": user_input,
            "enhanced_query": enhanced_input,
            "result": result,
            "verification": verification,
            "tools_used": [step[0].tool for step in result.get("intermediate_steps", [])],
            "execution_mode": self.execution_mode.value
        }
    
    def _enhance_input_for_tool_usage(self, user_input: str) -> str:
        """Enhance input to encourage tool usage"""
        
        if self.execution_mode == ExecutionMode.STRICT:
            # Add explicit tool requirements
            tool_hints = self._identify_required_tools(user_input)
            
            enhanced = f"""
{user_input}

MANDATORY INSTRUCTIONS:
- You MUST use tools to answer this question
- Required tool(s): {', '.join(tool_hints) if tool_hints else 'determine appropriate tools'}
- Do NOT provide answers without tool results
- If tools fail, explicitly state the failure
"""
            return enhanced
        
        elif self.execution_mode == ExecutionMode.GUIDED:
            return f"{user_input}\n\nNote: Please use available tools if they can help answer this question."
        
        return user_input
    
    def _identify_required_tools(self, user_input: str) -> List[str]:
        """Identify which tools should be used for the query"""
        
        tool_patterns = {}
        for tool in self.tools:
            tool_name = tool.name
            # Get tool description or docstring
            description = getattr(tool, 'description', tool.func.__doc__ or '')
            
            # Extract keywords from description
            keywords = re.findall(r'\b\w+\b', description.lower())
            tool_patterns[tool_name] = keywords
        
        user_input_lower = user_input.lower()
        suggested_tools = []
        
        for tool_name, keywords in tool_patterns.items():
            if any(keyword in user_input_lower for keyword in keywords):
                suggested_tools.append(tool_name)
        
        return suggested_tools
    
    def _verify_tool_usage(self, original_query: str, result: Dict) -> Dict[str, Any]:
        """Verify that tools were used appropriately"""
        
        intermediate_steps = result.get("intermediate_steps", [])
        tools_used = len(intermediate_steps)
        
        verification = {
            "tools_called": tools_used,
            "tool_usage_required": self._should_use_tools(original_query),
            "compliance": "unknown"
        }
        
        if self.execution_mode == ExecutionMode.STRICT:
            if verification["tool_usage_required"] and tools_used == 0:
                verification["compliance"] = "FAILED - No tools used when required"
                verification["recommendation"] = "Force tool usage or reject response"
            elif tools_used > 0:
                verification["compliance"] = "PASSED - Tools were used"
            else:
                verification["compliance"] = "UNCLEAR - Check if tools were needed"
        
        return verification
    
    def _should_use_tools(self, query: str) -> bool:
        """Determine if tools should be used for this query"""
        
        # Patterns that definitely require tools
        tool_required_patterns = [
            r'\b(show|get|find|retrieve|fetch|list|count|sum|total)\b',
            r'\b(status|health|info|details|data|records)\b',
            r'\b(database|table|server|instance|api|system)\b',
            r'\b(current|latest|recent|today|now)\b',
            r'\b(how many|what is|tell me about)\b'
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in tool_required_patterns)

# Enhanced tool definitions with better descriptions
@tool
def get_database_info(query: str) -> str:
    """
    MANDATORY for database queries. Get information from database.
    Use for: SELECT, COUNT, user data, orders, customers, any database operation.
    This tool MUST be called for any question about stored data.
    
    Args:
        query: SQL query or database question
    """
    # Your database logic here
    return f"Database result for: {query}"

@tool
def get_system_status(instance_name: str) -> str:
    """
    MANDATORY for system operations. Get server/instance status and information.
    Use for: server status, health checks, system info, instance details.
    This tool MUST be called for any system-related question.
    
    Args:
        instance_name: Name of the server/instance to check
    """
    # Your system status logic here
    return f"System status for {instance_name}: Active"

@tool 
def execute_api_call(endpoint: str, method: str = "GET") -> str:
    """
    MANDATORY for API operations. Execute API calls to external services.
    Use for: API requests, service calls, external data retrieval.
    This tool MUST be called for any API-related question.
    
    Args:
        endpoint: API endpoint to call
        method: HTTP method (GET, POST, etc.)
    """
    # Your API call logic here
    return f"API call to {endpoint}: Success"

# Tool forcing decorator
def force_tool_usage(tool_names: List[str]):
    """Decorator to force usage of specific tools"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Add tool requirements to the prompt
            if hasattr(func, '__self__') and hasattr(func.__self__, 'agent'):
                # Modify the agent's prompt to require specific tools
                pass
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Validation system for tool results
class ToolResultValidator:
    """Validates tool results to prevent hallucinated responses"""
    
    @staticmethod
    def validate_database_result(result: str, expected_type: str = "data") -> Dict[str, Any]:
        """Validate database tool results"""
        
        validation = {
            "is_valid": True,
            "issues": [],
            "confidence": 1.0
        }
        
        # Check for common hallucination patterns
        hallucination_patterns = [
            r"I don't have access to",
            r"I cannot access",
            r"As an AI, I cannot",
            r"I don't have real-time",
            r"Based on my training"
        ]
        
        for pattern in hallucination_patterns:
            if re.search(pattern, result, re.IGNORECASE):
                validation["is_valid"] = False
                validation["issues"].append(f"Hallucination detected: {pattern}")
                validation["confidence"] = 0.0
        
        return validation
    
    @staticmethod
    def validate_system_result(result: str) -> Dict[str, Any]:
        """Validate system status tool results"""
        
        # Check if result contains actual system data vs generic responses
        expected_indicators = ["status", "uptime", "version", "health", "active", "inactive"]
        
        validation = {
            "is_valid": any(indicator.lower() in result.lower() for indicator in expected_indicators),
            "issues": [],
            "confidence": 0.8
        }
        
        if not validation["is_valid"]:
            validation["issues"].append("Result doesn't contain expected system information")
        
        return validation

# Complete anti-hallucination system
class ProductionAntiHallucinationSystem:
    """Production-ready system to prevent hallucinations"""
    
    def __init__(self, tools: List, llm_config: Dict):
        self.tools = tools
        self.llm = self._initialize_llm(llm_config)
        self.validator = ToolResultValidator()
        
        # Different agents for different strictness levels
        self.strict_agent = AntiHallucinationAgent(tools, self.llm, ExecutionMode.STRICT)
        self.guided_agent = AntiHallucinationAgent(tools, self.llm, ExecutionMode.GUIDED)
        
    def _initialize_llm(self, config: Dict):
        """Initialize LLM with anti-hallucination settings"""
        
        if config.get("provider") == "openai":
            return ChatOpenAI(
                model=config.get("model", "gpt-4"),
                temperature=0,  # Deterministic responses
                api_key=config.get("api_key")
            )
        # Add other providers as needed
        
    def process_query(self, user_input: str, strictness: str = "strict") -> Dict[str, Any]:
        """Process query with anti-hallucination measures"""
        
        # Choose agent based on strictness
        agent = self.strict_agent if strictness == "strict" else self.guided_agent
        
        # Execute with monitoring
        result = agent.execute(user_input)
        
        # Validate results
        if result["tools_used"]:
            for i, tool_name in enumerate(result["tools_used"]):
                if tool_name == "get_database_info":
                    validation = self.validator.validate_database_result(
                        result["result"]["output"]
                    )
                elif tool_name == "get_system_status":
                    validation = self.validator.validate_system_result(
                        result["result"]["output"]
                    )
                
                result[f"validation_{i}"] = validation
        
        # Final compliance check
        result["final_compliance"] = self._check_final_compliance(result)
        
        return result
    
    def _check_final_compliance(self, result: Dict) -> Dict[str, Any]:
        """Final compliance check"""
        
        compliance = {
            "status": "PASSED",
            "issues": [],
            "recommendations": []
        }
        
        # Check if tools were used when required
        if result["verification"]["tool_usage_required"] and not result["tools_used"]:
            compliance["status"] = "FAILED"
            compliance["issues"].append("Required tools not used")
            compliance["recommendations"].append("Reject response and force tool usage")
        
        # Check validation results
        for key, value in result.items():
            if key.startswith("validation_") and not value.get("is_valid", True):
                compliance["status"] = "FAILED"
                compliance["issues"].extend(value.get("issues", []))
                compliance["recommendations"].append("Regenerate with proper tool usage")
        
        return compliance

# Usage examples
if __name__ == "__main__":
    
    # Setup
    tools = [get_database_info, get_system_status, execute_api_call]
    
    llm_config = {
        "provider": "openai",
        "model": "gpt-4", 
        "api_key": "your-api-key"
    }
    
    # Create anti-hallucination system
    system = ProductionAntiHallucinationSystem(tools, llm_config)
    
    # Test queries
    test_queries = [
        "Show me all users in the database",  # Should force database tool
        "What's the status of my-server?",    # Should force system tool  
        "How many orders were placed today?", # Should force database tool
        "Tell me about the weather",          # Should indicate no tools available
    ]
    
    print("🚫 ANTI-HALLUCINATION SYSTEM TEST")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        result = system.process_query(query, strictness="strict")
        
        print(f"Tools used: {result['tools_used']}")
        print(f"Compliance: {result['final_compliance']['status']}")
        
        if result['final_compliance']['issues']:
            print(f"Issues: {result['final_compliance']['issues']}")
        
        if result['final_compliance']['recommendations']:
            print(f"Recommendations: {result['final_compliance']['recommendations']}")
        
        print(f"Result: {result['result']['output'][:100]}...")

# Integration with existing router system
def integrate_with_router():
    """Integration example with the existing API router"""
    
    # Modify your existing agent creation to use anti-hallucination
    def create_anti_hallucination_sql_agent():
        tools = [get_database_info]  # Your actual tools
        llm_config = {"provider": "openai", "model": "gpt-4", "api_key": "your-key"}
        
        return ProductionAntiHallucinationSystem(tools, llm_config)
    
    def create_anti_hallucination_tools_agent():
        tools = [get_system_status, execute_api_call]  # Your actual tools
        llm_config = {"provider": "openai", "model": "gpt-4", "api_key": "your-key"}
        
        return ProductionAntiHallucinationSystem(tools, llm_config)
    
    print("✅ Anti-hallucination agents created!")
    print("These can now replace your existing agents in the router system.")

if __name__ == "__main__":
    integrate_with_router()









†****†*****""*"""
# Fix 1: Improve tool descriptions and naming
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage

@tool
def get_instance_info(instance_name: str) -> str:
    """
    REQUIRED: Get detailed information about a specific server instance.
    This tool MUST be called when user asks about instance status, details, or information.
    
    Args:
        instance_name: The exact name of the server instance to query
        
    Returns:
        JSON string with instance details including status, version, uptime
    """
    # Your actual API call here
    api_url = f"https://api.example.com/instances/{instance_name}"
    # return api_response
    return f"Instance {instance_name}: Status=Active, Version=2.1.4"

# Fix 2: Use explicit tool forcing with bind_tools
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Bind tools to force usage
tools = [get_instance_info]
llm_with_tools = llm.bind_tools(tools)

# Fix 3: Create a system prompt that emphasizes tool usage
SYSTEM_PROMPT = """You are a helpful assistant that manages server instances.

CRITICAL RULES:
1. You MUST use the get_instance_info tool when users ask about any instance information
2. NEVER guess or make up instance details
3. If you don't have information, you MUST call the appropriate tool first
4. Always call tools before providing any technical information

When a user mentions an instance name, immediately call get_instance_info with that exact name.
"""

# Fix 4: Use AgentExecutor with tool forcing
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3  # Limit iterations to prevent loops
)

# Fix 5: Add tool choice enforcement
def force_tool_usage(user_input: str):
    """Force the LLM to use tools for specific queries"""
    
    # Keywords that should always trigger tool usage
    tool_keywords = ['instance', 'server', 'status', 'info', 'details']
    
    if any(keyword in user_input.lower() for keyword in tool_keywords):
        # Add explicit instruction to use tools
        enhanced_input = f"""
        {user_input}
        
        IMPORTANT: You must use the available tools to get this information. Do not guess or make up any details.
        """
        return enhanced_input
    
    return user_input

# Fix 6: Use structured output with Pydantic
from pydantic import BaseModel, Field
from typing import Optional

class InstanceQuery(BaseModel):
    instance_name: str = Field(description="The name of the instance to query")
    action: str = Field(description="The action to perform: 'get_info', 'get_status', etc.")

@tool
def structured_instance_tool(query: InstanceQuery) -> str:
    """Get instance information using structured input"""
    return get_instance_info(query.instance_name)

# Fix 7: Create a chain that always checks tools first
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def create_tool_first_chain():
    prompt = PromptTemplate(
        input_variables=["user_input", "instance_name"],
        template="""
        User question: {user_input}
        Instance name mentioned: {instance_name}
        
        You MUST follow these steps:
        1. First, call get_instance_info with the instance name: {instance_name}
        2. Only after getting tool results, answer the user's question
        3. Base your response entirely on tool results, never guess
        
        If no instance name is provided, ask for it before proceeding.
        """
    )
    
    return LLMChain(llm=llm_with_tools, prompt=prompt)

# Fix 8: Pre-processing to extract instance names
import re

def extract_instance_name(user_input: str) -> Optional[str]:
    """Extract instance name from user input"""
    patterns = [
        r'instance[:\s]+([a-zA-Z0-9\-_]+)',
        r'server[:\s]+([a-zA-Z0-9\-_]+)', 
        r'([a-zA-Z0-9\-_]+)\s+instance',
        r'([a-zA-Z0-9\-_]+)\s+server'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

# Fix 9: Complete working example with error handling
class InstanceManager:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.tools = [get_instance_info]
        self.agent_executor = AgentExecutor(
            agent=create_tool_calling_agent(self.llm, self.tools, prompt),
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
    
    def process_query(self, user_input: str) -> str:
        try:
            # Extract instance name
            instance_name = extract_instance_name(user_input)
            
            if not instance_name:
                return "Please provide an instance name for me to help you."
            
            # Force tool usage with explicit instruction
            enhanced_input = f"""
            User query: {user_input}
            Instance name: {instance_name}
            
            MANDATORY: Call get_instance_info tool with instance_name="{instance_name}" before answering.
            """
            
            result = self.agent_executor.invoke({"input": enhanced_input})
            
            # Check if tools were actually called
            if not result.get('intermediate_steps'):
                return "Error: No tools were called. Please try again with a more specific request."
            
            return result['output']
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

# Fix 10: Alternative approach with function calling models
from langchain_openai import ChatOpenAI

def create_function_calling_chain():
    llm = ChatOpenAI(
        model="gpt-4",  # Use a model that supports function calling
        temperature=0,
        model_kwargs={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_instance_info",
                        "description": "Get information about a server instance",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "instance_name": {
                                    "type": "string",
                                    "description": "The name of the instance"
                                }
                            },
                            "required": ["instance_name"]
                        }
                    }
                }
            ],
            "tool_choice": "auto"  # or "required" to force tool usage
        }
    )
    return llm

# Usage example:
if __name__ == "__main__":
    manager = InstanceManager()
    
    # Test queries
    test_queries = [
        "What's the status of my-server instance?",
        "Tell me about production-db server",
        "Get info for test-env instance"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        response = manager.process_query(query)
        print(f"Response: {response}\n")

# Fix 11: Debug function to check tool calling
def debug_tool_calling(agent_executor, query):
    """Debug helper to see if tools are being called"""
    result = agent_executor.invoke({"input": query})
    
    print("=== DEBUG INFO ===")
    print(f"Query: {query}")
    print(f"Tools called: {len(result.get('intermediate_steps', []))}")
    
    for i, step in enumerate(result.get('intermediate_steps', [])):
        action, observation = step
        print(f"Step {i+1}: {action.tool} with input: {action.tool_input}")
        print(f"Result: {observation}")
    
    print(f"Final output: {result['output']}")
    print("==================")
    
    return result

    




# ============================================================================
# SESSION MEMORY FOR CONVERSATIONAL CONTEXT
# ============================================================================

import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# ============================================================================
# SESSION CONTEXT MANAGER
# ============================================================================

class SessionContextManager:
    """Manage conversational context across multiple exchanges"""
    
    def __init__(self):
        self.sessions = {}  # session_id -> context_data
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get context for a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "extracted_params": {},
                "user_intent": "",
                "conversation_history": [],
                "last_activity": datetime.now(),
                "pending_clarification": False
            }
        
        return self.sessions[session_id]
    
    def update_session_context(self, session_id: str, updates: Dict[str, Any]):
        """Update session context with new information"""
        context = self.get_session_context(session_id)
        context.update(updates)
        context["last_activity"] = datetime.now()
        
        self.logger.debug(f"Updated session {session_id} context: {context}")
    
    def merge_extracted_params(self, session_id: str, new_params: Dict[str, Any]):
        """Merge new extracted parameters with existing ones"""
        context = self.get_session_context(session_id)
        extracted_params = context.get("extracted_params", {})
        
        # Only update if new value is not empty/None
        for key, value in new_params.items():
            if value and value != "":
                extracted_params[key] = value
        
        context["extracted_params"] = extracted_params
        context["last_activity"] = datetime.now()
        
        self.logger.info(f"Merged params for session {session_id}: {extracted_params}")
        return extracted_params
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove old inactive sessions"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        old_sessions = [
            session_id for session_id, context in self.sessions.items()
            if context["last_activity"] < cutoff_time
        ]
        
        for session_id in old_sessions:
            del self.sessions[session_id]
        
        if old_sessions:
            self.logger.info(f"Cleaned up {len(old_sessions)} old sessions")

# Global session manager
session_manager = SessionContextManager()

# ============================================================================
# ENHANCED PARAMETER EXTRACTION WITH SESSION MEMORY
# ============================================================================

def extract_parameters_with_memory_node(self, state: AutosysState) -> AutosysState:
    """Extract parameters while preserving session context"""
    
    try:
        session_id = state["session_id"]
        available_instances = self.db_manager.list_instances()
        
        # Get existing session context
        session_context = session_manager.get_session_context(session_id)
        existing_params = session_context.get("extracted_params", {})
        
        # Create extraction prompt with context
        extraction_prompt = f"""
Extract parameters from this query, considering previous context:

CURRENT USER QUERY: "{state['user_question']}"
PREVIOUS EXTRACTED PARAMETERS: {json.dumps(existing_params, indent=2)}
AVAILABLE INSTANCES: {', '.join(available_instances)}

Instructions:
1. Extract any new parameters from the current query
2. Keep existing parameters unless explicitly overridden
3. If user provides missing information, use it to fill gaps

Return ONLY JSON:
{{
    "extracted_instance": "instance_name_or_current_or_null",
    "extracted_job_name": "job_name_or_current_or_null", 
    "extracted_calendar_name": "calendar_name_or_current_or_null",
    "user_intent": "what_user_wants_to_do",
    "context_used": true_if_previous_context_was_helpful,
    "missing_still": ["list", "of", "still", "missing", "params"]
}}

Examples:
- If user previously said "job ABC123" and now says "in PROD", extract both
- If no previous context exists, extract only from current query
- If user provides new job name, replace the old one
"""
        
        response = self.llm.invoke(extraction_prompt)
        extraction_result = self._safe_parse_llm_json(response)
        
        if not extraction_result:
            extraction_result = {"missing_still": ["extraction_failed"]}
        
        # Merge with existing parameters
        new_params = {
            "instance": extraction_result.get("extracted_instance"),
            "job_name": extraction_result.get("extracted_job_name"),
            "calendar_name": extraction_result.get("extracted_calendar_name")
        }
        
        # Merge parameters preserving existing values
        merged_params = session_manager.merge_extracted_params(session_id, new_params)
        
        # Update state with merged parameters
        state["extracted_instance"] = merged_params.get("instance", "")
        state["extracted_job_name"] = merged_params.get("job_name", "")
        state["extracted_calendar_name"] = merged_params.get("calendar_name", "")
        state["missing_parameters"] = extraction_result.get("missing_still", [])
        
        # Update session context
        session_manager.update_session_context(session_id, {
            "user_intent": extraction_result.get("user_intent", ""),
            "pending_clarification": bool(state["missing_parameters"])
        })
        
        # Store extraction for debugging
        if "llm_analysis" not in state:
            state["llm_analysis"] = {}
        state["llm_analysis"]["extraction_with_memory"] = extraction_result
        state["llm_analysis"]["merged_params"] = merged_params
        
        self.logger.info(f"Session {session_id} - Extracted: {merged_params}, Missing: {state['missing_parameters']}")
        
    except Exception as e:
        self.logger.error(f"Parameter extraction with memory failed: {str(e)}")
        state["missing_parameters"] = ["extraction_error"]
        
    return state

# ============================================================================
# ENHANCED INTENT ANALYSIS WITH SESSION CONTEXT
# ============================================================================

def analyze_with_llm_and_context_node(self, state: AutosysState) -> AutosysState:
    """Analyze intent considering session context"""
    
    try:
        session_id = state["session_id"]
        session_context = session_manager.get_session_context(session_id)
        existing_params = session_context.get("extracted_params", {})
        previous_intent = session_context.get("user_intent", "")
        
        analysis_prompt = f"""
Analyze this user message considering conversation context:

CURRENT MESSAGE: "{state['user_question']}"
PREVIOUS CONTEXT: {json.dumps(session_context, indent=2, default=str)}
AVAILABLE INSTANCES: {', '.join(self.db_manager.list_instances())}

Analyze considering:
1. Is this a continuation of previous conversation?
2. Is user providing missing information from previous request?
3. Is this a completely new request?
4. What is the user's overall intent?

Return ONLY JSON:
{{
    "is_general_conversation": false,
    "is_continuation": true_if_related_to_previous_context,
    "query_type": "job_details|calendar_details|general_query|clarification_response",
    "overall_intent": "complete_description_of_what_user_wants",
    "requires_job_name": true,
    "requires_calendar_name": false,
    "requires_instance": true,
    "confidence_level": "high|medium|low",
    "reasoning": "explanation_of_analysis"
}}
"""
        
        response = self.llm.invoke(analysis_prompt)
        analysis = self._safe_parse_llm_json(response)
        
        if not analysis:
            analysis = {
                "is_general_conversation": True,
                "is_continuation": False,
                "query_type": "general_conversation"
            }
        
        # Update state
        state["llm_analysis"] = analysis
        state["is_general_conversation"] = analysis.get("is_general_conversation", False)
        state["query_type"] = analysis.get("query_type", "general_query")
        
        # Update session context with intent
        session_manager.update_session_context(session_id, {
            "user_intent": analysis.get("overall_intent", ""),
            "is_continuation": analysis.get("is_continuation", False)
        })
        
        self.logger.info(f"Session {session_id} - Intent: {analysis.get('query_type')}, Continuation: {analysis.get('is_continuation')}")
        
    except Exception as e:
        self.logger.error(f"Intent analysis with context failed: {str(e)}")
        state["llm_analysis"] = {"error": str(e)}
        state["is_general_conversation"] = True
        
    return state

# ============================================================================
# ENHANCED CLARIFICATION WITH CONTEXT AWARENESS
# ============================================================================

def request_missing_params_with_context_node(self, state: AutosysState) -> AutosysState:
    """Generate contextual clarification requests"""
    
    try:
        session_id = state["session_id"]
        session_context = session_manager.get_session_context(session_id)
        existing_params = session_context.get("extracted_params", {})
        user_intent = session_context.get("user_intent", "")
        
        missing_params = state.get("missing_parameters", [])
        available_instances = self.db_manager.list_instances()
        instance_info = self.db_manager.get_instance_info()
        
        clarification_prompt = f"""
Generate a contextual clarification request for missing parameters:

CONTEXT:
- User's Overall Intent: {user_intent}
- Current Query: "{state['user_question']}"
- Already Have: {json.dumps({k: v for k, v in existing_params.items() if v}, indent=2)}
- Still Missing: {missing_params}
- Available Instances: {', '.join(available_instances)}

Create a clarification that:
1. Acknowledges what they already provided
2. Clearly states what's still needed
3. Provides relevant examples
4. Maintains conversational flow

Generate HTML clarification message that feels like a natural conversation.
Include the instance information and examples.

INSTANCE INFO:
{instance_info}
"""
        
        response = self.llm.invoke(clarification_prompt)
        
        if hasattr(response, 'content'):
            clarification_html = response.content
        else:
            clarification_html = str(response)
        
        # Clean HTML markers
        clarification_html = re.sub(r'```html\s*', '', clarification_html, flags=re.IGNORECASE)
        clarification_html = re.sub(r'```\s*$', '', clarification_html)
        
        # Mark session as pending clarification
        session_manager.update_session_context(session_id, {
            "pending_clarification": True
        })
        
        state["formatted_output"] = clarification_html
        
        self.logger.info(f"Session {session_id} - Requested clarification for: {missing_params}")
        
    except Exception as e:
        self.logger.error(f"Contextual clarification failed: {e}")
        # Fallback clarification
        have_params = [f"{k}: {v}" for k, v in existing_params.items() if v]
        missing_str = ", ".join(missing_params)
        
        state["formatted_output"] = f"""
        <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #856404;">Additional Information Needed</h4>
            <p style="color: #856404;">I have: {', '.join(have_params) if have_params else 'nothing yet'}</p>
            <p style="color: #856404;">I still need: {missing_str}</p>
            <div style="background: #f8f9fa; border-radius: 4px; padding: 10px; margin: 10px 0;">
                <strong>Available Instances:</strong><br>
                {self.db_manager.get_instance_info()}
            </div>
        </div>
        """
    
    return state

# ============================================================================
# UPDATE MAIN WORKFLOW
# ============================================================================

def _build_graph_with_memory(self) -> StateGraph:
    """Build workflow with session memory support"""
    
    workflow = StateGraph(AutosysState)
    
    # Add nodes with memory support
    workflow.add_node("analyze_with_context", self.analyze_with_llm_and_context_node)
    workflow.add_node("handle_conversation", self.handle_conversation_node)
    workflow.add_node("extract_parameters_with_memory", self.extract_parameters_with_memory_node)
    workflow.add_node("request_missing_params_with_context", self.request_missing_params_with_context_node)
    workflow.add_node("generate_sql", self.generate_sql_llm_node)
    workflow.add_node("execute_query", self.execute_query_node)
    workflow.add_node("format_results", self.format_results_llm_node)
    workflow.add_node("handle_error", self.handle_error_llm_node)
    
    # Set entry point
    workflow.set_entry_point("analyze_with_context")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "analyze_with_context",
        self._should_handle_conversation,
        {
            "conversation": "handle_conversation",
            "database": "extract_parameters_with_memory"
        }
    )
    
    workflow.add_edge("handle_conversation", END)
    
    workflow.add_conditional_edges(
        "extract_parameters_with_memory", 
        self._needs_parameters,
        {
            "needs_params": "request_missing_params_with_context",
            "has_all_params": "generate_sql"
        }
    )
    
    workflow.add_edge("request_missing_params_with_context", END)
    
    # ... rest of the workflow remains the same
    
    return workflow.compile(checkpointer=self.memory)

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
CONVERSATION FLOW WITH MEMORY:

User: "Show job details"
System: "I need the job name and database instance. Which job would you like details for?"

User: "job ABC123"  
System: "Thanks! I have job ABC123. Which database instance? (PROD, DEV, TEST)"

User: "PROD"
System: [Executes query for job ABC123 in PROD instance]

The system remembers "ABC123" from the previous exchange and only asks for the missing instance.
"""

# ============================================================================
# INTEGRATION INSTRUCTIONS
# ============================================================================

"""
TO ADD SESSION MEMORY TO YOUR EXISTING SYSTEM:

1. Add the SessionContextManager class above your existing code

2. Replace these methods in your LLMDrivenAutosysSystem class:
   - analyze_with_llm_node → analyze_with_llm_and_context_node
   - extract_parameters_llm_node → extract_parameters_with_memory_node  
   - request_missing_params_llm_node → request_missing_params_with_context_node
   - _build_graph → _build_graph_with_memory

3. Update the method names in your workflow builder

4. Add session cleanup (optional):
   ```python
   # Clean up old sessions periodically
   session_manager.cleanup_old_sessions(max_age_hours=24)
   ```

KEY BENEFITS:
- Remembers parameters across conversation turns
- Natural conversational flow
- Reduces user frustration with repetitive questions  
- Maintains context for complex multi-step queries
- Automatic session cleanup prevents memory leaks
"""



&&&&&______&&&&&






# ============================================================================
# FIXES FOR STRING INDICES ERROR - KEEP EXISTING FUNCTION/CLASS NAMES
# ============================================================================

# Fix 1: Add this helper method to your LLMDrivenAutosysSystem class
class LLMDrivenAutosysSystem:
    
    def _safe_parse_llm_json(self, llm_response) -> Dict[str, Any]:
        """Helper method to safely parse LLM JSON responses"""
        try:
            # Extract content properly
            if hasattr(llm_response, 'content'):
                content = llm_response.content
            else:
                content = str(llm_response)
            
            self.logger.debug(f"Raw LLM response: {content}")
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                parsed_json = json.loads(json_str)
                return parsed_json
            else:
                self.logger.warning("No JSON found in LLM response")
                return {}
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            self.logger.error(f"Content was: {content}")
            return {}
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return {}

# Fix 2: Update your analyze_with_llm_node method
def analyze_with_llm_node(self, state: AutosysState) -> AutosysState:
    """Analyze user intent with proper error handling"""
    
    try:
        available_instances = self.db_manager.list_instances()
        
        analysis_prompt = f"""
You are an expert Autosys database assistant. Analyze this user message.

USER MESSAGE: "{state['user_question']}"
AVAILABLE INSTANCES: {', '.join(available_instances)}

Return ONLY valid JSON in this exact format (no other text):
{{
    "is_general_conversation": false,
    "query_type": "job_details",
    "confidence_level": "high",
    "requires_job_name": true,
    "requires_calendar_name": false,
    "requires_instance": true,
    "extracted_instance": null,
    "extracted_job_name": null,
    "extracted_calendar_name": null,
    "missing_parameters": ["job_name", "instance"],
    "user_intent_summary": "User wants job details",
    "recommended_action": "parameter_collection",
    "reasoning": "Analysis reasoning here"
}}
"""
        
        response = self.llm.invoke(analysis_prompt)
        
        # FIX: Use safe parsing method
        analysis = self._safe_parse_llm_json(response)
        
        # FIX: Provide defaults if parsing failed
        if not analysis:
            analysis = {
                "is_general_conversation": True,
                "query_type": "general_conversation", 
                "missing_parameters": [],
                "extracted_instance": None,
                "extracted_job_name": None,
                "extracted_calendar_name": None
            }
        
        # FIX: Safe assignment with get() method
        state["llm_analysis"] = analysis
        state["is_general_conversation"] = analysis.get("is_general_conversation", False)
        state["query_type"] = analysis.get("query_type", "general_query")
        state["extracted_instance"] = analysis.get("extracted_instance") or ""
        state["extracted_job_name"] = analysis.get("extracted_job_name") or ""
        state["extracted_calendar_name"] = analysis.get("extracted_calendar_name") or ""
        state["missing_parameters"] = analysis.get("missing_parameters", [])
        
    except Exception as e:
        self.logger.error(f"LLM analysis failed: {str(e)}")
        # FIX: Safe fallback values
        state["llm_analysis"] = {"error": str(e)}
        state["is_general_conversation"] = True
        state["query_type"] = "general_conversation"
        state["extracted_instance"] = ""
        state["extracted_job_name"] = ""
        state["extracted_calendar_name"] = ""
        state["missing_parameters"] = []
    
    return state

# Fix 3: Update your extract_parameters_llm_node method
def extract_parameters_llm_node(self, state: AutosysState) -> AutosysState:
    """Extract parameters with proper error handling"""
    
    try:
        available_instances = self.db_manager.list_instances()
        
        extraction_prompt = f"""
Extract parameters and return ONLY valid JSON:

User Query: "{state['user_question']}"
Available Instances: {', '.join(available_instances)}

{{
    "validated_instance": null,
    "validated_job_name": null,
    "validated_calendar_name": null,
    "instance_confidence": "none",
    "job_confidence": "none",
    "calendar_confidence": "none",
    "missing_critical_params": [],
    "can_proceed": false,
    "validation_notes": "Parameter extraction notes"
}}
"""
        
        response = self.llm.invoke(extraction_prompt)
        
        # FIX: Use safe parsing
        validation = self._safe_parse_llm_json(response)
        
        # FIX: Provide defaults if parsing failed
        if not validation:
            validation = {
                "validated_instance": None,
                "validated_job_name": None,
                "validated_calendar_name": None,
                "missing_critical_params": ["extraction_failed"]
            }
        
        # FIX: Safe parameter assignment
        state["extracted_instance"] = validation.get("validated_instance") or ""
        state["extracted_job_name"] = validation.get("validated_job_name") or ""
        state["extracted_calendar_name"] = validation.get("validated_calendar_name") or ""
        state["missing_parameters"] = validation.get("missing_critical_params", [])
        
        # FIX: Ensure llm_analysis exists before updating
        if "llm_analysis" not in state:
            state["llm_analysis"] = {}
        state["llm_analysis"]["validation"] = validation
        
    except Exception as e:
        self.logger.error(f"Parameter extraction failed: {str(e)}")
        # FIX: Safe defaults
        state["extracted_instance"] = ""
        state["extracted_job_name"] = ""
        state["extracted_calendar_name"] = ""
        state["missing_parameters"] = ["parameter_extraction_failed"]
        
    return state

# Fix 4: Update your query method
def query(self, user_question: str, session_id: str) -> Dict[str, Any]:
    """Main query method with comprehensive error handling"""
    
    # FIX: Initialize all required state fields
    initial_state = {
        "messages": [],
        "user_question": user_question,
        "llm_analysis": {},
        "is_general_conversation": False,
        "extracted_instance": "",
        "extracted_job_name": "",
        "extracted_calendar_name": "", 
        "missing_parameters": [],
        "query_type": "",
        "sql_query": "",
        "query_results": {},
        "formatted_output": "",
        "error": "",
        "session_id": session_id
    }
    
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        final_state = self.graph.invoke(initial_state, config=config)
        
        # FIX: Safe result extraction with proper defaults
        return {
            "success": not bool(final_state.get("error", "")),
            "formatted_output": final_state.get("formatted_output", ""),
            "is_conversation": final_state.get("is_general_conversation", False),
            "needs_clarification": bool(final_state.get("missing_parameters", [])),
            "query_type": final_state.get("query_type", ""),
            "llm_analysis": final_state.get("llm_analysis", {}),
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

# Fix 5: Update your get_chat_response function
def get_chat_response(message: str, session_id: str) -> str:
    """Chat function with proper error handling"""
    global _autosys_system
    
    try:
        if not message or not message.strip():
            message = "Hello! How can I help you today?"
        
        if not _autosys_system:
            return """
            <div style="border: 1px solid #ffc107; background: #fff3cd; color: #856404; padding: 15px; border-radius: 5px;">
                <h4>System Not Ready</h4>
                <p>The system is not initialized. Please contact administrator.</p>
            </div>
            """
        
        # Process message
        result = _autosys_system.query(message.strip(), session_id)
        
        # FIX: Check result type before accessing
        if isinstance(result, dict):
            return result.get("formatted_output", "No output generated")
        elif isinstance(result, str):
            return result
        else:
            return str(result)
        
    except Exception as e:
        logger.error(f"Chat response error: {e}", exc_info=True)
        return f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;">
            <h4>Processing Error</h4>
            <p>Error processing request: {str(e)}</p>
            <p style="font-size: 12px;">Please try rephrasing your request.</p>
        </div>
        """

# Fix 6: Update your format_results_llm_node method
def format_results_llm_node(self, state: AutosysState) -> AutosysState:
    """Format results with proper error handling"""
    
    try:
        # FIX: Safe access to query results
        query_results = state.get("query_results", {})
        if not isinstance(query_results, dict):
            query_results = {}
            
        results = query_results.get("results", [])
        
        if not results:
            instance_used = query_results.get("instance_used", "Unknown")
            state["formatted_output"] = f"""
            <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px;">
                <h4 style="margin: 0 0 10px 0; color: #856404;">No Results Found</h4>
                <p style="margin: 0; color: #856404;">No records found in instance: <strong>{instance_used}</strong></p>
            </div>
            """
            return state
        
        # Continue with LLM formatting...
        formatting_prompt = f"""
Create HTML for Autosys query results:

Results Count: {len(results)}
Instance: {query_results.get('instance_used', 'Unknown')}

Data: {json.dumps(results[:5], indent=2, default=str)}

Generate professional HTML with styling.
"""
        
        response = self.llm.invoke(formatting_prompt)
        
        # FIX: Safe content extraction
        if hasattr(response, 'content'):
            formatted_html = response.content
        else:
            formatted_html = str(response)
        
        # Clean HTML markers
        formatted_html = re.sub(r'```html\s*', '', formatted_html, flags=re.IGNORECASE)
        formatted_html = re.sub(r'```\s*$', '', formatted_html)
        
        state["formatted_output"] = formatted_html
        
    except Exception as e:
        self.logger.error(f"Result formatting failed: {e}")
        # FIX: Fallback formatting
        results_count = len(state.get("query_results", {}).get("results", []))
        state["formatted_output"] = f"""
        <div style="border: 1px solid #dee2e6; border-radius: 5px; padding: 15px;">
            <h4>Query Results</h4>
            <p>Found {results_count} results</p>
            <p style="color: #666; font-size: 12px;">Result formatting encountered an error: {str(e)}</p>
        </div>
        """
    
    return state

# Fix 7: Add debugging utility (add this as a method to your class)
def _debug_state_safely(self, state, node_name="Unknown"):
    """Debug utility to safely log state information"""
    try:
        self.logger.debug(f"=== {node_name} STATE DEBUG ===")
        for key, value in state.items():
            value_type = type(value).__name__
            if isinstance(value, (dict, list)):
                value_preview = f"{value_type} with {len(value)} items"
            else:
                value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            self.logger.debug(f"{key}: {value_type} = {value_preview}")
    except Exception as e:
        self.logger.error(f"Debug logging failed: {e}")

# Fix 8: Update your handle_error_llm_node method
def handle_error_llm_node(self, state: AutosysState) -> AutosysState:
    """Handle errors with safe fallback"""
    
    try:
        error_msg = state.get("error", "Unknown error occurred")
        
        # Try LLM-generated error message
        error_prompt = f"""
Create a user-friendly HTML error message for this error:

Error: {error_msg}
User Query: {state.get('user_question', 'Unknown')}

Generate helpful HTML error message.
"""
        
        response = self.llm.invoke(error_prompt)
        
        # FIX: Safe response handling
        if hasattr(response, 'content'):
            error_html = response.content
        else:
            error_html = str(response)
            
        # Clean HTML markers
        error_html = re.sub(r'```html\s*', '', error_html, flags=re.IGNORECASE)
        error_html = re.sub(r'```\s*$', '', error_html)
        
        state["formatted_output"] = error_html
        
    except Exception as e:
        # FIX: Ultimate fallback if LLM fails
        self.logger.error(f"Error handling failed: {e}")
        error_msg = state.get("error", "System error occurred")
        state["formatted_output"] = f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;">
            <h4>System Error</h4>
            <p>Error: {error_msg}</p>
            <p style="font-size: 12px;">Please try rephrasing your request or contact support.</p>
        </div>
        """
    
    return state

# ============================================================================
# IMPLEMENTATION INSTRUCTIONS
# ============================================================================

"""
TO APPLY THESE FIXES TO YOUR EXISTING CODE:

1. Add the _safe_parse_llm_json method to your LLMDrivenAutosysSystem class

2. Replace the content of these existing methods with the fixed versions:
   - analyze_with_llm_node
   - extract_parameters_llm_node  
   - query
   - format_results_llm_node
   - handle_error_llm_node

3. Replace your existing get_chat_response function with the fixed version

4. Add the _debug_state_safely method for debugging

KEY CHANGES MADE:
- Added safe JSON parsing for all LLM responses
- Added proper fallback values when parsing fails
- Added type checking before dictionary access
- Added comprehensive error handling in all methods
- Added safe state initialization
- Added debug logging capabilities

The fixes maintain all your existing function and class names while making them robust against the string indices error.
"""

†*****"****"""""†



# ============================================================================
# DEBUG AND FIX STRING INDICES ERROR
# ============================================================================

import json
import logging
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

# ============================================================================
# COMMON CAUSES AND FIXES FOR "string indices must be integers" ERROR
# ============================================================================

def debug_llm_response_parsing():
    """Debug LLM response parsing issues"""
    
    # PROBLEM 1: LLM returns string instead of expected JSON
    def safe_parse_llm_json(llm_response) -> Dict[str, Any]:
        """Safely parse LLM response that should contain JSON"""
        
        try:
            # Get content from LLM response
            if hasattr(llm_response, 'content'):
                content = llm_response.content
            else:
                content = str(llm_response)
            
            logger.debug(f"Raw LLM response: {content}")
            
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                try:
                    parsed_json = json.loads(json_str)
                    logger.debug(f"Successfully parsed JSON: {parsed_json}")
                    return parsed_json
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed: {e}")
                    logger.error(f"JSON string was: {json_str}")
                    return {}
            else:
                logger.warning("No JSON found in LLM response")
                return {}
                
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {}

    return safe_parse_llm_json

# ============================================================================
# FIXED LLM-DRIVEN NODES
# ============================================================================

class FixedLLMDrivenAutosysSystem:
    """Fixed version of LLM-driven system with proper error handling"""
    
    def __init__(self, db_manager, llm_instance):
        self.db_manager = db_manager
        self.llm = llm_instance
        self.logger = logging.getLogger(self.__class__.__name__)
        # ... rest of initialization

    def analyze_with_llm_node(self, state) -> Dict[str, Any]:
        """Fixed LLM analysis node with proper error handling"""
        
        try:
            available_instances = self.db_manager.list_instances()
            instance_info = self.db_manager.get_instance_info()
            
            analysis_prompt = f"""
You are an expert Autosys database assistant. Analyze this user message comprehensively.

USER MESSAGE: "{state['user_question']}"
AVAILABLE DATABASE INSTANCES: {', '.join(available_instances)}

Provide your analysis in this exact JSON format (return ONLY the JSON, no other text):
{{
    "is_general_conversation": false,
    "query_type": "job_details",
    "confidence_level": "high",
    "requires_job_name": true,
    "requires_calendar_name": false,
    "requires_instance": true,
    "extracted_instance": null,
    "extracted_job_name": null,
    "extracted_calendar_name": null,
    "missing_parameters": ["job_name", "instance"],
    "user_intent_summary": "User wants job details",
    "recommended_action": "parameter_collection",
    "reasoning": "User asked for job details but didn't specify job name or instance"
}}
"""
            
            response = self.llm.invoke(analysis_prompt)
            
            # FIXED: Proper response handling
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            logger.debug(f"LLM analysis raw response: {content}")
            
            # Parse JSON safely
            try:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    logger.debug(f"Parsed analysis: {analysis}")
                else:
                    logger.warning("No JSON found in LLM response, using defaults")
                    analysis = self._get_default_analysis()
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
                logger.error(f"Raw content: {content}")
                analysis = self._get_default_analysis()
            
            # FIXED: Safe dictionary access
            state["llm_analysis"] = analysis
            state["is_general_conversation"] = analysis.get("is_general_conversation", False)
            state["query_type"] = analysis.get("query_type", "general_query")
            state["extracted_instance"] = analysis.get("extracted_instance") or ""
            state["extracted_job_name"] = analysis.get("extracted_job_name") or ""
            state["extracted_calendar_name"] = analysis.get("extracted_calendar_name") or ""
            state["missing_parameters"] = analysis.get("missing_parameters", [])
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            # FIXED: Provide safe defaults
            state["llm_analysis"] = self._get_default_analysis()
            state["is_general_conversation"] = True  # Safe fallback
            state["missing_parameters"] = []
        
        return state

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Provide safe default analysis when LLM fails"""
        return {
            "is_general_conversation": True,
            "query_type": "general_conversation",
            "confidence_level": "low",
            "requires_job_name": False,
            "requires_calendar_name": False,
            "requires_instance": False,
            "extracted_instance": None,
            "extracted_job_name": None,
            "extracted_calendar_name": None,
            "missing_parameters": [],
            "user_intent_summary": "Analysis failed, treating as conversation",
            "recommended_action": "conversation",
            "reasoning": "LLM analysis failed, falling back to safe defaults"
        }

    def extract_parameters_llm_node(self, state) -> Dict[str, Any]:
        """Fixed parameter extraction with proper error handling"""
        
        try:
            available_instances = self.db_manager.list_instances()
            
            extraction_prompt = f"""
Extract parameters from this query and return ONLY JSON:

User Query: "{state['user_question']}"
Available Instances: {', '.join(available_instances)}

{{
    "validated_instance": null,
    "validated_job_name": null, 
    "validated_calendar_name": null,
    "instance_confidence": "none",
    "job_confidence": "none",
    "calendar_confidence": "none",
    "missing_critical_params": [],
    "can_proceed": true,
    "fuzzy_matches_applied": [],
    "validation_notes": "Parameter extraction analysis",
    "recommendation": "proceed"
}}
"""
            
            response = self.llm.invoke(extraction_prompt)
            
            # FIXED: Safe response handling
            content = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"Parameter extraction raw response: {content}")
            
            try:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    validation = json.loads(json_match.group())
                else:
                    validation = self._get_default_validation()
            except json.JSONDecodeError as e:
                logger.error(f"Parameter extraction JSON parsing failed: {e}")
                validation = self._get_default_validation()
            
            # FIXED: Safe parameter assignment
            state["extracted_instance"] = validation.get("validated_instance") or ""
            state["extracted_job_name"] = validation.get("validated_job_name") or ""
            state["extracted_calendar_name"] = validation.get("validated_calendar_name") or ""
            state["missing_parameters"] = validation.get("missing_critical_params", [])
            
            # Store validation for debugging
            if "llm_analysis" not in state:
                state["llm_analysis"] = {}
            state["llm_analysis"]["validation"] = validation
            
        except Exception as e:
            logger.error(f"Parameter extraction failed: {str(e)}")
            # Safe defaults
            state["extracted_instance"] = ""
            state["extracted_job_name"] = ""
            state["extracted_calendar_name"] = ""
            state["missing_parameters"] = ["parameter_extraction_failed"]
        
        return state

    def _get_default_validation(self) -> Dict[str, Any]:
        """Safe default validation when extraction fails"""
        return {
            "validated_instance": None,
            "validated_job_name": None,
            "validated_calendar_name": None,
            "instance_confidence": "none",
            "job_confidence": "none", 
            "calendar_confidence": "none",
            "missing_critical_params": ["extraction_failed"],
            "can_proceed": False,
            "fuzzy_matches_applied": [],
            "validation_notes": "Parameter extraction failed",
            "recommendation": "request_clarification"
        }

    def query(self, user_question: str, session_id: str) -> Dict[str, Any]:
        """Fixed main query method with comprehensive error handling"""
        
        # FIXED: Ensure all required state fields exist
        initial_state = {
            "messages": [],
            "user_question": user_question,
            "llm_analysis": {},
            "is_general_conversation": False,
            "extracted_instance": "",
            "extracted_job_name": "",
            "extracted_calendar_name": "", 
            "missing_parameters": [],
            "query_type": "",
            "sql_query": "",
            "query_results": {},
            "formatted_output": "",
            "error": "",
            "session_id": session_id
        }
        
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            final_state = self.graph.invoke(initial_state, config=config)
            
            # FIXED: Safe result extraction
            return {
                "success": not bool(final_state.get("error", "")),
                "formatted_output": final_state.get("formatted_output", ""),
                "is_conversation": final_state.get("is_general_conversation", False),
                "needs_clarification": bool(final_state.get("missing_parameters", [])),
                "query_type": final_state.get("query_type", ""),
                "llm_analysis": final_state.get("llm_analysis", {}),
                "error": final_state.get("error", "")
            }
            
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            return {
                "success": False,
                "formatted_output": self._create_error_html(str(e)),
                "error": str(e)
            }

    def _create_error_html(self, error_message: str) -> str:
        """Create error HTML when system fails"""
        return f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;">
            <h4>System Error</h4>
            <p>An error occurred while processing your request: {error_message}</p>
            <p style="font-size: 12px; margin-top: 10px;">Please try rephrasing your question or contact support if the issue persists.</p>
        </div>
        """

# ============================================================================
# FIXED CHAT RESPONSE FUNCTION
# ============================================================================

def get_chat_response_fixed(message: str, session_id: str) -> str:
    """Fixed chat function with comprehensive error handling"""
    global _autosys_system
    
    try:
        if not message or not message.strip():
            message = "Hello! How can I help you today?"
        
        if not _autosys_system:
            return """
            <div style="border: 1px solid #ffc107; background: #fff3cd; color: #856404; padding: 15px; border-radius: 5px;">
                <h4>System Not Ready</h4>
                <p>The system is not initialized. Please contact administrator.</p>
            </div>
            """
        
        # FIXED: Process with error handling
        result = _autosys_system.query(message.strip(), session_id)
        
        # FIXED: Safe result access
        if isinstance(result, dict):
            return result.get("formatted_output", "No output generated")
        elif isinstance(result, str):
            return result
        else:
            return str(result)
        
    except Exception as e:
        logger.error(f"Chat response error: {e}", exc_info=True)
        return f"""
        <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;">
            <h4>Processing Error</h4>
            <p>Error: {str(e)}</p>
            <p style="font-size: 12px;">Please try a simpler question or contact support.</p>
        </div>
        """

# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

def debug_state_types(state):
    """Debug utility to check state variable types"""
    logger.info("=== STATE TYPE DEBUGGING ===")
    for key, value in state.items():
        logger.info(f"{key}: {type(value)} = {value}")
    logger.info("=== END STATE DEBUGGING ===")

def safe_get_from_state(state, key: str, default_value=None):
    """Safely get value from state with type checking"""
    try:
        value = state.get(key, default_value)
        logger.debug(f"Retrieved {key}: {type(value)} = {value}")
        return value
    except Exception as e:
        logger.error(f"Error accessing state['{key}']: {e}")
        return default_value

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
TO FIX THE STRING INDICES ERROR:

1. Replace your existing LLM nodes with the fixed versions above
2. Use get_chat_response_fixed() instead of get_chat_response()
3. Add debug logging to identify where the error occurs
4. Ensure LLM responses are properly parsed as JSON

COMMON CAUSES:
- LLM returning plain text instead of JSON
- Accessing dictionary keys on string variables
- Missing error handling in LLM response parsing
- State variables not initialized properly

DEBUGGING STEPS:
1. Add debug_state_types(state) in your nodes
2. Check LLM response content before parsing
3. Use safe_get_from_state() for state access
4. Add try/catch around all LLM calls

"""

££££






# ============================================================================
# LLM-DRIVEN MULTI-DATABASE AUTOSYS SYSTEM - FULLY AI-POWERED
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
# ENHANCED STATE FOR LLM-DRIVEN SYSTEM
# ============================================================================

class AutosysState(TypedDict):
    """State for LLM-driven multi-database Autosys workflow"""
    messages: Annotated[List[Dict[str, str]], operator.add]
    user_question: str
    llm_analysis: Dict[str, Any]  # LLM's complete analysis
    is_general_conversation: bool
    extracted_instance: str
    extracted_job_name: str
    extracted_calendar_name: str
    missing_parameters: List[str]  # LLM-determined missing params
    query_type: str
    sql_query: str
    query_results: Dict[str, Any]
    formatted_output: str
    error: str
    session_id: str

# ============================================================================
# DATABASE MANAGER (Keep existing implementation)
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
            status = "Connected" if instance.is_connected else "Disconnected"
            desc = f" - {instance.description}" if instance.description else ""
            info_lines.append(f"{name}: {status}{desc}")
        
        return "\n".join(info_lines)

# ============================================================================
# FULLY LLM-DRIVEN AUTOSYS SYSTEM
# ============================================================================

class LLMDrivenAutosysSystem:
    """Completely LLM-driven system for Autosys queries"""
    
    def __init__(self, db_manager: DatabaseManager, llm_instance):
        self.db_manager = db_manager
        self.llm = llm_instance
        self.memory = MemorySaver()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Build workflow
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LLM-driven workflow"""
        
        workflow = StateGraph(AutosysState)
        
        # Add nodes - all driven by LLM
        workflow.add_node("analyze_with_llm", self.analyze_with_llm_node)
        workflow.add_node("handle_conversation", self.handle_conversation_node)
        workflow.add_node("extract_parameters_llm", self.extract_parameters_llm_node)
        workflow.add_node("request_missing_params_llm", self.request_missing_params_llm_node)
        workflow.add_node("generate_sql_llm", self.generate_sql_llm_node)
        workflow.add_node("execute_query", self.execute_query_node)
        workflow.add_node("format_results_llm", self.format_results_llm_node)
        workflow.add_node("handle_error_llm", self.handle_error_llm_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_with_llm")
        
        # Add conditional routing - decisions made by LLM
        workflow.add_conditional_edges(
            "analyze_with_llm",
            self._llm_routing_decision,
            {
                "conversation": "handle_conversation",
                "database": "extract_parameters_llm"
            }
        )
        
        workflow.add_edge("handle_conversation", END)
        
        workflow.add_conditional_edges(
            "extract_parameters_llm", 
            self._llm_parameter_check,
            {
                "needs_params": "request_missing_params_llm",
                "has_all_params": "generate_sql_llm"
            }
        )
        
        workflow.add_edge("request_missing_params_llm", END)
        
        workflow.add_conditional_edges(
            "generate_sql_llm",
            self._llm_execution_check,
            {
                "execute": "execute_query",
                "error": "handle_error_llm"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_query",
            self._llm_result_check,
            {
                "format": "format_results_llm",
                "error": "handle_error_llm"
            }
        )
        
        workflow.add_edge("format_results_llm", END)
        workflow.add_edge("handle_error_llm", END)
        
        return workflow.compile(checkpointer=self.memory)

    def analyze_with_llm_node(self, state: AutosysState) -> AutosysState:
        """Complete LLM-driven analysis of user intent and requirements"""
        
        try:
            available_instances = self.db_manager.list_instances()
            instance_info = self.db_manager.get_instance_info()
            
            analysis_prompt = f"""
You are an expert Autosys database assistant. Analyze this user message comprehensively.

USER MESSAGE: "{state['user_question']}"
AVAILABLE DATABASE INSTANCES: {', '.join(available_instances)}
INSTANCE DETAILS:
{instance_info}

ANALYZE THE MESSAGE FOR:
1. Intent Classification:
   - Is this general conversation (greetings, thanks, casual chat)?
   - Is this an Autosys database query?
   - What specific type of query (job details, calendar details, status check, list operations)?

2. Parameter Requirements:
   - Does this query require a specific job name?
   - Does this query require a specific calendar name?
   - Does this query require a database instance?
   - What parameters are missing that are essential for the query?

3. Extraction Analysis:
   - What job names, calendar names, or instance names can be extracted?
   - How confident are you in each extraction?
   - What fuzzy matches might apply for instances?

4. Query Complexity:
   - Is this a simple status check, detailed analysis, or complex reporting?
   - What database tables would be involved?

Provide your analysis in this exact JSON format:
{{
    "is_general_conversation": boolean,
    "query_type": "conversation|job_details|calendar_details|job_status|job_list|general_query",
    "confidence_level": "high|medium|low",
    "requires_job_name": boolean,
    "requires_calendar_name": boolean,
    "requires_instance": boolean,
    "extracted_instance": "instance_name_or_null",
    "extracted_job_name": "job_name_or_null",
    "extracted_calendar_name": "calendar_name_or_null",
    "missing_parameters": ["list", "of", "required", "missing", "params"],
    "user_intent_summary": "brief description of what user wants",
    "recommended_action": "conversation|parameter_collection|direct_query",
    "reasoning": "explanation of your analysis"
}}

Be thorough and accurate in your analysis. Consider context, implied requirements, and user expectations.
"""
            
            response = self.llm.invoke(analysis_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse LLM analysis
            try:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    
                    # Store complete LLM analysis
                    state["llm_analysis"] = analysis
                    
                    # Set state based on LLM decisions
                    state["is_general_conversation"] = analysis.get("is_general_conversation", False)
                    state["query_type"] = analysis.get("query_type", "general_query")
                    state["extracted_instance"] = analysis.get("extracted_instance", "") or ""
                    state["extracted_job_name"] = analysis.get("extracted_job_name", "") or ""
                    state["extracted_calendar_name"] = analysis.get("extracted_calendar_name", "") or ""
                    state["missing_parameters"] = analysis.get("missing_parameters", [])
                    
                    self.logger.info(f"LLM Analysis: {analysis.get('user_intent_summary', 'Unknown intent')}")
                    self.logger.info(f"Recommended Action: {analysis.get('recommended_action', 'Unknown')}")
                    
                else:
                    # Fallback if JSON parsing fails
                    state["llm_analysis"] = {"error": "Failed to parse LLM response"}
                    state["is_general_conversation"] = "conversation" in content.lower()
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"LLM analysis JSON parsing failed: {e}")
                state["llm_analysis"] = {"error": f"JSON parsing failed: {str(e)}"}
                state["is_general_conversation"] = True  # Safe fallback
                
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {str(e)}")
            state["error"] = f"Analysis failed: {str(e)}"
            state["llm_analysis"] = {"error": str(e)}
            state["is_general_conversation"] = True
        
        return state

    def handle_conversation_node(self, state: AutosysState) -> AutosysState:
        """LLM-driven conversation handling"""
        try:
            conversation_prompt = f"""
You are a helpful AI assistant for a multi-database Autosys job scheduling system.

CONTEXT:
- User message: "{state['user_question']}"
- Available database instances: {', '.join(self.db_manager.list_instances())}
- System capabilities: Query job details, calendar information, status checks across multiple database environments

INSTRUCTIONS:
- Respond naturally and professionally to the user's message
- If they're greeting you, greet them back warmly
- If they ask about capabilities, explain you can help with Autosys job and calendar queries across multiple database instances
- If they ask about available instances, mention the ones available
- Keep responses conversational but informative
- Don't use bullet points or formal lists in casual conversation

Provide a friendly, helpful response:
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
                    Hi! I can help you query Autosys job and calendar information across multiple database instances. How can I assist you today?
                </div>
            </div>
            """
        
        return state

    def extract_parameters_llm_node(self, state: AutosysState) -> AutosysState:
        """LLM-driven parameter extraction and validation"""
        
        try:
            available_instances = self.db_manager.list_instances()
            
            extraction_prompt = f"""
You are an expert at extracting and validating Autosys database parameters.

CONTEXT:
- User Query: "{state['user_question']}"
- Previous Analysis: {json.dumps(state.get('llm_analysis', {}), indent=2)}
- Available Database Instances: {', '.join(available_instances)}

TASKS:
1. EXTRACT PARAMETERS with high precision:
   - Database instance names (validate against available instances)
   - Specific job names (look for job identifier patterns)
   - Calendar names (look for calendar identifiers)

2. VALIDATE EXTRACTIONS:
   - Check if extracted instance exists in available instances
   - Apply fuzzy matching for instance names (PROD/production, DEV/development, etc.)
   - Assess confidence levels for each extraction

3. IDENTIFY MISSING REQUIREMENTS:
   - Based on the query type, what parameters are absolutely required?
   - What information is missing that would prevent a successful query?

4. PROVIDE RECOMMENDATIONS:
   - Should we proceed with current parameters?
   - What clarification is needed from the user?

Return your analysis in JSON format:
{{
    "validated_instance": "final_instance_name_or_null",
    "validated_job_name": "final_job_name_or_null", 
    "validated_calendar_name": "final_calendar_name_or_null",
    "instance_confidence": "high|medium|low|none",
    "job_confidence": "high|medium|low|none",
    "calendar_confidence": "high|medium|low|none",
    "missing_critical_params": ["list", "of", "missing", "required", "parameters"],
    "can_proceed": boolean,
    "fuzzy_matches_applied": ["list", "of", "any", "fuzzy", "matches", "made"],
    "validation_notes": "explanation of validation decisions",
    "recommendation": "proceed|request_clarification|error"
}}

Be thorough in validation and conservative in confidence assessment.
"""
            
            response = self.llm.invoke(extraction_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse validation results
            try:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    validation = json.loads(json_match.group())
                    
                    # Update state with validated parameters
                    state["extracted_instance"] = validation.get("validated_instance", "") or ""
                    state["extracted_job_name"] = validation.get("validated_job_name", "") or ""
                    state["extracted_calendar_name"] = validation.get("validated_calendar_name", "") or ""
                    state["missing_parameters"] = validation.get("missing_critical_params", [])
                    
                    # Log validation results
                    self.logger.info(f"Parameter validation: Instance={state['extracted_instance']}, Job={state['extracted_job_name']}, Calendar={state['extracted_calendar_name']}")
                    self.logger.info(f"Missing parameters: {state['missing_parameters']}")
                    
                    # Store validation details for later use
                    state["llm_analysis"]["validation"] = validation
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"Parameter validation parsing failed: {e}")
                state["missing_parameters"] = ["validation_failed"]
                
        except Exception as e:
            self.logger.error(f"LLM parameter extraction failed: {str(e)}")
            state["error"] = f"Parameter extraction failed: {str(e)}"
        
        return state

    def request_missing_params_llm_node(self, state: AutosysState) -> AutosysState:
        """LLM-generated clarification requests"""
        
        try:
            available_instances = self.db_manager.list_instances()
            instance_info = self.db_manager.get_instance_info()
            
            clarification_prompt = f"""
Generate a professional clarification request for missing Autosys parameters.

CONTEXT:
- Original User Query: "{state['user_question']}"
- Query Type: {state.get('query_type', 'unknown')}
- Missing Parameters: {state.get('missing_parameters', [])}
- Available Instances: {', '.join(available_instances)}
- LLM Analysis: {json.dumps(state.get('llm_analysis', {}), indent=2)}

REQUIREMENTS:
1. Create a professional, helpful clarification message
2. Explain clearly what information is needed and why
3. Provide 3-4 specific examples based on the user's original intent
4. Include available instance information
5. Use encouraging, supportive tone
6. Format as HTML with good visual design

INSTANCE INFORMATION:
{instance_info}

Generate an HTML clarification message that:
- Has a clear title indicating what's needed
- Explains the requirement in context of their original query
- Shows available database instances in a formatted way
- Provides realistic examples
- Uses professional styling with colors and spacing
- Encourages the user to provide the missing information

Return only the HTML:
"""
            
            response = self.llm.invoke(clarification_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up HTML if wrapped in markdown
            content = re.sub(r'```html\s*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'```\s*$', '', content)
            
            state["formatted_output"] = content
            
        except Exception as e:
            # Fallback clarification if LLM fails
            missing_params_str = ", ".join(state.get("missing_parameters", ["required information"]))
            state["formatted_output"] = f"""
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 10px 0;">
                <h4 style="margin: 0 0 10px 0; color: #856404;">Additional Information Required</h4>
                <p style="margin: 0 0 15px 0; color: #856404;">
                    To process your request "{state['user_question']}", I need: {missing_params_str}
                </p>
                <div style="background: #f8f9fa; border-radius: 4px; padding: 10px;">
                    <strong>Available Instances:</strong><br>
                    {self.db_manager.get_instance_info()}
                </div>
            </div>
            """
        
        return state

    def generate_sql_llm_node(self, state: AutosysState) -> AutosysState:
        """LLM-driven SQL generation"""
        
        try:
            sql_generation_prompt = f"""
Generate optimized Oracle SQL for Autosys database query using advanced query construction.

CONTEXT:
- User Query: "{state['user_question']}"
- Query Type: {state.get('query_type', 'general')}
- Database Instance: {state.get('extracted_instance', 'Unknown')}
- Job Name: {state.get('extracted_job_name', 'None')}
- Calendar Name: {state.get('extracted_calendar_name', 'None')}
- LLM Analysis: {json.dumps(state.get('llm_analysis', {}), indent=2)}

AUTOSYS SCHEMA:
- aedbadmin.ujo_jobst: Main job status table (job_name, status, last_start, last_end, joid)
- aedbadmin.ujo_job: Job definition table (joid, owner, machine, job_type, description)
- aedbadmin.UJO_INTCODES: Status code lookup (code, TEXT)

STATUS CODES:
- 4 = SUCCESS, 7 = FAILURE, 8 = RUNNING, 1 = INACTIVE, 9 = TERMINATED

TIME CONVERSION:
- Autosys timestamps are epoch seconds: TO_CHAR(TO_DATE('01.01.1970 19:00:00','DD.MM.YYYY HH24:Mi:Ss') + (last_start / 86400), 'MM/DD/YYYY HH24:Mi:Ss')

SQL CONSTRUCTION RULES:
1. Always include proper JOINs between tables
2. Use status code translation: LEFT JOIN aedbadmin.UJO_INTCODES ic ON js.status = ic.code
3. Apply appropriate WHERE clauses based on parameters
4. Include time-based filters for recent queries
5. Limit results with ROWNUM <= 50
6. Order by relevance (status, then time)

QUERY TYPES:
- job_details: Focus on specific job with comprehensive information
- calendar_details: Focus on calendar-related jobs and schedules
- job_status: Current status information for jobs
- job_list: List of jobs matching criteria
- general_query: Broad query based on user intent

Generate the complete, optimized SQL query. Return only the SQL:
"""
            
            response = self.llm.invoke(sql_generation_prompt)
            sql_content = response.content if hasattr(response, 'content') else str(response)
            
            # Clean SQL
            sql_content = re.sub(r'```sql\s*', '', sql_content, flags=re.IGNORECASE)
            sql_content = re.sub(r'```\s*$', '', sql_content)
            sql_content = ' '.join(sql_content.split()).strip()
            
            state["sql_query"] = sql_content
            
            self.logger.info(f"Generated SQL for {state.get('query_type', 'query')}: {len(sql_content)} characters")
            
        except Exception as e:
            state["error"] = f"SQL generation failed: {str(e)}"
            self.logger.error(f"LLM SQL generation failed: {e}")
        
        return state

    def execute_query_node(self, state: AutosysState) -> AutosysState:
        """Execute database query (same as before)"""
        try:
            instance_name = state.get("extracted_instance")
            sql_query = state.get("sql_query")
            
            if not instance_name or not sql_query:
                state["error"] = "Missing instance name or SQL query"
                return state
            
            # Get database instance
            instance = self.db_manager.get_instance(instance_name)
            if not instance:
                state["error"] = f"Database instance '{instance_name}' not found"
                return state
            
            # Execute query
            start_time = datetime.now()
            
            if hasattr(instance.autosys_db, 'run'):
                raw_results = instance.autosys_db.run(sql_query)
            else:
                raise Exception(f"Database connection method not found for instance {instance_name}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Process results
            processed_results = self._process_results(raw_results)
            
            state["query_results"] = {
                "success": True,
                "results": processed_results,
                "row_count": len(processed_results),
                "execution_time": execution_time,
                "instance_used": instance_name
            }
            
        except Exception as e:
            state["error"] = f"Query execution failed: {str(e)}"
            state["query_results"] = {"success": False}
        
        return state

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

    def format_results_llm_node(self, state: AutosysState) -> AutosysState:
        """LLM-driven result formatting"""
        
        try:
            results = state["query_results"]["results"]
            
            formatting_prompt = f"""
Create professional, visually appealing HTML presentation for Autosys query results.

CONTEXT:
- Original Query: "{state['user_question']}"
- Query Type: {state.get('query_type', 'query')}
- Database Instance: {state['query_results'].get('instance_used', 'Unknown')}
- Results Count: {len(results)}
- Execution Time: {state['query_results'].get('execution_time', 0):.2f}s
- Job Name: {state.get('extracted_job_name', 'N/A')}
- Calendar Name: {state.get('extracted_calendar_name', 'N/A')}

QUERY RESULTS:
{json.dumps(results[:10], indent=2, default=str)}

REQUIREMENTS:
1. Create responsive HTML with inline CSS
2. Use professional color scheme and typography
3. Include clear header with context information
4. Format data in readable table or card layout
5. Add status badges with appropriate colors:
   - SUCCESS: Green background
   - FAILURE/FAILED: Red background  
   - RUNNING: Blue background
   - INACTIVE: Gray background
6. Include summary statistics
7. Add metadata footer with execution details
8. Handle empty results gracefully
9. Use mobile-friendly responsive design
10. Include proper spacing and visual hierarchy

Generate complete HTML with professional styling. Return only HTML:
"""
            
            response = self.llm.invoke(formatting_prompt)
            formatted_html = response.content if hasattr(response, 'content') else str(response)
            
            # Clean HTML if wrapped in markdown
            formatted_html = re.sub(r'```html\s*', '', formatted_html, flags=re.IGNORECASE)
            formatted_html = re.sub(r'```\s*$', '', formatted_html)
            
            state["formatted_output"] = formatted_html
            
        except Exception as e:
            # Fallback formatting
            results_count = len(state["query_results"].get("results", []))
            instance_used = state["query_results"].get("instance_used", "Unknown")
            
            if results_count == 0:
                state["formatted_output"] = f"""
                <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px;">
                    <h4 style="margin: 0 0 10px 0; color: #856404;">No Results Found</h4>
                    <p style="margin: 0; color: #856404;">No records found matching your criteria in instance: <strong>{instance_used}</strong></p>
                </div>
                """
            else:
                state["formatted_output"] = f"""
                <div style="border: 1px solid #dee2e6; border-radius: 5px; padding: 15px;">
                    <h4 style="margin: 0 0 10px 0;">Query Results</h4>
                    <p>Found {results_count} results from instance: <strong>{instance_used}</strong></p>
                    <pre style="background: #f8f9fa; padding: 10px; border-radius: 3px; font-size: 11px; overflow: auto;">{json.dumps(state["query_results"]["results"][:5], indent=2, default=str)}</pre>
                </div>
                """
        
        return state

    def handle_error_llm_node(self, state: AutosysState) -> AutosysState:
        """LLM-driven error handling and user guidance"""
        
        try:
            error_handling_prompt = f"""
Create a helpful, user-friendly error message and guidance for this Autosys system error.

ERROR CONTEXT:
- User Query: "{state['user_question']}"
- Error Message: {state.get('error', 'Unknown error')}
- SQL Query: {state.get('sql_query', 'Not generated')}
- Available Instances: {', '.join(self.db_manager.list_instances())}
- System Analysis: {json.dumps(state.get('llm_analysis', {}), indent=2)}

REQUIREMENTS:
1. Create empathetic, professional error message
2. Explain what went wrong in user-friendly terms
3. Provide specific suggestions for resolution
4. Include helpful examples of correct query formats
5. Show available resources (instances, etc.)
6. Maintain encouraging tone
7. Format as professional HTML with good visual design
8. Include technical details in collapsible section if helpful

Generate supportive HTML error message. Return only HTML:
"""
            
            response = self.llm.invoke(error_handling_prompt)
            error_html = response.content if hasattr(response, 'content') else str(response)
            
            # Clean HTML
            error_html = re.sub(r'```html\s*', '', error_html, flags=re.IGNORECASE)
            error_html = re.sub(r'```\s*$', '', error_html)
            
            state["formatted_output"] = error_html
            
        except Exception as e:
            # Fallback error message if LLM fails
            error_msg = state.get("error", "Unknown error occurred")
            state["formatted_output"] = f"""
            <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;">
                <h4 style="margin: 0 0 10px 0;">System Error</h4>
                <p style="margin: 0;"><strong>Error:</strong> {error_msg}</p>
                <p style="margin: 10px 0 0 0; font-size: 12px;">Please try rephrasing your request or contact support if the issue persists.</p>
            </div>
            """
        
        return state

    # LLM-driven conditional edge functions
    def _llm_routing_decision(self, state: AutosysState) -> str:
        """LLM determines routing based on analysis"""
        analysis = state.get("llm_analysis", {})
        return "conversation" if analysis.get("is_general_conversation", False) else "database"

    def _llm_parameter_check(self, state: AutosysState) -> str:
        """LLM determines if parameters are sufficient"""
        missing_params = state.get("missing_parameters", [])
        return "needs_params" if missing_params else "has_all_params"

    def _llm_execution_check(self, state: AutosysState) -> str:
        """Check if ready for execution"""
        return "error" if state.get("error") else "execute"

    def _llm_result_check(self, state: AutosysState) -> str:
        """Check query results"""
        if state.get("error"):
            return "error"
        elif state.get("query_results", {}).get("success"):
            return "format"
        else:
            return "error"

    def query(self, user_question: str, session_id: str) -> Dict[str, Any]:
        """Main method to process user input with complete LLM analysis"""
        
        initial_state = {
            "messages": [],
            "user_question": user_question,
            "llm_analysis": {},
            "is_general_conversation": False,
            "extracted_instance": "",
            "extracted_job_name": "",
            "extracted_calendar_name": "", 
            "missing_parameters": [],
            "query_type": "",
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
                "needs_clarification": bool(final_state.get("missing_parameters", [])),
                "query_type": final_state.get("query_type", ""),
                "llm_analysis": final_state.get("llm_analysis", {}),
                "error": final_state.get("error", "")
            }
            
        except Exception as e:
            self.logger.error(f"LLM-driven system execution failed: {e}")
            return {
                "success": False,
                "formatted_output": f"""
                <div style="border: 1px solid #dc3545; background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px;">
                    <h4>System Error</h4>
                    <p>LLM-driven system execution failed: {str(e)}</p>
                </div>
                """,
                "error": str(e)
            }

# ============================================================================
# LLM-DRIVEN INTERFACE FUNCTIONS
# ============================================================================

# Global system instance
_autosys_system = None

def setup_autosys_multi_database_system(database_configs: Dict[str, Any], llm_instance):
    """
    Setup the LLM-driven multi-database Autosys system
    
    Args:
        database_configs: Dictionary of database configurations
        llm_instance: Your LLM instance
    
    Returns:
        Configured system status
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
        
        # Initialize the LLM-driven system
        _autosys_system = LLMDrivenAutosysSystem(db_manager, llm_instance)
        
        logger.info("LLM-driven multi-database Autosys system initialized successfully")
        
        return {
            "status": "ready",
            "instances": db_manager.list_instances(),
            "features": [
                "Complete LLM-driven intent analysis",
                "AI-powered parameter extraction and validation", 
                "LLM-generated clarification requests",
                "Intelligent SQL query generation",
                "AI-driven result formatting",
                "Smart error handling with user guidance",
                "Context-aware conversation handling",
                "Session persistence with memory"
            ]
        }
        
    except Exception as e:
        logger.error(f"LLM-driven system setup failed: {e}")
        raise Exception(f"Failed to initialize LLM-driven system: {str(e)}")

def get_chat_response(message: str, session_id: str) -> str:
    """LLM-driven chat function"""
    global _autosys_system
    
    try:
        if not message or not message.strip():
            message = "Hello! How can I help you today?"
        
        if not _autosys_system:
            return """
            <div style="border: 1px solid #ffc107; background: #fff3cd; color: #856404; padding: 15px; border-radius: 5px;">
                <h4>System Not Ready</h4>
                <p>The LLM-driven multi-database system is not initialized. Please contact administrator.</p>
            </div>
            """
        
        # Process message with complete LLM analysis
        result = _autosys_system.query(message.strip(), session_id)
        
        return result["formatted_output"]
        
    except Exception as e:
        logger.error(f"LLM-driven chat response error: {e}")
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
# LLM-DRIVEN USAGE EXAMPLES
# ============================================================================

def demonstrate_llm_driven_system():
    """Examples of LLM-driven behavior"""
    
    llm_driven_examples = [
        {
            "scenario": "Ambiguous Job Request",
            "user_input": "I need job information",
            "llm_analysis": "Detects job_details intent, determines job_name and instance required",
            "system_response": "LLM generates contextual clarification with examples",
            "outcome": "User provides specific job name and instance"
        },
        {
            "scenario": "Complex Calendar Query", 
            "user_input": "Show me calendar stuff for production",
            "llm_analysis": "Identifies calendar_details intent, extracts PROD instance, needs calendar_name",
            "system_response": "LLM requests specific calendar name with PROD context",
            "outcome": "Targeted query with proper parameters"
        },
        {
            "scenario": "Natural Language Status Check",
            "user_input": "Are there any failed jobs today in dev environment?",
            "llm_analysis": "Detects job_status intent, extracts DEV instance, no specific job needed",
            "system_response": "LLM generates appropriate SQL for failed jobs today in DEV",
            "outcome": "Direct execution with results"
        },
        {
            "scenario": "Casual Conversation",
            "user_input": "Thanks for your help earlier!",
            "llm_analysis": "Identifies general conversation, no database intent",
            "system_response": "LLM generates friendly conversational response",
            "outcome": "Natural conversation handling"
        }
    ]
    
    return llm_driven_examples

def main():
    """LLM-driven system demonstration"""
    print("LLM-DRIVEN Multi-Database Autosys System")
    print("========================================")
    print()
    print("COMPLETE LLM INTEGRATION:")
    print("- Intent analysis entirely LLM-driven")
    print("- Parameter extraction using AI reasoning")
    print("- SQL generation with intelligent context")
    print("- Result formatting optimized by LLM")
    print("- Error handling with AI-generated guidance")
    print("- Conversation responses naturally generated")
    print()
    
    examples = demonstrate_llm_driven_system()
    print("LLM-DRIVEN CONVERSATION FLOWS:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['scenario']}:")
        print(f"   User: '{example['user_input']}'")
        print(f"   LLM Analysis: {example['llm_analysis']}")
        print(f"   System: {example['system_response']}")
        print(f"   Result: {example['outcome']}")
        print()

if __name__ == "__main__":
    main()

# ============================================================================
# LLM-DRIVEN INTEGRATION GUIDE
# ============================================================================

"""
COMPLETE LLM-DRIVEN AUTOSYS SYSTEM

This system eliminates ALL rule-based logic and uses LLM analysis throughout:

KEY LLM INTEGRATIONS:

1. INTENT ANALYSIS (analyze_with_llm_node):
   - LLM comprehensively analyzes user intent
   - Determines conversation vs database query
   - Identifies specific query types (job_details, calendar_details, etc.)
   - Assesses parameter requirements
   - Provides confidence scoring and reasoning

2. PARAMETER EXTRACTION (extract_parameters_llm_node):
   - LLM validates and extracts all parameters
   - Applies fuzzy matching intelligence
   - Assesses extraction confidence
   - Determines missing critical parameters

3. CLARIFICATION GENERATION (request_missing_params_llm_node):
   - LLM creates contextual clarification messages
   - Generates appropriate examples
   - Formats professional HTML responses
   - Maintains encouraging, helpful tone

4. SQL GENERATION (generate_sql_llm_node):
   - LLM constructs optimized Oracle SQL
   - Incorporates all extracted parameters
   - Applies complex query logic
   - Handles various query types intelligently

5. RESULT FORMATTING (format_results_llm_node):
   - LLM creates professional HTML presentations
   - Applies appropriate styling and colors
   - Includes summary statistics
   - Handles empty results gracefully

6. ERROR HANDLING (handle_error_llm_node):
   - LLM generates user-friendly error messages
   - Provides specific resolution guidance
   - Creates supportive, professional responses

7. CONVERSATION HANDLING (handle_conversation_node):
   - LLM generates natural conversation responses
   - Maintains context and personality
   - Provides helpful system information

ADVANTAGES:
- No hardcoded rules or patterns
- Intelligent context understanding
- Natural language processing
- Adaptive to user communication styles
- Self-improving through LLM capabilities

USAGE: Drop-in replacement for existing system with same API
"""







@@@@@@@@@@@@
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
