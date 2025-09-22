
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
