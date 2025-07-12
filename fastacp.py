from typing import List, Dict, Callable, Optional, Union, Any, AsyncGenerator
import importlib.resources
import yaml
import json
from dataclasses import dataclass
from enum import Enum
from colorama import Fore
from acp_sdk.client import Client
from acp_sdk.models import (
    Message,
    MessagePart,
)

# === AgentCollection Implementation ===

class Agent:
    """Representation of an ACP Agent."""
    
    def __init__(self, name: str, description: str, capabilities: List[str]):
        self.name = name
        self.description = description
        self.capabilities = capabilities
    
    def __str__(self):
        return f"Agent(name='{self.name}', description='{self.description}')"


class AgentCollection:
    """
    A collection of agents available on ACP servers.
    Allows users to discover available agents on ACP servers.
    """
    
    def __init__(self):
        self.agents = []
    
    @classmethod
    async def from_acp(cls, *servers) -> 'AgentCollection':
        """
        Creates an AgentCollection by fetching agents from the provided ACP servers.
        
        Args:
            *servers: ACP server client instances to fetch agents from
            
        Returns:
            AgentCollection: Collection containing all discovered agents
        """
        collection = cls()
        
        for server in servers:
            async for agent in server.agents():
                collection.agents.append((server,agent))
        
        return collection
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """
        Find an agent by name in the collection.
        
        Args:
            name: Name of the agent to find
            
        Returns:
            Agent or None: The found agent or None if not found
        """
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None
    
    def __iter__(self):
        """Allows iteration over all agents in the collection."""
        return iter(self.agents)


# === ACPCallingAgent Implementation ===

@dataclass
class ToolCall:
    """Represents a tool call with name, arguments, and optional ID."""
    name: str
    arguments: Union[Dict[str, Any], str]
    id: Optional[str] = None


@dataclass
class ChatMessage:
    """Represents a chat message with content and optional tool calls."""
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]] = None
    raw: Any = None


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Logger:
    """Simple logger for agent operations."""
    
    def log(self, content, level=LogLevel.INFO):
        print(f"[{level.value.upper()}] {content}")
    
    def log_markdown(self, content, title=None, level=LogLevel.INFO):
        if title:
            print(f"[{level.value.upper()}] {title}")
        print(f"[{level.value.upper()}] {content}")


class AgentError(Exception):
    """Base class for agent errors."""
    
    def __init__(self, message, logger=None):
        super().__init__(message)
        if logger:
            logger.log(message, level=LogLevel.ERROR)


class AgentParsingError(AgentError):
    """Error when parsing agent output."""
    pass


class AgentToolCallError(AgentError):
    """Error when making a tool call."""
    pass


class AgentToolExecutionError(AgentError):
    """Error when executing a tool."""
    pass


class ActionStep:
    """Represents a step in the agent's reasoning process."""
    
    def __init__(self):
        self.model_input_messages = []
        self.model_output_message = None
        self.model_output = None
        self.tool_calls = []
        self.action_output = None
        self.observations = None


class Tool:
    """Base class for tools that agents can use."""
    
    def __init__(self, name, description, inputs, output_type, client=None):
        self.name = name
        self.description = description
        self.inputs = inputs
        self.output_type = output_type
        self.client = client
    
    async def __call__(self, *args, **kwargs):
        print(Fore.YELLOW + 'Tool being called with args: ' + str(args) + ' and kwargs: ' + str(kwargs) + Fore.RESET)
    
        # Extract the input content from either args or kwargs
        content = ""
        if args and isinstance(args[0], str):
            content = args[0]
        elif "prompt" in kwargs:
            content = kwargs["prompt"]
        elif "input" in kwargs:
            content = kwargs["input"]
        elif kwargs:
            # If no specific key is found, use the first value
            content = next(iter(kwargs.values()))
            
        # Now use the extracted content in your message
        print(Fore.MAGENTA + content + Fore.RESET) 
        response = await self.client.run_sync(
            agent=self.name, 
            input=content
        )
        print(Fore.RED + str(response) + Fore.RESET) 
        return response.output[0].parts[0].content


class MultiStepAgent:
    """Base class for agents that operate in multiple steps."""
    
    def __init__(
        self,
        tools: Dict[str, Tool],
        model: Callable,
        prompt_templates: Dict[str, str],
        planning_interval: Optional[int] = None,
        **kwargs
    ):
        self.tools = tools
        self.model = model
        self.prompt_templates = prompt_templates
        self.planning_interval = planning_interval
        self.managed_agents = kwargs.get("managed_agents", {})
        self.logger = Logger()
        self.state = {}
        self.input_messages = []
    
    def initialize_system_prompt(self) -> str:
        """Generate the system prompt for the agent."""
        raise NotImplementedError
    
    def write_memory_to_messages(self):
        """Convert agent memory to a list of messages for the model."""
        # Implementation would depend on memory structure
        return self.input_messages
    
    async def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """Perform one step in the agent's reasoning process."""
        raise NotImplementedError


def populate_template(template: str, variables: Dict[str, Any]) -> str:
    """Helper function to populate a template with variables."""
    result = template
    for key, value in variables.items():
        placeholder = "{" + key + "}"
        result = result.replace(placeholder, str(value))
    return result


class ACPCallingAgent(MultiStepAgent):
    """
    This agent uses JSON-like ACP agent calls, similarly to how ToolCallingAgent uses tool calls,
    but directed at remote ACP agents instead of local tools.
    
    Args:
        acp_agents (`dict[str, Agent]`): ACP agents that this agent can call.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`Dict[str, str]`], *optional*): Prompt templates.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        acp_agents: Dict[str, Agent],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        prompt_templates: Optional[Dict[str, str]] = None,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        # Default prompt templates if none provided
        if prompt_templates is None:
            prompt_templates = {
                "system_prompt": """You are a supervisory agent that delegates tasks to specialized ACP agents.
                Available agents:
                {agents}

                WORKFLOW:
                1. Analyze the user's complete request
                2. Call appropriate agent(s) to gather information and perform actions  
                3. **STOP** when the user's full request is completed
                4. **ALWAYS** call final_answer tool with a summary of what was accomplished

                COMPLETION CRITERIA - Call final_answer when:
                - All requested emails have been checked and processed
                - All required actions (delete spam, accept meetings, etc.) have been performed
                - You have a complete summary to provide to the user
                - No more actions are needed to fulfill the original request

                TOOL CALL FORMAT:
                <tool_call>
                {{
                "tool_name": "agent_name", 
                "arguments": {{"input": "clear request here"}}
                }}
                </tool_call>

                CRITICAL: 
                - Use final_answer tool when task is COMPLETE, not when you need more information
                - Do NOT keep checking for emails endlessly - check once, process results, then provide final_answer
                - The final_answer tool is the ONLY way to return results to the user
                - Never provide answers in regular messages
                """
            }
        
        # Convert ACP agents to a format similar to tools
        acp_tools = {}
        for name, agent in acp_agents.items():
            # Create a callable that will invoke the ACP agent
            acp_tools[name] = Tool(
                name=name,
                description=agent['agent'].description,
                inputs={"input": {"type":"string","description":"the prompt to pass to the agent"}},
                output_type="str",
                client=agent['client']
            )
            
            # Override the __call__ method to make it actually call the ACP agent
            def make_caller(agent_name, client):
                async def call_agent(prompt, **kwargs):
                    print(f"Calling {agent_name} with prompt: {prompt}")
                    response = await client.run_sync(
                        agent=agent_name, 
                        input=prompt
                    )
                    return response.output[0].parts[0].content
                return call_agent

            acp_tools[name].__call__ = make_caller(name, agent['client'])
        
        # Add final_answer tool
        acp_tools["final_answer"] = Tool(
            name="final_answer",
            description="Provide the final answer to the user's request",
            inputs={"answer": "The final answer to provide to the user"},
            output_type="str"
        )
        
        async def final_answer(answer, **kwargs):
            return answer
        
        acp_tools["final_answer"].__call__ = final_answer
        
        super().__init__(
            tools=acp_tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )
        
        self.acp_agents = acp_agents
    
    def initialize_system_prompt(self) -> str:
        """Generate the system prompt for the agent with ACP agent information."""
        agent_descriptions = "\n".join(
            [f"- {name}: {agent['agent'].description}" for name, agent in self.acp_agents.items()]
        )
        
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={"agents": agent_descriptions},
        )
        return system_prompt

    def save_to_memory(self, key: str, value: Any) -> None:
        """Save a value to the agent's persistent memory."""
        self.state[key] = value
        self.logger.log(f"Saved to memory: {key}={value}", level=LogLevel.ERROR)

    def _clean_json(self, json_str: str) -> str:
        """Clean up common JSON formatting issues from model output."""
        import re
        
        # If the JSON is too malformed, fall back to simple structure
        if '<!--' in json_str or '<<--' in json_str or json_str.count(':') > json_str.count(',') + 2:
            # Try to extract just the tool name and create a simple structure
            tool_name_match = re.search(r'"(?:tool_name|tool|agent)":\s*"([^"]+)"', json_str)
            if tool_name_match:
                tool_name = tool_name_match.group(1)
                # Look for any meaningful input text
                input_match = re.search(r'"(?:input|query|request)":\s*"([^"]+)"', json_str)
                if input_match:
                    input_text = input_match.group(1)
                else:
                    input_text = "check emails"  # fallback
                
                return f'{{"tool_name": "{tool_name}", "arguments": {{"input": "{input_text}"}}}}'
        
        # Standard cleaning for minor issues
        # Remove HTML/XML comments
        json_str = re.sub(r'<!--.*?-->', '', json_str, flags=re.DOTALL)
        json_str = re.sub(r'<<--.*?-->', '', json_str, flags=re.DOTALL)
        
        # Remove trailing commas before closing braces
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str.strip()


    async def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the reasoning process: the agent thinks, calls ACP agents, and observes results.
        Returns None if the step is not final.
        """
        # print(f"{self.write_memory_to_messages()} \n\n")
        # # Convert messages to LiteLLM format
        # memory_messages = self.write_memory_to_messages()
        # # Make sure all messages are in the correct LiteLLM format
        # for i, message in enumerate(memory_messages):
        #     if "content" in message and not isinstance(message["content"], list):
        #         memory_messages[i]["content"] = [{"type": "text", "text": message["content"]}]


        # self.input_messages = memory_messages
        # memory_step.model_input_messages = memory_messages.copy()

        # # 1️⃣  ---- Convert whatever is in memory to litellm.Message ----------
        # raw_memory = self.write_memory_to_messages()
    
        # def _to_message(m) -> Message:
        #     # Already the right type
        #     if isinstance(m, Message):
        #         return m
    
        #     # Legacy dicts ---------------------------------------------
        #     role = m.get("role", "user")
        #     content_field = m.get("content", "")
    
        #     # Legacy code sometimes stores content as a list of parts
        #     if isinstance(content_field, list) and content_field:
        #         # expect [{'type':'text','text': ...}]
        #         if isinstance(content_field[0], dict) and "text" in content_field[0]:
        #             content_str = content_field[0]["text"]
        #         else:
        #             # join anything that looks like text
        #             content_str = " ".join(str(p) for p in content_field)
        #     else:
        #         content_str = str(content_field)

        #     print(f"role: {role} \n content_str: {content_str}\n\n")
    
        #     return Message(role=role, content=content_str)
        #     # -----------------------------------------------------------
    
        # memory_messages = [_to_message(m) for m in raw_memory]

        # print("="*20)
    
        # # Debug (optional)
        # for msg in memory_messages:
        #     print("→", msg.role, repr(msg.content))
    
        # self.input_messages = memory_messages
        # memory_step.model_input_messages = list(memory_messages)  # shallow copy ok

        from litellm import Message

        def _to_msg(m) -> Message:
            if isinstance(m, Message):
                return m  # Already valid
        
            role = m.get("role", "user")
            content = m.get("content", "")
        
            # Unwrap `content` from old structure
            if isinstance(content, list) and content and isinstance(content[0], dict):
                text = content[0].get("text", "")
            else:
                text = str(content)
        
            # Use .content instead of .parts
            if role in {"user", "assistant", "system", "tool", "function"}:
                return Message(role=role, content=text)
            else:
                return Message(role=f"agent/{role}", content=text)

        # ────────────────────────────────────────────────────────────────
        
        # -- use it ------------------------------------------------------
        raw_memory      = self.write_memory_to_messages()
        memory_messages = [_to_msg(m) for m in raw_memory]
        
        self.input_messages              = memory_messages
        memory_step.model_input_messages = list(memory_messages)  # shallow copy


        
        try:  
            model_message: ChatMessage = self.model(
                memory_messages,
                tools_to_call_from=list(self.tools.values())[:-1],
                stop_sequences=["Observation:", "Calling agents:"],
            ) 
            print(f"\nmodel_message:\n {model_message} \n----------------\n")
            memory_step.model_output_message = model_message
        except Exception as e:
            raise AgentParsingError(f"Error while generating or parsing output:\n{e}", self.logger) from e
        
        self.logger.log_markdown(
            content=model_message.content if model_message.content else str(model_message.raw),
            title="Output message of the LLM:",
            level=LogLevel.ERROR,
        )
        
        # Check if the model called any tools/agents
        if not hasattr(model_message, 'tool_calls') or model_message.tool_calls is None or len(model_message.tool_calls) == 0:
            # If no tool calls, treat content as final answer
            if model_message.content and "final_answer" in model_message.content.lower():
                # Extract just the response content from final_answer tool calls
                try:
                    import re
                    # Extract JSON from XML-like tool call format
                    tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', model_message.content, re.DOTALL)
                    if tool_call_match:
                        tool_call_json = tool_call_match.group(1)
                        tool_call_data = json.loads(tool_call_json)
                        
                        # Extract arguments/parameters from the tool call
                        args = tool_call_data.get("arguments", 
                                 tool_call_data.get("function_arg", 
                                   tool_call_data.get("data", 
                                     tool_call_data.get("parameters", 
                                       tool_call_data.get("tool_input", 
                                         tool_call_data.get("action_input",
                                           tool_call_data.get("message", {})))))))
                        
                        # Find the actual answer content in the arguments
                        if isinstance(args, str):
                            clean_answer = args
                        elif isinstance(args, dict):
                            # Look for common answer field names
                            clean_answer = (args.get("answer") or 
                                          args.get("response") or 
                                          args.get("message") or 
                                          args.get("data") or
                                          args.get("content") or
                                          next(iter(args.values())) if args else "")
                        else:
                            clean_answer = str(args)
                        
                        if clean_answer:
                            # Unescape newlines
                            clean_answer = clean_answer.replace('\\n', '\n')
                            self.logger.log(f"Final answer extracted: {clean_answer}", level=LogLevel.INFO)
                            memory_step.action_output = clean_answer
                            return clean_answer
                except Exception as e:
                    self.logger.log(f"Error parsing final answer JSON: {e}", level=LogLevel.DEBUG)
                
                self.logger.log(
                    f"Final answer detected in content: {model_message.content}",
                    level=LogLevel.INFO,
                )
                memory_step.action_output = model_message.content
                return model_message.content
            else:
                # Try to extract a tool call from the content using a simple parser
                content = model_message.content or ""
                if "<tool_call>" in content and "</tool_call>" in content:
                    try:
                        # Extract JSON from XML-like tool call format
                        import re
                        tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
                        if tool_call_match:
                            tool_call_json = tool_call_match.group(1)
                            
                            # Clean up common JSON formatting issues
                            tool_call_json = self._clean_json(tool_call_json)
                            
                            tool_call_data = json.loads(tool_call_json)
                            
                            agent_name = tool_call_data.get("tool_name", 
                                       tool_call_data.get("tool", 
                                         tool_call_data.get("function_name", 
                                           tool_call_data.get("agent_name",
                                             tool_call_data.get("agent", 
                                               tool_call_data.get("name", ""))))))
                            agent_arguments = tool_call_data.get("arguments", 
                                             tool_call_data.get("function_arg", 
                                               tool_call_data.get("data", 
                                                 tool_call_data.get("parameters", 
                                                   tool_call_data.get("tool_input", 
                                                     tool_call_data.get("task", 
                                                       tool_call_data.get("function", 
                                                         tool_call_data.get("action_input",
                                                           tool_call_data.get("message", {})))))))))
                            
                            # Convert arguments to the format expected by ACP agents
                            if isinstance(agent_arguments, str):
                                # If it's a direct string, use it as input
                                agent_arguments = {"input": agent_arguments}
                            elif isinstance(agent_arguments, dict):
                                # Convert any parameter to "input" which is what ACP agents expect
                                if "email_address" in agent_arguments:
                                    agent_arguments = {"input": agent_arguments["email_address"]}
                                elif "prompt" not in agent_arguments and "input" not in agent_arguments:
                                    # If no standard input field, use the first value or convert the whole dict to string
                                    if len(agent_arguments) == 1:
                                        agent_arguments = {"input": next(iter(agent_arguments.values()))}
                                    else:
                                        agent_arguments = {"input": str(agent_arguments)}
                            else:
                                # Handle other types (numbers, etc.) by converting to string
                                agent_arguments = {"input": str(agent_arguments)}
                            
                            # Create a synthetic tool call
                            memory_step.model_output = str(f"Called Agent: '{agent_name}' with arguments: {agent_arguments}")
                            memory_step.tool_calls = [ToolCall(name=agent_name, arguments=agent_arguments, id="synthetic_id")]
                            
                            # Process the extracted tool call below
                            return await self._process_tool_call(memory_step, agent_name, agent_arguments)
                    except Exception as e:
                        self.logger.log(f"Error parsing XML tool call: {e}", level=LogLevel.ERROR)
                elif "tool:" in content.lower() or "agent:" in content.lower():
                    try:
                        # Fallback: Very simple extraction for other formats
                        lines = content.split('\n')
                        tool_line = next((line for line in lines if "tool:" in line.lower() or "agent:" in line.lower()), None)
                        if tool_line:
                            parts = tool_line.split(":", 1)
                            if len(parts) > 1:
                                agent_name = parts[1].strip()
                                # Find arguments
                                arg_line = next((line for line in lines if "arguments:" in line.lower()), None)
                                agent_arguments = {}
                                if arg_line:
                                    arg_parts = arg_line.split(":", 1)
                                    if len(arg_parts) > 1:
                                        try:
                                            # Try to parse as JSON
                                            agent_arguments = json.loads(arg_parts[1].strip())
                                        except json.JSONDecodeError:
                                            # If not valid JSON, use as string
                                            agent_arguments = {"input": arg_parts[1].strip()}
                                else:
                                    # If no arguments line, use rest of content as prompt
                                    remaining_content = "\n".join(lines[lines.index(tool_line)+1:])
                                    agent_arguments = {"input": remaining_content.strip()}
                                
                                # Create a synthetic tool call
                                memory_step.model_output = str(f"Called Agent: '{agent_name}' with arguments: {agent_arguments}")
                                memory_step.tool_calls = [ToolCall(name=agent_name, arguments=agent_arguments, id="synthetic_id")]
                                
                                # Process the extracted tool call below
                                return await self._process_tool_call(memory_step, agent_name, agent_arguments)
                    except json.JSONDecodeError as e:
                        self.logger.log(f"JSON parsing error in tool call: {e}", level=LogLevel.ERROR)
                        self.logger.log(f"Problematic JSON: {tool_call_json[:200]}...", level=LogLevel.DEBUG)
                    except Exception as e:
                        self.logger.log(f"Error parsing tool call from content: {e}", level=LogLevel.ERROR)
                
                raise AgentParsingError(
                    "Model did not call any agents and no final answer detected. Content: " + (model_message.content or "None"), 
                    self.logger
                )
        
        # Process the tool call
        tool_call = model_message.tool_calls[0]
        
        # Handle different tool call format structures
        if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
            # Standard OpenAI-like format
            agent_name = tool_call.function.name
            agent_arguments = tool_call.function.arguments
            tool_call_id = getattr(tool_call, 'id', 'unknown_id')
        elif hasattr(tool_call, 'name'):
            # Simplified format
            agent_name = tool_call.name
            agent_arguments = getattr(tool_call, 'arguments', {})
            tool_call_id = getattr(tool_call, 'id', 'unknown_id')
        else:
            # Try to parse as dict
            agent_name = tool_call.get('name', tool_call.get('function', {}).get('name', 'unknown'))
            agent_arguments = tool_call.get('arguments', tool_call.get('function', {}).get('arguments', {}))
            tool_call_id = tool_call.get('id', 'unknown_id')
            
        memory_step.model_output = str(f"Called Agent: '{agent_name}' with arguments: {agent_arguments}")
        memory_step.tool_calls = [ToolCall(name=agent_name, arguments=agent_arguments, id=tool_call_id)]
        
        return await self._process_tool_call(memory_step, agent_name, agent_arguments)
        
    async def _process_tool_call(self, memory_step: ActionStep, agent_name: str, agent_arguments: Any) -> Union[None, Any]:
        """
        Process a tool call with the given name and arguments.
        """
        
        # Execute the tool call
        self.logger.log(
            f"Calling agent: '{agent_name}' with arguments: {agent_arguments}",
            level=LogLevel.INFO,
        )
        
        if agent_name == "final_answer":
            # Handle the final answer
            if isinstance(agent_arguments, dict):
                if "answer" in agent_arguments:
                    answer = agent_arguments["answer"]
                elif "response" in agent_arguments:
                    answer = agent_arguments["response"]
                elif "data" in agent_arguments:
                    answer = agent_arguments["data"]
                elif "tool_input" in agent_arguments:
                    answer = agent_arguments["tool_input"]
                elif "message" in agent_arguments:
                    answer = agent_arguments["message"]
                elif "action_input" in agent_arguments:
                    answer = agent_arguments["action_input"]
                elif "content" in agent_arguments:
                    answer = agent_arguments["content"]
                else:
                    # Take the first string value from the dict
                    answer = next((v for v in agent_arguments.values() if isinstance(v, str)), str(agent_arguments))
            else:
                answer = agent_arguments
            
            # If answer refers to a state variable, use that
            if isinstance(answer, str) and answer in self.state:
                final_answer = self.state[answer]
                self.logger.log(
                    f"Final answer: Extracting key '{answer}' from state to return value '{final_answer}'.",
                    level=LogLevel.INFO,
                )
            else:
                final_answer = answer
                self.logger.log(
                    f"Final answer: {final_answer}",
                    level=LogLevel.INFO,
                )
            
            memory_step.action_output = final_answer
            return final_answer
        else:
            # Execute the ACP agent call
            if agent_arguments is None:
                agent_arguments = {}
            
            observation = await self.execute_tool_call(agent_name, agent_arguments)
            updated_information = str(observation).strip()

            self.save_to_memory(f"{agent_name}_response", updated_information)
            
            self.logger.log(
                f"Observations: {updated_information}",
                level=LogLevel.INFO,
            )
            
            memory_step.observations = updated_information
            return None
    
    def _substitute_state_variables(self, arguments: Union[Dict[str, str], str]) -> Union[Dict[str, Any], str]:
        """Replace string values in arguments with their corresponding state values if they exist."""
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value) if isinstance(value, str) else value
                for key, value in arguments.items()
            }
        return arguments
    
    async def execute_tool_call(self, agent_name: str, arguments: Union[Dict[str, str], str]) -> Any:
        """
        Execute an ACP agent call with the provided arguments.
        
        Args:
            agent_name (`str`): Name of the ACP agent to call.
            arguments (dict[str, str] | str): Arguments passed to the agent call.
        """
        # Check if the agent exists
        available_tools = {**self.tools}
        if agent_name not in available_tools:
            raise AgentToolExecutionError(
                f"Unknown agent {agent_name}, should be one of: {', '.join(available_tools)}.", self.logger
            )
        
        # Get the tool and substitute state variables in arguments
        tool = available_tools[agent_name]
        arguments = self._substitute_state_variables(arguments)
        
        try:
            # Call agent with appropriate arguments
            if isinstance(arguments, dict):
                return await tool(**arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, str):
                return await tool(arguments, sanitize_inputs_outputs=True)
            else:
                raise TypeError(f"Unsupported arguments type: {type(arguments)}")
                
        except TypeError as e:
            # Handle invalid arguments
            description = getattr(tool, "description", "No description")
            error_msg = (
                f"Invalid call to agent '{agent_name}' with arguments {json.dumps(arguments)}: {e}\n"
                "You should call this agent with correct input arguments.\n"
                f"Expected inputs: {json.dumps(tool.inputs)}\n"
                f"Returns output type: {tool.output_type}\n"
                f"Agent description: '{description}'"
            )
            raise AgentToolCallError(error_msg, self.logger) from e
            
        except Exception as e:
            # Handle execution errors
            error_msg = (
                f"Error executing agent '{agent_name}' with arguments {json.dumps(arguments)}: {type(e).__name__}: {e}\n"
                "Please try again or use another agent"
            )
            raise AgentToolExecutionError(error_msg, self.logger) from e

    async def run(self, query: str, max_steps: int = 10) -> str:
        """
        Run the agent to completion with a user query.
        
        Args:
            query (str): The user's query or request
            max_steps (int): Maximum number of steps before giving up, default 10
            
        Returns:
            str: Final answer from the agent
        """
        # Initialize memory with the user query in the correct format for LiteLLM
        user_message = {"role": "user", "content": [{"type": "text", "text": query}]}
        system_message = {"role": "system", "content": [{"type": "text", "text": self.initialize_system_prompt()}]}
        self.input_messages = [system_message, user_message]
        
        # Run steps until we get a final answer or hit max steps
        result = None
        for step_num in range(max_steps):
            self.logger.log(f"Step {step_num + 1}/{max_steps}", level=LogLevel.INFO)

            # Add memory context to the messages if we have any state
            if self.state and step_num > 0:
                memory_context = "Current memory state:\n"
                for key, value in self.state.items():
                    memory_context += f"- {key}: {value}\n"
                
                self.input_messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": memory_context}]
                })
            
            # Create a new action step and execute it
            memory_step = ActionStep()
            
            try:
                result = await self.step(memory_step)
                
                # If we got a final result, return it
                if result is not None:
                    return result
                
                # Otherwise, add observation to input messages for next step
                if hasattr(memory_step, 'observations') and memory_step.observations:
                    # Add assistant message if it exists
                    if hasattr(memory_step, 'model_output_message') and memory_step.model_output_message:
                        content = ""
                        if hasattr(memory_step.model_output_message, 'content'):
                            content = memory_step.model_output_message.content
                        elif hasattr(memory_step.model_output_message, 'raw'):
                            content = str(memory_step.model_output_message.raw)
                        
                        if content:
                            self.input_messages.append({
                                "role": "assistant",
                                "content": [{"type": "text", "text": content}]
                            })
                    
                    # Add observation as user message
                    self.input_messages.append({
                        "role": "user", 
                        "content": [{"type": "text", "text": f"Observation: {memory_step.observations}"}]
                    })
            except Exception as e:
                self.logger.log(f"Error in step {step_num + 1}: {str(e)}", level=LogLevel.ERROR)
                # Add error message to conversation
                self.input_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": f"Error occurred: {str(e)}. Please try a different approach or provide a final answer."}]
                })
        
        # If we hit max steps without a final answer
        return "I wasn't able to complete this task within the maximum number of steps."