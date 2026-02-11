import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import asyncio
import json
import requests
from typing import TypedDict

from langgraph.graph import StateGraph, END

from mcp import ClientSession
from mcp.client.sse import sse_client


# ---------------- CONFIG ----------------
MCP_SERVER_URL = "http://localhost:8000/sse"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"
# ----------------------------------------


# ---- STATE ----
class AgentState(TypedDict):
    user_input: str
    llm_output: str
    tool_result: str
    session: object
    tools: list

# ---- OLLAMA CALL ----
def ask_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]

def ask_ollama_structured(prompt: str, tools: list) -> str:
    tool_names = [t.name for t in tools]

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "format": "json",   # Force valid JSON output
            "stream": False
        }
    )

    return response.json()["response"]



# ---- LLM NODE ----
def llm_node(state: AgentState):
    user_input = state["user_input"]
    tools = state["tools"]

    tool_descriptions = ""

    for tool in tools:
        tool_descriptions += f"""
Tool Name: {tool.name}
Description: {tool.description}
"""

    prompt = f"""
You are an ocean data assistant.

You have access to the following tools:

{tool_descriptions}

Your task is to select the correct tool and generate valid arguments.

If the query requires database access, you MUST return a tool call.
If no tool is needed, return:
{{"tool": "ping", "arguments": {{}}}}

CRITICAL ARGUMENT RULES:

1. Use ONLY the keys defined in the schema below.
2. Do NOT invent new keys.
3. Do NOT use generic keys like "temperature" or "depth".
4. If a field is not explicitly mentioned, set it to null.
5. Do NOT use values like "none", "all", "unknown", or repeat generic words.
6. Only assign a value to "ocean" if a specific ocean name is mentioned 
   (e.g., Indian Ocean, Pacific Ocean, Atlantic Ocean).
   If not explicitly mentioned, set ocean to null.

Mapping Rules:
- "temperature above X" → min_temperature = X
- "temperature below X" → max_temperature = X
- "depth greater than X" → min_depth = X
- "after YEAR" → recent_after_year = YEAR

When calling semantic_float_query, arguments MUST follow this EXACT structure:

{{
"ocean": string | null,
"float_type": string | null,
"min_depth": number | null,
"min_temperature": number | null,
"max_temperature": number | null,
"qc_required": boolean | null,
"data_mode": string | null,
"recent_after_year": number | null
}}

All fields MUST be present.
If not used, set them to null.

Return ONLY valid JSON.
Return exactly ONE tool call.

User: {user_input}
"""


    structured_output = ask_ollama_structured(prompt, tools)

    return {"llm_output": structured_output}



def response_llm_node(state: AgentState):

    user_query = state["user_input"]
    tool_output = state.get("tool_result", "")

    prompt = f"""
You are an ocean data assistant.

User Question:
{user_query}

Tool Result:
{tool_output}

Generate a clear and well-structured response in an easy-to-read format.

Base the answer strictly and only on the data returned from the MCP output. Do not add, assume, infer, or fabricate any information that is not present in the tool response.

If no floats are returned, clearly state that no floats matched the specified criteria. Do not generate alternative interpretations or unrelated information.

Include complete float details wherever available. Avoid unnecessary trimming of information, but summarize only if required for clarity.

Do not reference any tools or internal processing.

Do not include latitude or longitude unless explicitly requested.

Do not use greetings or conversational fillers. Provide the response directly and clearly.

The Float ID is the Platform Number.

Expand data modes as follows:

A — Adjusted
R — Real-time
D — Delayed

If more than five floats are returned, display only the first five.
If five or fewer floats are returned, display all of them.

Do not assume or fabricate any missing field values. If a field is absent in the output, omit it from the response.

BCG stands for Biogeochemical floats.
"""


    final_answer = ask_ollama(prompt)

    return {"tool_result": final_answer}



# ---- TOOL NODE ----
async def tool_node(state: AgentState):
    try:
        tool_call = json.loads(state["llm_output"])
        tool_name = tool_call["tool"]
        arguments = tool_call.get("arguments", {})

        session = state["session"]
        tools = state["tools"]

        print("\nCalling tool:")
        print("Name:", tool_name)
        print("Raw Arguments:", arguments)

        # ---- Validate Tool Exists ----
        tool = next((t for t in tools if t.name == tool_name), None)

        if not tool:
            return {"tool_result": f"Invalid tool requested: {tool_name}"}

        # ---- Sanitize LLM Output ----
        for key, value in arguments.items():
            if value == "null":
                arguments[key] = None

        # Convert numeric strings safely
        numeric_fields = [
            "min_temperature",
            "max_temperature",
            "min_depth",
            "recent_after_year"
        ]

        for field in numeric_fields:
            if field in arguments and arguments[field] is not None:
                try:
                    if field == "recent_after_year":
                        arguments[field] = int(arguments[field])
                    else:
                        arguments[field] = float(arguments[field])
                except (ValueError, TypeError):
                    return {
                        "tool_result": f"Invalid numeric value for {field}: {arguments[field]}"
                    }

        # ---- Detect if tool expects wrapped input ----
        input_schema = tool.inputSchema or {}
        properties = input_schema.get("properties", {})

        if len(properties) == 1 and "input" in properties:
            # Tool expects { "input": {...} }
            payload = {"input": arguments}
        else:
            # Tool expects flat arguments
            payload = arguments

        print("Final Payload Sent to MCP:", payload)

        # ---- Call Tool ----
        result = await session.call_tool(tool_name, payload)

        # ---- Extract Content Safely ----
        if hasattr(result, "content") and result.content:
            return {"tool_result": result.content[0].text}

        return {"tool_result": str(result)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"tool_result": f"Tool execution error: {str(e)}"}



def router(state: AgentState):
    try:
        data = json.loads(state["llm_output"])
        if "tool" in data:
            return "tool"
    except:
        pass
    return END


# ---- BUILD GRAPH ----
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("llm", llm_node)
    builder.add_node("tool", tool_node)
    builder.add_node("response_llm", response_llm_node)


    builder.set_entry_point("llm")
    builder.add_conditional_edges(
    "llm",
    router,
    {
        "tool": "tool",
        END: END
    }
    )
    # builder.add_edge("tool", END)
    
    builder.add_edge("tool", "response_llm")
    builder.add_edge("response_llm", END)

    return builder.compile()


# ---- MAIN LOOP ----
async def main():
    print("Starting Ocean Agent...")
    print(f"Connecting to MCP server at {MCP_SERVER_URL} ...")

    try:
        async with sse_client(MCP_SERVER_URL) as (read_stream, write_stream):
            print("Connected to MCP transport.")

            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                print("MCP session initialized successfully.")

                # ✅ Fetch tools ONCE
                tools_response = await session.list_tools()
                available_tools = tools_response.tools

                graph = build_graph()

                print("LangGraph Ocean Agent Ready. Type 'exit' to quit.")

                while True:
                    user_input = await asyncio.to_thread(input, "\nAsk: ")

                    if user_input.lower() == "exit":
                        print("Shutting down agent...")
                        break

                    result = await graph.ainvoke({
                        "user_input": user_input,
                        "session": session,
                        "tools": available_tools  # ✅ PASS TO STATE
                    })

                    print(
                        "\nResponse:\n",
                        result.get("tool_result", result.get("llm_output"))
                    )

    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
