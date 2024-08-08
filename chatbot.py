import os
from pprint import pprint
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List

# Setup environment
os.environ["GROQ_API_KEY"] = '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'

GROQ_LLM = ChatGroq(model="llama3-70b-8192")

## Utils
def write_markdown_file(content, filename):
    """Writes the given content as a markdown file to the local directory."""
    with open(f"{filename}.md", "w") as f:
        f.write(content)

# Define Prompts
video_game_prompt = PromptTemplate(
    template="""system
    You are a chatbot that only talks about video games. If the user asks about anything else, do not respond.
    user
    {user_message}
    assistant
    """,
    input_variables=["user_message"],
)

response_generator = video_game_prompt | GROQ_LLM | StrOutputParser()

# Define state
class GraphState(TypedDict):
    user_message: str
    response: str
    num_steps: int

## Nodes
def generate_response(state):
    """Generate a response for video game-related messages"""
    print("---GENERATING RESPONSE---")
    user_message = state['user_message']
    num_steps = int(state['num_steps'])
    num_steps += 1

    response = response_generator.invoke({"user_message": user_message})
    write_markdown_file(response, "response")

    return {"response": response, "num_steps": num_steps}

def state_printer(state):
    """Print the state"""
    # print("---STATE PRINTER---")
    # print(f"User Message: {state['user_message']} \n")
    # print(f"Response: {state['response']} \n")
    # print(f"Num Steps: {state['num_steps']} \n")
    return

## Conditional Edges
def is_video_game_related(state):
    """
    Determine if the user message is related to video games.
    Args:
        state (dict): The current graph state
    Returns:
        str: Next node to call
    """
    # print("---CHECKING IF VIDEO GAME RELATED---")
    user_message = state["user_message"]

    if "game" in user_message.lower() or "video" in user_message.lower():
        return "generate_response"
    elif "stop" in user_message.lower():
        return "state_printer"
    else:
        return "state_printer"

## Build the Graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate_response", generate_response)
workflow.add_node("state_printer", state_printer)

### Add Edges
workflow.set_entry_point("generate_response")

workflow.add_conditional_edges(
    "generate_response",
    is_video_game_related,
    {
        "generate_response": "state_printer",
        "state_printer": "state_printer",
    },
)
workflow.add_edge("state_printer", END)

# Compile
app = workflow.compile()

# Function to handle conversation
def chat_with_bot(user_message):
    inputs = {"user_message": user_message, "num_steps": 0}
    output = app.invoke(inputs)
    # print(output['response'])
    return output['response']

# Conversational loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "stop":
        print("Bot: Goodbye!")
        break
    response = chat_with_bot(user_input)
    print(f"Bot: {response}")