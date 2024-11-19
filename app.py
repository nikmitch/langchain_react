from flask import Flask, request, jsonify, send_from_directory
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import PromptTemplate
import os

from langchain.callbacks import FileCallbackHandler

# Create a callback class that captures intermediate thoughts, actions, and observations
class CaptureIntermediateStepsCallback(FileCallbackHandler):
    def __init__(self):
        self.steps = []

    def on_tool_start(self, tool_name, tool_input, **kwargs):
        self.steps.append({"thought": f"Using tool: {tool_name}", "action_input": tool_input})

    def on_tool_end(self, tool_output, **kwargs):
        self.steps[-1]["observation"] = tool_output

    def on_thought(self, thought, **kwargs):
        self.steps.append({"thought": thought})

    def get_steps(self):
        return self.steps

app = Flask(__name__)

llm = ChatOpenAI(model_name="gpt-4o-mini")
api_wrapper = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tools = [wikipedia_tool]

prompt_template = PromptTemplate.from_template(
    '''Answer the following questions as best you can.
    You have access to the following tools:
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    Begin!
    Question: {input}
    Thought:{agent_scratchpad}'''
)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/query', methods=['POST'])
def query_agent():
    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        callback = CaptureIntermediateStepsCallback()
        agent_executor.invoke({"input": query}, callbacks=[callback])


        # Execute the agent and capture the detailed reasoning
        final_answer = agent_executor.invoke({"input": query})['output']
        detailed_reasoning = callback.get_steps()
        print("test")
        print(detailed_reasoning)

        # Format the detailed reasoning for display
        detailed_reasoning_str = "\n".join(
            f"Thought: {step.get('thought', '')}\nAction: {step.get('action', '')}\nAction Input: {step.get('action_input', '')}\nObservation: {step.get('observation', '')}\n"
            for step in detailed_reasoning
        )
        print(detailed_reasoning_str)

        return jsonify({
            "response": final_answer,
            "details": detailed_reasoning_str
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)