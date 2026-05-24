from dotenv import load_dotenv
from datetime import datetime
from langchain.agents import tool, AgentType, initialize_agent
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from zoneinfo import ZoneInfo
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

@tool
def get_todayDate(timezone : str):
    """ Get today's date in the '%Y-%m-%d %H:%M:%S' based on the specified timezone for example America/New_York."""    
    #print("Timezone received in tool: ", timezone)
    return datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S")

@tool
def search(query: str):
    """Search for the query using TavilySearchResults tool and return the results."""
    search_tool = TavilySearchResults(search_depth="basic")
    return search_tool.run(query)

tools= [get_todayDate, search]

agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.invoke({"input": "What is the current date and time in Quebec City?"})
