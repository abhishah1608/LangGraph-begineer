from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from datetime import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

@tool
def get_todayDate(format="%Y-%m-%d %H:%M:%S"):
    """ Get today's date in the specified format. Default format is 'YYYY-MM-DD HH:MM:SS'.
    """    
    return datetime.now().strftime(format)

search = TavilySearchResults(search_depth="basic")
tools = [search, get_todayDate]
agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.invoke({"input": "When was SpaceX's last launch and how many days ago was that from this instant?"})
