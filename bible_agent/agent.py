from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent as Agent1
# from browser_use.logging_config import setup_logging
import asyncio
import os
from dotenv import load_dotenv
from google.adk.agents import Agent as Agent2
# import logging
# from google.adk.tools import google_search

# Setup Logging
# logging.getLogger("browser_use").setLevel(logging.DEBUG)
# setup_logging()
load_dotenv()

async def main():
    # Get task prompt from the user
    user_prompt = input()

    # Initialize Gemini API
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # Create the agent with the raw user prompt
    agent = Agent1(
        task=user_prompt,
        llm=llm
    )
if __name__ == "__main__":
    asyncio.run(main())


root_agent = Agent2(
    name="bible_agent",
    model="gemini-2.0-flash",
    instruction=""" Get the prompt from the user and send it to main() then the browsr_use automate the process  """,
    tools=[main],
)
