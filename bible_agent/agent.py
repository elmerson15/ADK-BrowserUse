from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
import asyncio
import os
from dotenv import load_dotenv
import logging
from google.adk.agents import Agent

# Setup Logging
logging.getLogger("browser_use").setLevel(logging.DEBUG)
load_dotenv()

async def main() -> dict:
    # Get task prompt from the user
    user_prompt = input()

    # Initialize Gemini API
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Create the agent with the raw user prompt
    agent = Agent(
        task=user_prompt,
        llm=llm
    )
if __name__ == "__main__":
    asyncio.run(main())


root_agent = Agent(
    name="bible_agent",
    model="gemini-2.0-flash",
    instruction=""" go to the main() and in the user_prompt give the query and  """,
    tools=[main],
    output_key="generated_answer",  # Save result to state
)
