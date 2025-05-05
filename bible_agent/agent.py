import os
import asyncio
import argparse
import logging
from typing import Optional
from dotenv import load_dotenv
from pydantic import SecretStr
from google.adk.agents import Agent
from google.adk.cli.fast_api import get_fast_api_app
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent as BrowserAgent, Browser, BrowserContextConfig, BrowserConfig
from browser_use.browser.browser import BrowserContext
from fastapi import FastAPI
import uvicorn

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s")
logger = logging.getLogger("ADKBrowserIntegration")

# Recovery Prompt for Browser Actions
RECOVERY_PROMPT = """
If an action fails multiple times (e.g., 3 times) or the screenshot is unchanged after 3 attempts,
do not repeat the same action. Use the `go_back` action and try a different navigation path or search query.
"""

async def setup_browser(headless: bool = False):
    """Initialize and configure the browser."""
    browser = Browser(
        config=BrowserConfig(headless=headless)
    )
    context_config = BrowserContextConfig(
        wait_for_network_idle_page_load_time=3.0,
        highlight_elements=True,
        viewport_expansion=500
    )
    browser_context = BrowserContext(browser=browser, config=context_config)
    return browser, browser_context

async def execute_browser_task(task: dict, browser_context: BrowserContext, llm: ChatGoogleGenerativeAI) -> dict:
    """Execute a browser task using browser_use."""
    query = task.get("query", "")
    initial_url = task.get("initial_url", None)
    initial_actions = [{"open_tab": {"url": initial_url}}] if initial_url else None

    browser_agent = BrowserAgent(
        task=query,
        llm=llm,
        browser_context=browser_context,
        use_vision=True,
        generate_gif=False,
        initial_actions=initial_actions,
        max_failures=3
    )

    result_history = await browser_agent.run()
    final_result = result_history.final_result() if result_history else None
    success_status = result_history.is_successful() if result_history else "Unknown"
    logger.info(f"Browser task finished. Success: {success_status}. Result: {final_result}")
    
    return {
        "status": "success" if success_status else "error",
        "result": final_result or "No result returned"
    }

def browser_task_tool(query: str, initial_url: Optional[str] = None) -> dict:
    """Tool to execute browser tasks via browser_use."""
    logger.info(f"Executing browser_task_tool with query: {query}, initial_url: {initial_url}")
    loop = asyncio.get_event_loop()
    browser, browser_context = loop.run_until_complete(setup_browser(headless=True))
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        api_key=SecretStr(os.getenv("GEMINI_API_KEY"))
    )
    
    try:
        result = loop.run_until_complete(
            execute_browser_task(
                {"query": query, "initial_url": initial_url},
                browser_context,
                llm
            )
        )
    finally:
        loop.run_until_complete(browser.close())
    
    return result

# Define ADK Agent with Callable Tool
root_agent = Agent(
    name="browser_automation_agent",
    model="gemini-2.0-flash",
    description="Processes user queries and automates browser tasks like playing YouTube videos or searching.",
    tools=[browser_task_tool]  # Direct callable function
)

# FastAPI Integration for Web Interface
logger.info("Initializing FastAPI app with agent_dir: browser_automation_agent")
app = get_fast_api_app(agent_dir="browser_automation_agent", session_db_url="sqlite:///sessions.db", web=True)

async def main():
    load_dotenv()
    os.environ["ANONYMIZED_TELEMETRY"] = "false"
    logger.info("Starting main function")

    parser = argparse.ArgumentParser(description="Run ADK agent with browser_use integration.")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument("--query", type=str, help="Single query to process.")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server (default: 8000)")
    args = parser.parse_args()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY not found.")
        return

    if args.query:
        logger.info(f"Processing single query: {args.query}")
        browser, browser_context = await setup_browser(headless=args.headless)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            api_key=SecretStr(gemini_api_key)
        )
        try:
            result = await execute_browser_task(
                {"query": args.query, "initial_url": "https://www.youtube.com"},
                browser_context,
                llm
            )
            print("\n--- FINAL RESULT ---")
            print(result["result"])
        finally:
            await browser.close()
    else:
        logger.info(f"Starting FastAPI server on port {args.port}")
        print(f"Run with `adk web --app browser_automation_agent` or `uvicorn browser_automation_agent.agent:app --host 0.0.0.0 --port {args.port}` for interactive mode.")
        uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    asyncio.run(main())