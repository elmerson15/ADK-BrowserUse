from browser_use import Agent as BrowserAgent
# Check if __init__ is wrapped
if hasattr(BrowserAgent.__init__, '__wrapped__'):
    print(BrowserAgent.__init__.__wrapped__.__code__.co_varnames)
else:
    print(BrowserAgent.__init__.__code__.co_varnames)