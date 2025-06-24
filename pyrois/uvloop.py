# dummy uvloop for Windows
import asyncio

def install():
    pass

asyncio.set_event_loop_policy = lambda policy: None
