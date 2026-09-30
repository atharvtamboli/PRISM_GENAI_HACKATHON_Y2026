import asyncio
from livekit.plugins import openai

from livekit.agents import llm as agents_llm
async def test_llm():
    try:
        llm = openai.LLM(model="gpt-4o-mini")
        print("LLM initialized")
        ctx = agents_llm.ChatContext()
        ctx.append(role="user", text="Hello")
        stream = llm.chat(chat_ctx=ctx)
        async for chunk in stream:
            print(chunk.choices[0].delta.content, end="")
        print("\nSuccess")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test_llm())
