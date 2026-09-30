from livekit.agents import llm

class MyTools(llm.Toolset):
    pass

@llm.function_tool(description="test")
async def test_tool():
    pass

print(type(test_tool))
print(isinstance(test_tool, llm.Tool))
