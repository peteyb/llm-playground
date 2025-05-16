from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "web_search_preview"}],
    input="Tell me about a UK based company called The Key Suppoer Services Ltd?",
)

print(response.output_text)
