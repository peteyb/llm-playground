from openai import OpenAI

client = OpenAI()

response = client.images.generate(
    prompt="A man holding a fishing rod on a boat catching a large tuna fish",
    n=2,
    size="1024x1024",
)

print(response.data[0].url)
