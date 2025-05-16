from openai import OpenAI
from rich import print

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small", input="The food was delicious and the waiter..."
)

print(response)
