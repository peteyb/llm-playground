import json
from urllib.parse import quote_plus

import requests
from openai import OpenAI


def get_keystone_organisation(name):
    url = f"https://kauth.qa-keylabs.com/api/v2/organisations/?name={quote_plus(name)}"
    print(f"URL: {url}")
    response = requests.get(url)
    data = response.json()
    if not data["results"]:
        raise ValueError("No organisation found")
    return data["results"][0]


client = OpenAI()

tools = [
    {
        "type": "function",
        "name": "get_keystone_organisation",
        "description": "Retrieves information about a keystone organisation based on its name.",
        "strict": True,
        "parameters": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the organisation to retrieve information for.",
                }
            },
            "additionalProperties": False,
        },
    },
]

input_messages = [
    {"role": "developer", "content": "Be conscise, do not follow up with any extras prompts."},
    {
        "role": "user",
        "content": "Can you tell me the postcode for The Key Employees and when they last completed onboarding in Keystone?",
    },
]

response = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools,
)

print(response.output)

tool_call = response.output[0]
args = json.loads(tool_call.arguments)
print(f"Tool call: {tool_call.name} with arguments: {args}")

result = get_keystone_organisation(args["name"])


input_messages.append(tool_call)  # append model's function call message
input_messages.append(  # append result message
    {
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": str(result),
    }
)

response_2 = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools,
)
print(response_2.output_text)
