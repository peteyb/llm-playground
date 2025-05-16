import asyncio

from agents import Agent, Runner

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
)


async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print("spanish? : ", result.final_output)

    result = await Runner.run(
        triage_agent,
        input="I am in need of some information about my Spanish friend; his name is Juan. How can I see hello to him in Spanish?",
    )
    print("english? :", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

# ¡Hola! Estoy bien, gracias por preguntar. ¿Y tú, cómo estás?
