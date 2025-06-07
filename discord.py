import discord
from langchain.schema import HumanMessage
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os


def load_local_llm():
    pipe = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
    )
    return HuggingFacePipeline(pipeline=pipe)


llm = load_local_llm()

# Discord 설정
intents = discord.Intents.default()
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"✅ 봇 접속: {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!질문 "):
        query = message.content[4:]
        response = llm.invoke(query)
        await message.channel.send(response[:2000])


load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
client.run(TOKEN)
