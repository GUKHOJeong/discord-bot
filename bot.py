import discord
from discord.ext import commands
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import fitz  # PyMuPDF
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from googleapiclient.discovery import build
import re
import asyncio

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
gemini_Token = os.getenv("gemini_token")
TARGET_CHANNEL_ID = 1380857846706471004
# ✅ 권장 모델 (성능 안정)
HF_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

# 🔗 LangChain LLM 초기화
llm = ChatOpenAI(
    temperature=0.3,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4.1-nano",  # 모델명
)
llm2 = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash", google_api_key=gemini_Token, temperature=0.7
)
# llm3 = ChatGoogleGenerativeAI(
#     model="models/gemini-1.5-pro", google_api_key=gemini_Token, temperature=0.7
# )
llm3 = ChatOpenAI(
    temperature=0.3,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4.1-mini",  # 모델명
)
template1 = [
    (
        "system",
        "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 국멘입니다."
        "어떤 언어로 질문하더라도 항상 한국어로 정중하게 답변하세요.",
    ),
    ("human", "다음 질문에 답변해 주세요:\n\n{query}"),
]


template2 = [
    (
        "system",
        "당신은 친절한 PDF 요약 AI, '국멘_pdf'입니다. "
        "항상 한국어로 정중하게 요약하며, 영어 입력 시 '입력하신 영어 내용을 한국어로 요약해 드립니다.'라는 문장을 먼저 붙이세요. "
        "긴 문서는 섹션별 요약 및 인사이트 중심으로 구조화해 주세요.",
    ),
    ("human", "다음 내용을 한국어로 정리해서 요약해줘:\n\n{input_text}"),
    (
        "ai",
        """안녕하세요, 국멘_pdf입니다. 요청하신 PDF 내용을 다음과 같이 요약해 드립니다:

**[문서 개요]**
- 제목: (있다면 표시)
- 저자: (있다면 표시)
- 주요 내용: (한 줄 핵심 요약)

**[주요 섹션 요약]**
1. **서론**: …
2. **본론**: …
3. **결론**: …

**[핵심 주장 및 인사이트]**
- …""",
    ),
]


template3 = [
    (
        "system",
        "당신은 세계 최초의 완벽한 통역 AI입니다. 당신의 이름은 무료 통역가입니다.,어떤 언어를 물어보면 한국어로 통역을 해세요",
    ),
    ("human", "다음 내용을 한국어로 통역해줘:\n\n{language}"),
]

template4 = [
    (
        "system",
        "당신은 프롬프트 생성 전문 AI입니다. 당신의 이름은 '프롬프트_국'입니다. "
        "사용자가 특정 주제를 제공하면 해당 주제에 맞는 프롬프트를 생성해 주세요. "
        "각 프롬프트는 영문과 한글과 일본어로 각각 제공되어야 합니다.",
    ),
    ("human", "다음 주제에 맞는 프롬프트를 생성해 주세요:\n\n{prompt}"),
    (
        "ai",
        "**[English Prompt]**\n질문에 대한 프롬프트를 영어로 생성해 주세요.\n\n"
        "**[한국어 프롬프트]**\n질문에 대한 프롬프트를 한국어로 생성해 주세요.\n\n"
        "**[일본어 프롬프트]**\n질문에 대한 프롬프트를 일본어로 생성해 주세요.\n\n",
    ),
]

template5 = [
    (
        "system",
        "당신은 질문을 분석하여 그에 맞는 가수 노래를 5개 추천하는 친구입니다.\n"
        "노래를 추천할 때는 다음 형식을 반드시 따르세요:\n"
        "'가수 - 노래명'\n"
        "예시: '김나영 - 솔직하게 말해서 나'\n"
        "무조건 작은 따옴표('') 안에 가수와 제목을 넣어야 하며, 그 외의 설명은 자유롭게 작성해 주세요.",
    ),
    ("human", "{sings}"),
]
# llm = HuggingFaceEndpoint(
#     repo_id=HF_MODEL_ID,
#     huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
#     temperature=0.7,
#     max_new_tokens=1024,
# )


def search_youtube(query, max_results=1):
    YOUTUBE_API_KEY = os.getenv("youtube_token")
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(
        part="snippet", q=query, type="video", maxResults=max_results
    )
    response = request.execute()
    results = []
    for item in response["items"]:
        video_id = item["id"]["videoId"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        results.append(url)

    return results


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            # 1. 텍스트 추출
            page_text = page.get_text().strip()
            text += page_text
    return text


# Discord 설정
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="/", intents=intents)


@bot.event
async def on_ready():
    print(f"✅ 봇 접속: {bot.user.name}")


@bot.command(name="질문")
async def question(ctx, *, query: str):
    if ctx.channel.id != TARGET_CHANNEL_ID:
        await ctx.send(f"❌ 이 명령어는 지정된 채널에서만 사용할 수 있습니다.")
        return
    prompt = ChatPromptTemplate.from_messages(template1)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"query": query})
    MAX_LENGTH = 1500
    for i in range(0, len(result), MAX_LENGTH):
        chunk = result[i : i + MAX_LENGTH]
        await ctx.channel.send(f"{ctx.author.mention}{chunk}")


@bot.command(name="번역")
async def transfer(ctx, *, language: str):
    if ctx.channel.id != TARGET_CHANNEL_ID:
        await ctx.send(f"❌ 이 명령어는 지정된 채널에서만 사용할 수 있습니다.")
        return
    prompt = ChatPromptTemplate.from_messages(template3)
    chain = prompt | llm2 | StrOutputParser()
    result = chain.invoke({"language": language})
    MAX_LENGTH = 1500
    for i in range(0, len(result), MAX_LENGTH):
        chunk = result[i : i + MAX_LENGTH]
        await ctx.channel.send(f"{ctx.author.mention}{chunk}")


@bot.command(name="노래추천")
async def sing(ctx, *, sings: str):
    prompt = ChatPromptTemplate.from_messages(template5)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"sings": sings})
    matches = re.findall(r"'(.+?)'", result)
    link_list = []
    for raw in matches:
        links = search_youtube(raw)
        links = links[0]
        link_list.append(links)
    print(link_list)
    MAX_LENGTH = 1500
    for i in range(0, len(result), MAX_LENGTH):
        chunk = result[i : i + MAX_LENGTH]
        await ctx.channel.send(f"{ctx.author.mention}{chunk}")
    for sing in link_list:
        await ctx.channel.send(f"m!play {sing}")
        await asyncio.sleep(2)


@bot.command(name="생성")
async def prompt(ctx, *, prompts: str):
    if ctx.channel.id != TARGET_CHANNEL_ID:
        await ctx.send(f"❌ 이 명령어는 지정된 채널에서만 사용할 수 있습니다.")
        return
    prompt = ChatPromptTemplate.from_messages(template4)
    chain = prompt | llm3 | StrOutputParser()
    result = chain.invoke({"prompt": prompts})
    MAX_LENGTH = 1500
    for i in range(0, len(result), MAX_LENGTH):
        chunk = result[i : i + MAX_LENGTH]
        await ctx.channel.send(f"{ctx.author.mention}{chunk}")


@bot.command(name="요약")
async def summary_content(ctx, *, query: str = None):
    if ctx.channel.id != TARGET_CHANNEL_ID:
        await ctx.send(f"❌ 이 명령어는 지정된 채널에서만 사용할 수 있습니다.")
        return
    if query:
        prompt = ChatPromptTemplate.from_messages(template2)
        chain = prompt | llm2 | StrOutputParser()
        response = chain.invoke({"input_text": query})
        await ctx.channel.send(f"{ctx.author.mention} 📄 요약 결과:\n{response}")
        return

        # 첨부된 PDF 요약 처리
    elif ctx.message.attachments:
        for file in ctx.message.attachments:
            if file.filename.endswith(".pdf"):
                file_path = f"./{file.filename}"
                await file.save(file_path)

                content = extract_text_from_pdf(file_path)

                prompt = ChatPromptTemplate.from_messages(template2)
                chain = prompt | llm2 | StrOutputParser()
                result = chain.invoke({"input_text": content})
                MAX_LENGTH = 1500
                for i in range(0, len(result), MAX_LENGTH):
                    chunk = result[i : i + MAX_LENGTH]

                    await ctx.channel.send(f"{ctx.author.mention}{chunk}")
                os.remove(file_path)


bot.run(DISCORD_TOKEN)
