import discord
from discord.ext import commands
import yt_dlp
import asyncio
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
import re
from googleapiclient.discovery import build
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from yt_dlp import YoutubeDL

load_dotenv()
sing_token = os.getenv("sing_token")

openai_api_key = os.getenv("OPENAI_API_KEY")

gemini_Token = os.getenv("gemini_token")
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="/", intents=intents)


template5 = [
    (
        "system",
        "당신은 질문을 분석하여 그에 맞는 가수 노래를 n개 추천하는 친구입니다.\n"
        "노래를 추천할 때는 다음 형식을 반드시 따르세요:\n"
        "'가수 - 노래명'\n"
        "예시: '김나영 - 솔직하게 말해서 나'\n"
        "무조건 작은 따옴표('') 안에 가수와 제목을 넣어야 하며, 그 외의 설명은 자유롭게 작성해 주세요.",
    ),
    ("human", "{sings}"),
]
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash", google_api_key=gemini_Token, temperature=0.2
)


def search_youtube(query):
    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "default_search": "ytsearch",  # 🔥 중요
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=False)
        if "entries" in info:
            info = info["entries"][0]  # 검색 결과 중 첫 번째 영상

        return {
            "title": info["title"],
            "source": info["url"],  # ✅ ffmpeg가 사용할 실제 오디오 스트림 URL
        }


music_queue = []
is_playing = False
voice_client = None


async def play_next(ctx):
    global is_playing, voice_client

    if len(music_queue) > 0:
        is_playing = True
        song = music_queue.pop(0)

        # ffmpeg_opts = {
        #     "before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
        #     "options": "-vn -af volume={volume_level}",
        # }
        source = discord.FFmpegPCMAudio(
            source=song["source"],
            executable="C:/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe",
            before_options="-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -protocol_whitelist file,http,https,tcp,tls,crypto",
            options="-vn",
        )

        volume_applied = discord.PCMVolumeTransformer(source, volume=volume_level)

        voice_client.play(
            volume_applied,
            after=lambda e: asyncio.run_coroutine_threadsafe(play_next(ctx), bot.loop),
        )
        await ctx.send(f"🎶 다음곡은 : **{song['title']}**")
    else:
        is_playing = False


volume_level = 0.5


@bot.command(name="기본볼륨")
async def set_volume(ctx, level: int):
    global volume_level
    if level < 1 or level > 200:
        await ctx.send("⚠️ 볼륨은 1~200 사이 숫자로 설정해 주세요.")
        return
    volume_level = level / 100  # 예: 10 → 0.01, 100 → 0.1
    await ctx.send(
        f"🔊 볼륨이 {level}%로 설정되었습니다. (실제 FFmpeg: {volume_level})"
    )


async def add_song(ctx, query):
    global voice_client, is_playing
    song = search_youtube(query)
    music_queue.append(song)
    print("🎧 추출된 URL:", song["source"])
    if ctx.author.voice is None:
        await ctx.send("🎧 음성 채널에 먼저 들어가주세요.")
        return

    if voice_client is None or not voice_client.is_connected():
        voice_client = await ctx.author.voice.channel.connect(timeout=50)

    if not is_playing:
        await play_next(ctx)
    else:
        await ctx.send(f"✅ **{song['title']}** 추가되었습니다.")


@bot.command(name="추가")
async def play(ctx, *, query):
    await add_song(ctx, query)


@bot.command(name="스킵")
async def skip(ctx):
    global voice_client
    if voice_client and voice_client.is_playing():
        voice_client.stop()
        await ctx.send("⏭️ 스킵했습니다!")


@bot.command(name="스탑")
async def stop(ctx):
    global voice_client, music_queue, is_playing
    music_queue.clear()
    is_playing = False
    if voice_client:
        await voice_client.disconnect()
        voice_client = None
    await ctx.send("🛑 음악 중지 및 연결 해제됨.")


@bot.command(name="노래추천")
async def sing(ctx, *, sings: str):
    prompt = ChatPromptTemplate.from_messages(template5)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"sings": sings})
    matches = re.findall(r"'(.+?)'", result)
    await ctx.channel.send(f"{ctx.author.mention}{result}")
    for raw in matches:
        await add_song(ctx, raw)


@bot.command(name="대기열")
async def queq(ctx):
    if len(music_queue) >= 1:
        embed = discord.Embed(title="🎶 음악 대기열", color=0x1DB954)
        for idx, track in enumerate(music_queue, start=1):
            embed.add_field(
                name=f"{idx}. {track['title']}", value="\u200b", inline=False
            )
            if idx == 24:
                embed.set_footer(text=f"…그리고 +{len(music_queue)-25}곡 더 있습니다.")
                break
        await ctx.send(embed=embed)
    else:
        await ctx.send("📭 현재 대기 중인 곡이 없습니다.")


@bot.command(name="삭제")
async def drop(ctx, idx: int):
    idx = idx - 1
    if len(music_queue) == 0:
        await ctx.send("📭 현재 대기 중인 곡이 없습니다.")
    else:
        title = music_queue.pop(idx)["title"]
        await ctx.send(f"{title}이 삭제되었습니다.")


bot.run(sing_token)
