import asyncio
import sys
import numpy as np
import sounddevice as sd
from audio_util import AudioPlayerAsync, CHANNELS, SAMPLE_RATE
from getchar import getkeys
from agents import Agent
from agents.voice import StreamedAudioInput
from agents.voice import SingleAgentVoiceWorkflow
from agents.voice import VoicePipeline
from agents.mcp import MCPServerStdio

google_search_server = MCPServerStdio(
    name="GoogleSearch",
    params={
        "command": "uv",
        "args": [
            "--directory",
            "C:/temp/mcp_test",
            "run",
            "server_google_search.py"
        ]
    }
) 

# 隱藏游標
def hide_cursor():
    print("\r\033[?25l", end="")

# 顯示游標
def show_cursor():
    print("\r\033[?25h", end="")

agent = Agent(
    name="Assistant",
    instructions="使用台灣繁體中文",
    model="gpt-4.1-mini",
)

player = AudioPlayerAsync()

class MyVoiceWorkflow(SingleAgentVoiceWorkflow):
    async def run(self, transcription: str):
        print(f"\r>>> {transcription}")
        player.stop()  # 停止播放之前的音訊
        async for chunk in super().run(transcription):
            # 在這裡可以處理每個文字片段
            print(chunk, end="")
            yield chunk
        print('')

should_send_audio = asyncio.Event()
audio_input = StreamedAudioInput()
        
async def start_voice_pipeline() -> None:
    pipeline = VoicePipeline(
        workflow=MyVoiceWorkflow(agent)
    )
    await google_search_server.connect()
    agent.mcp_servers = [google_search_server]
    try:
        result = await pipeline.run(audio_input)

        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                player.add_data(event.data)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        player.stop()
        await google_search_server.cleanup()

async def send_mic_audio() -> None:
    read_size = int(SAMPLE_RATE * 0.02)

    stream = sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype=np.int16,
    )
    stream.start()

    try:
        while True:
            # 先累積基本的音訊資料
            if stream.read_available < read_size:
                await asyncio.sleep(0)
                continue

            # 等待按下 r 鍵才開始傳送音訊資料
            await should_send_audio.wait()

            data, _ = stream.read(read_size)

            # 傳送音訊資料給伺服端，伺服端會自動判斷段落就回應
            await audio_input.add_audio(data)
            await asyncio.sleep(0)
    except KeyboardInterrupt:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop()
        stream.close()

async def main() -> None:
    mic_task = asyncio.create_task(send_mic_audio())
    realtime_task = asyncio.create_task(start_voice_pipeline())

    is_recording = False
    print("\r⏹", end="")
    hide_cursor()
    while True:
        keys = getkeys()
        if len(keys) == 0:            
            await asyncio.sleep(0.1)
            continue
        key = keys.pop().lower()
        if key == "r":
            is_recording = not is_recording
            if is_recording:
                print("\r⏺", end="")
                should_send_audio.set()
            else:
                should_send_audio.clear()
                print("\r⏹", end="")
        elif key == "q":
            break

    show_cursor()
    mic_task.cancel()
    realtime_task.cancel()
    await asyncio.gather(mic_task, realtime_task)

if __name__ == "__main__":
    asyncio.run(main())