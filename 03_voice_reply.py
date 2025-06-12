import asyncio
import time
import numpy as np
import sounddevice as sd
from getchar import getkeys
from audio_util import AudioPlayerAsync
from audio_util import CHANNELS, SAMPLE_RATE
from agents import Agent
from agents.voice import AudioInput
from agents.voice import SingleAgentVoiceWorkflow
from agents.voice import VoicePipeline

# 隱藏游標
def hide_cursor():
    print("\r\033[?25l", end="")

# 顯示游標
def show_cursor():
    print("\r\033[?25h", end="")

agent = Agent(
    name="小助理",
    instructions="使用台灣繁體中文",
    model="gpt-4.1-mini",
)

class MyVoiceWorkflow(SingleAgentVoiceWorkflow):
    async def run(self, transcription: str):
        print(f"\r>>> {transcription}")
        async for chunk in super().run(transcription):
            # 在這裡可以處理每個文字片段
            print(chunk, end="")
            yield chunk
        print('')

pipeline = VoicePipeline( # 負責語音文字間轉換
    workflow=MyVoiceWorkflow(agent) # 負責執行 agent
)

audio_player = AudioPlayerAsync()

recording  = False  # 是否錄音中
audio_buffer = []   # 音訊暫存區

# 接收新音訊
def _audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Status: {status}\n")
    if recording:
        audio_buffer.append(indata.copy())

# 建立麥克風輸入串流
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype=np.int16,
    callback=_audio_callback
)

# 取得單次提問的語音資料
def get_audio_data():
    global recording, audio_buffer
    while True:
        keys = getkeys()
        if len(keys) < 1:  # 沒有按鍵被按下
            time.sleep(0.01)
            continue
        key = keys[0].lower()
        if key == "q":
            return None
        if key != "r":
            continue
        recording = not recording
        if recording:
            print("\r⏺", end="")
            continue
        print("\r⏹", end="")
        # 合併音訊片段
        if audio_buffer:
            audio_data = np.concatenate(audio_buffer, axis=0)
        else:
            audio_data = np.empty((0,), dtype=np.float32)
        return audio_data
    
async def main():
    hide_cursor()
    print('按下 "r" 開始/結束錄音，按下 "q" 結束程式')
    print("\r⏹", end="")

    stream.start()

    while True:
        audio_data = get_audio_data()
        if audio_data is None:
            break
        audio_input = AudioInput(buffer=audio_data)
        result = await pipeline.run(audio_input)
        # 播放串流音訊
        status_ch = '▶'
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                audio_player.add_data(event.data)
                print(f'\r{status_ch}', end="")
                status_ch = '▶' if status_ch == ' ' else ' '
        print("\r⏹", end="")
        audio_buffer.clear()
            
    show_cursor()
    print("\r結束")
    audio_player.stop()
    stream.stop()
    stream.close()

if __name__ == "__main__":
    asyncio.run(main())
