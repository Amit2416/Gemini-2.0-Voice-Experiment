import asyncio
import base64
import json
import os
import pyaudio
import numpy as np
from websockets.asyncio.client import connect
from dotenv import load_dotenv

load_dotenv()


class SimpleGeminiVoice:
    def __init__(self):
        self.audio_queue = asyncio.Queue()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model = "gemini-2.0-flash-exp"
        self.uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
        # Audio settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.CHUNK = 512
        self.RATE = 16000
        self.SILENCE_THRESHOLD = 1000
        self.is_speaking = False
        self.silence_counter = 0
        self.MAX_SILENCE_COUNT = 50
        self.is_model_responding = False
        self.waiting_for_response = False

    async def start(self):
        self.ws = await connect(
            self.uri, additional_headers={"Content-Type": "application/json"}
        )
        await self.ws.send(json.dumps({"setup": {"model": f"models/{self.model}"}}))
        await self.ws.recv(decode=False)
        print("Connected to Gemini, You can start talking now")
        
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.capture_audio())
            tg.create_task(self.stream_audio())
            tg.create_task(self.play_response())

    def is_silent(self, audio_data):
        data_np = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(np.square(data_np)))
        return rms < self.SILENCE_THRESHOLD

    async def capture_audio(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )
        
        print("Started listening...")
        while True:
            try:
                # Don't capture audio while model is responding
                if self.is_model_responding:
                    await asyncio.sleep(0.1)
                    continue

                data = await asyncio.to_thread(stream.read, self.CHUNK)
                
                if self.is_silent(data):
                    self.silence_counter += 1
                else:
                    self.silence_counter = 0
                    self.is_speaking = True
                    self.waiting_for_response = False

                if self.is_speaking and self.silence_counter < self.MAX_SILENCE_COUNT:
                    await self.ws.send(
                        json.dumps(
                            {
                                "realtime_input": {
                                    "media_chunks": [
                                        {
                                            "data": base64.b64encode(data).decode(),
                                            "mime_type": "audio/pcm",
                                        }
                                    ]
                                }
                            }
                        )
                    )
                elif self.is_speaking and self.silence_counter >= self.MAX_SILENCE_COUNT:
                    if not self.waiting_for_response:
                        print("\nSilence detected, waiting for response...")
                        self.waiting_for_response = True
                    self.is_speaking = False
                    
            except Exception as e:
                print(f"Error capturing audio: {e}")
                await asyncio.sleep(0.1)

    async def stream_audio(self):
        response_started = False
        async for msg in self.ws:
            response = json.loads(msg)
            
            try:
                # Check if we're getting audio data
                if "serverContent" in response and "modelTurn" in response["serverContent"]:
                    if not response_started:
                        print("\nModel is responding...")
                        response_started = True
                        self.is_model_responding = True
                    
                    audio_data = response["serverContent"]["modelTurn"]["parts"][0]["inlineData"]["data"]
                    await self.audio_queue.put(base64.b64decode(audio_data))
                
                # Check for turn completion
                if response.get("serverContent", {}).get("turnComplete"):
                    if response_started:
                        print("Response complete, you can speak now...")
                        response_started = False
                        self.is_model_responding = False
                        await asyncio.sleep(1.0)  # Longer pause between turns
            except Exception as e:
                print(f"Error processing response: {e}")

    async def play_response(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.FORMAT, channels=self.CHANNELS, rate=24000, output=True
        )
        while True:
            try:
                data = await self.audio_queue.get()
                await asyncio.to_thread(stream.write, data)
            except Exception as e:
                print(f"Error playing audio: {e}")
                await asyncio.sleep(0.1)


if __name__ == "__main__":
    client = SimpleGeminiVoice()
    asyncio.run(client.start())
