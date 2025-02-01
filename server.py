import binascii
import uuid
import os
import asyncio
import numpy as np
from quart import Quart, request, jsonify, Response, render_template
import json
import edge_tts
from aiocache import SimpleMemoryCache
import vosk
import sounddevice as sd

cache = SimpleMemoryCache()

app = Quart(__name__)

# Vosk 语音识别模型初始化
model = vosk.Model("model/vosk-model-small-cn-0.22")
sample_rate = 16000

@app.route('/')
async def home():
    return await render_template('home.html')


@app.route('/generate', methods=['POST'])
async def generate():
    data = await request.get_data()
    payload = binascii.hexlify(data).decode('utf8')
    audio_id = str(uuid.uuid4())
    await cache.set(audio_id, payload, ttl=600)  # 设置缓存有效期 10 分钟
    t = {"code": 1, "audio_url": f"/tts?audio_id={audio_id}"}
    return jsonify(t)


@app.route('/tts', methods=['GET'])
async def tts():
    audio_id = request.args.get("audio_id")
    if not audio_id:
        return jsonify({"code": -1, "message": "Missing audio_id"}), 400
    # 从缓存中获取 payload
    payload = await cache.get(audio_id)
    if not payload:
        return jsonify({"code": -1, "message": "Audio ID expired or invalid"}), 404
    plaintext = binascii.unhexlify(payload).decode('utf8')
    json_obj = json.loads(plaintext)
    content = json_obj["content"]
    voice = json_obj.get("voice", "zh-CN-XiaoxiaoNeural")
    headers = {'Transfer-Encoding': 'chunked', 'Content-Type': 'audio/mpeg'}
    response_content = ms_tts_async(content, voice)
    return Response(response_content, headers=headers)


async def ms_tts_async(text_input, voice="zh-CN-XiaoxiaoNeural"):
    communicate = edge_tts.Communicate(text_input, voice)
    async for chunked in communicate.stream():
        if chunked["type"] == "audio":
            yield chunked["data"]


@app.route('/recognize', methods=['POST'])
async def recognize():
    data = await request.get_data()
    audio = np.frombuffer(data, dtype=np.int16)
    rec = vosk.KaldiRecognizer(model, sample_rate)
    rec.AcceptWaveform(audio)
    result = json.loads(rec.Result())
    return jsonify(result)


if __name__ == '__main__':
    app.run()