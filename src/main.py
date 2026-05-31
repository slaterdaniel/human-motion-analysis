from fastapi import FastAPI, UploadFile, Form, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.using_tool import analyze
import traceback
import shutil
import queue
import asyncio

app = FastAPI()
app.mount('/outputs', StaticFiles(directory="outputs"))

frame_queue = queue.Queue(maxsize=2)
video_metadata = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.post() --> update data
# @app.put() --> create data
# @app.delete() --> delete data
# @app.get() --> receive data

@app.post('/inputs')
async def processInputs(
        video_file: UploadFile = File(...),
        model: str = Form(...),
        show: bool = Form(...)
):
    print('\n🚀 TRYING TO RUN PIPELINE...\n')
    print(f"Received file: {video_file.filename}")
    print(f"Received model: {model}")
    print(f"Received show flag: {show}")

    user_video = f'data/user_input/{video_file.filename}'
    with open(user_video, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)

    try:
        outputs = await asyncio.to_thread(
            analyze,
            user_video=user_video,
            engine=model,
            show=show,
        )

        outputs['status'] = 'Success'
        print('OUTPUTS:\n\n', outputs)
        return outputs

    except Exception as e:

        print("!!! PYTHON PIPELINE CRASHED !!!")
        traceback.print_exc()

        return {
            "status": "Error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }

@app.websocket('/ws')
async def image_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while "frame_count" not in video_metadata:
            await asyncio.sleep(0.1)

        await websocket.send_json({
            "type": "init",
            "frame_count": video_metadata["frame_count"]
        })

        while True:
            frame = await asyncio.to_thread(frame_queue.get)
            if frame is None:
                print('\n\n!!! FRAME NOT FOUND: BREAKING !!!\n\n')
                break
            await websocket.send_bytes(frame)

    except Exception as e:
        print(f'WebSocket error: {e}')

    finally:
        await websocket.close()




