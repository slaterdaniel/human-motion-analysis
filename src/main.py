from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from src.using_tool import analyze
from src import shared
import traceback
import shutil
import time
import queue

app = FastAPI()

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
        outputs = analyze(
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
    
def get_processing_preview(filename: str):
    start_time = time.time()
    while filename not in shared.shared_queues:
        time.sleep(0.1)
        if time.time() - start_time > 10:
            print(f"\n\n!!! Timeout: No processing preview available for {filename} after 10 seconds !!!\n\n")
            return
    
    q = shared.shared_queues.get(filename)

    while True:
        try:
            frame = q.get(timeout=2)
            if not frame:
                break
            print(f'frame sending')
            yield frame

        except queue.Empty:
            print('No frame received in the last 2 seconds, ending stream.')
            break

    shared.shared_queues.pop(filename, None)

@app.get('/processing_preview')
async def processing_preview(filename: str):
    return StreamingResponse(get_processing_preview(filename), media_type='multipart/x-mixed-replace; boundary=frame')




