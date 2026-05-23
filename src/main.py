from fastapi import FastAPI, Form
from pydantic import BaseModel
from src.using_tool import analyze
import traceback

app = FastAPI()

# @app.post() --> update data
# @app.put() --> create data
# @app.delete() --> delete data
# @app.get() --> receive data

class InputFormat(BaseModel):
    user_video: str
    model: str
    show: bool

@app.post('/inputs')
async def processInputs(user_inputs: InputFormat):
    try:
        # 1. Try running your heavy pipeline
        outputs = analyze(
            user_video=user_inputs.user_video,
            engine=user_inputs.model,
            show=user_inputs.show,
        )

        # If it works perfectly, mark it as a success
        outputs['status'] = 'Success'
        return outputs

    except Exception as e:
        # 2. If ANYTHING crashes inside analyze(), catch it here!
        print("!!! PYTHON PIPELINE CRASHED !!!")
        traceback.print_exc()  # Prints the exact line number of the crash in your terminal

        # 3. Send a clean, successful HTTP response containing the error data
        return {
            "status": "Error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }




