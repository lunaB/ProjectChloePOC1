from fastapi import FastAPI
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    print("[ENV_NAME]", os.getenv("ENV_NAME"))

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)