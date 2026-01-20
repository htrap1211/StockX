import uvicorn
import os

if __name__ == "__main__":
    # Ensure env vars are loaded or passed
    # In prod, use uvicorn main:app
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
