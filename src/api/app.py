from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.endpoints import router as main_router
from src.api.intraday_endpoints import router as intraday_router

app = FastAPI(title="AI Stock Recommender API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(main_router, prefix="/api/v1", tags=["Swing Trading"])
app.include_router(intraday_router, prefix="/api/v1", tags=["Intraday Trading"])

@app.get("/")
def root():
    return {"message": "AI Stock Recommender API", "version": "1.0.0"}
