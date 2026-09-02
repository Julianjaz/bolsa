from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Swing Trading Decision Support System",
    description="API for quantitative Swing Trading signal evaluation using deterministic indicators and LLMs.",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
