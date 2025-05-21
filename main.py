# main.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import routes

app = FastAPI(
    title="Data Analysis API",
    description="Upload Excel/CSV files and ask natural language questions to get insights, visualizations, and explanations.",
    version="0.1.0"
)

# CORS (Cross-Origin Resource Sharing) - Allow all for development
# For production, restrict origins to your frontend's domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(routes.router, prefix="/api")

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    # It's better to run uvicorn from the command line:
    # uvicorn main:app --reload
    # But this is here for convenience if you run `python main.py`
    uvicorn.run(app, host="0.0.0.0", port=8000)