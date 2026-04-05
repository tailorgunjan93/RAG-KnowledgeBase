from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from Src.Router.UploadRouter import upload_router
from Src.Router.EmbeddingRouter import embedding_router
from Src.Router.SimilaritySearchRouter import similarity_search_router
from Src.Router.GraderRouter import GradeRouter
from Src.Router.chatbotRouter import chatbot_router


app = FastAPI()

app.include_router(upload_router)
app.include_router(embedding_router)
app.include_router(similarity_search_router)
app.include_router(GradeRouter)
app.include_router(chatbot_router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
