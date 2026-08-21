from fastapi import FastAPI
from src.api.v1.routes import query
from src.api.v1.routes.upload_routes import router as upload_router
from src.api.v1.agents.agents import build_rag_graph

from contextlib import asynccontextmanager

from src.core.checkpoint import init_checkpoint, close_checkpoint

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("========== Starting application ==========")

    # Initialize async postgres checkpoint
    checkpoint = await init_checkpoint()

    # Build graph with async checkpoint
    app.state.rag_graph = build_rag_graph(
        checkpoint
    )

    print("========== RAG graph initialized ==========")


    yield


    print("========== Shutting down ==========")

    await close_checkpoint()


app = FastAPI(
    lifespan=lifespan
)

#app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


app.include_router(query.router)


# Upload routes
app.include_router(upload_router)


# @app.on_event("startup")
# async def startup():

#     checkpoint = await init_checkpoint()

#     app.state.rag_graph = build_rag_graph(checkpoint)


# @app.on_event("shutdown")
# async def shutdown():

#     await close_checkpoint()
