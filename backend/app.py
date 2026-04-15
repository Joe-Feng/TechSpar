"""FastAPI app factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import (
    auth,
    copilot,
    history,
    interview,
    knowledge,
    profile,
    recording,
    resume,
    settings,
    topics,
    voiceprint,
)
from backend.startup import preload_models


def create_app() -> FastAPI:
    app = FastAPI(title="TechSpar", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_event_handler("startup", preload_models)

    app.include_router(auth.router)
    app.include_router(resume.router)
    app.include_router(recording.router)
    app.include_router(topics.router)
    app.include_router(profile.router)
    app.include_router(settings.router)
    app.include_router(voiceprint.router)
    app.include_router(interview.router)
    app.include_router(knowledge.router)
    app.include_router(history.router)
    app.include_router(copilot.rest_router)
    app.include_router(copilot.ws_router)
    return app
