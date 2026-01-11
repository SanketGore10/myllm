"""
FastAPI application factory.

Creates and configures the FastAPI application with all routers,
middleware, and lifecycle hooks.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import api_router
from app.core.config import get_settings
from app.storage.database import init_database, close_database
from app.models.registry import get_registry
from app.services.model_loader import shutdown_model_loader
from app.utils.logging import setup_logging, get_logger
from app.utils.errors import MyLLMError

# Setup logging first
settings = get_settings()
setup_logging(level=settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown tasks.
    """
    # Startup
    logger.info("Starting MyLLM server...")
    
    # Initialize database
    await init_database()
    logger.info("Database initialized")
    
    # Scan for available models
    registry = get_registry()
    models = registry.scan_models()
    logger.info(f"Found {len(models)} models")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MyLLM server...")
    
    # Unload all models
    shutdown_model_loader()
    logger.info("Models unloaded")
    
    # Close database
    await close_database()
    logger.info("Database closed")
    
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI instance
    """
    app = FastAPI(
        title="MyLLM",
        description="Production-grade local LLM runtime system",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routers
    app.include_router(api_router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with server information."""
        registry = get_registry()
        models = registry.list_models()
        
        return {
            "name": "MyLLM",
            "version": "0.1.0",
            "status": "running",
            "models_available": len(models),
        }
    
    # Health check endpoint
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    # Global exception handler for MyLLM errors
    @app.exception_handler(MyLLMError)
    async def myllm_error_handler(request, exc: MyLLMError):
        """Handle custom MyLLM errors."""
        logger.error(f"MyLLM error: {exc.message}")
        return JSONResponse(
            status_code=400,
            content={"error": exc.message, "details": exc.details},
        )
    
    logger.info("FastAPI application created")
    
    return app


# Create app instance
app = create_app()
