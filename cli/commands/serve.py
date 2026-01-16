"""
Serve command - start API server.
"""

import typer
import uvicorn

from app.core.config import get_settings


def serve_command(
    host: str = typer.Option(None, "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(None, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (development)"),
):
    """
    Start the MyLLM API server.
    
    Runs the FastAPI application with uvicorn.
    """
    settings = get_settings()
    
    # Use settings defaults if not specified
    host = host or settings.host
    port = port or settings.port
    
    typer.echo(f"Starting MyLLM server on {host}:{port}")
    console.print(f"[cyan]API docs available at http://{host}:{port}/docs[/cyan]\n")
    
    # Use MYLLM_LOG_LEVEL env var for uvicorn log level (default: warning for clean UX)
    import os
    log_level = os.getenv("MYLLM_LOG_LEVEL", "warning").lower()
    
    # Run uvicorn server
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
