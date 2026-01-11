"""
CLI entry point using Typer.

Provides command-line interface for MyLLM operations.
"""

import typer
from typing import Optional
from pathlib import Path

from app.core.config import get_settings
from app.utils.logging import setup_logging

# Create Typer app
app = typer.Typer(
    name="myllm",
    help="MyLLM - Production-grade local LLM runtime",
    add_completion=False,
)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    models_dir: Optional[Path] = typer.Option(None, "--models-dir", help="Override models directory"),
):
    """
    MyLLM CLI - Run large language models locally.
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    
    # Override models_dir if specified
    if models_dir:
        import os
        os.environ["MODELS_DIR"] = str(models_dir)


# Import and register commands
from cli.commands import serve, run, pull

# Register command routers
app.command(name="serve")(serve.serve_command)
app.command(name="run")(run.run_command)
app.command(name="pull")(pull.pull_command)


# List command
@app.command()
def list():
    """List all available models."""
    from app.models.registry import get_registry
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    registry = get_registry()
    models = registry.list_models()
    
    if not models:
        console.print("[yellow]No models found in models directory[/yellow]")
        console.print(f"Models directory: {get_settings().models_dir}")
        return
    
    # Create table
    table = Table(title="Available Models")
    table.add_column("Name", style="cyan")
    table.add_column("Family", style="magenta")
    table.add_column("Quantization", style="green")
    table.add_column("Context Size", style="blue")
    table.add_column("Size", style="yellow")
    
    for model in models:
        size_str = f"{model.size_mb} MB" if model.size_mb else "unknown"
        table.add_row(
            model.name,
            model.family,
            model.quantization,
            str(model.context_size),
            size_str,
        )
    
    console.print(table)
    console.print(f"\nTotal: {len(models)} models")


# Show command
@app.command()
def show(model_name: str = typer.Argument(..., help="Model name to show")):
    """Show detailed information about a model."""
    from app.models.registry import get_registry
    from app.utils.errors import ModelNotFoundError
    from rich.console import Console
    from rich.panel import Panel
    from rich.json import JSON
    import json
    
    console = Console()
    
    try:
        registry = get_registry()
        model_info = registry.get_model(model_name)
        model_config = registry.get_model_config(model_name)
        model_path = registry.get_model_path(model_name)
        
        # Format information
        info_text = f"""[cyan]Name:[/cyan] {model_info.name}
[cyan]Family:[/cyan] {model_info.family}
[cyan]Quantization:[/cyan] {model_info.quantization}
[cyan]Context Size:[/cyan] {model_info.context_size}
[cyan]Template:[/cyan] {model_info.template}
[cyan]Size:[/cyan] {model_info.size_mb} MB
[cyan]Path:[/cyan] {model_path}
[cyan]Loaded:[/cyan] {'Yes' if model_info.loaded else 'No'}

[cyan]Parameters:[/cyan]
{json.dumps(model_info.parameters, indent=2)}"""
        
        console.print(Panel(info_text, title=f"Model: {model_name}", border_style="blue"))
    
    except ModelNotFoundError:
        console.print(f"[red]Error: Model '{model_name}' not found[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
