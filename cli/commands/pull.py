"""
Pull command - download models from catalog.

Enhanced implementation with automatic downloads from Hugging Face.
"""

import os
import typer
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn
from rich.table import Table
from rich.panel import Panel

from app.models.catalog import (
    get_model_from_catalog,
    list_catalog_models,
    is_model_in_catalog,
)
from app.utils.download import (
    download_model_from_hf,
    DownloadError,
    check_disk_space,
    format_size,
)
from app.models.config_generator import create_model_config
from app.core.config import get_settings


console = Console()


def show_available_models():
    """Display table of available models in catalog."""
    models = list_catalog_models()
    
    if not models:
        console.print("[yellow]No models available in catalog[/yellow]")
        return
    
    table = Table(title="Available Models in Catalog")
    table.add_column("Name", style="cyan")
    table.add_column("Size", style="yellow")
    table.add_column("Family", style="magenta")
    table.add_column("Context", style="blue")
    table.add_column("Description", style="white")
    
    for model in models:
        table.add_row(
            model["name"],
            format_size(model["size_mb"] * 1024 * 1024),
            model["family"],
            str(model["context_size"]),
            model["description"][:60] + "..." if len(model["description"]) > 60 else model["description"],
        )
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(models)} models available[/dim]")
    console.print("\n[dim]Download with: myllm pull <model-name>[/dim]")


def pull_command(
    model: str = typer.Argument(..., help="Model identifier to download"),
    list_models: bool = typer.Option(False, "--list", "-l", help="List available models"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download even if model exists"),
):
    """
    Download a model from Hugging Face.
    
    Downloads GGUF models from the catalog and automatically configures them for use.
    """
    # Handle --list flag
    if list_models:
        show_available_models()
        return
    
    # Check if model is in catalog
    if not is_model_in_catalog(model):
        console.print(f"[red]✗[/red] Model '{model}' not found in catalog\n")
        console.print("[yellow]Available models:[/yellow]")
        show_available_models()
        raise typer.Exit(1)
    
    # Get model metadata
    model_metadata = get_model_from_catalog(model)
    
    # Setup paths
    settings = get_settings()
    model_dir = settings.models_dir / model
    model_file = model_dir / "model.gguf"
    config_file = model_dir / "config.json"
    
    # Check if model already exists
    if model_file.exists() and config_file.exists() and not force:
        console.print(f"[yellow]Model '{model}' already exists[/yellow]")
        console.print(f"Use --force to re-download")
        raise typer.Exit(0)
    
    # Display download info
    console.print(Panel(
        f"[cyan]Downloading: {model}[/cyan]\n\n"
        f"Size: [yellow]{format_size(model_metadata['size_mb'] * 1024 * 1024)}[/yellow]\n"
        f"Family: [magenta]{model_metadata['family']}[/magenta]\n"
        f"Context: [blue]{model_metadata['context_size']}[/blue]\n\n"
        f"{model_metadata['description']}",
        title="Model Download",
        border_style="blue",
    ))
    
    # Check disk space
    required_bytes = model_metadata['size_mb'] * 1024 * 1024 * 1.1  # Add 10% buffer
    if not check_disk_space(settings.models_dir, required_bytes):
        console.print(f"[red]✗[/red] Insufficient disk space")
        console.print(f"Required: {format_size(required_bytes)}")
        raise typer.Exit(1)
    
    # Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Get HuggingFace token if available
    hf_token = os.environ.get("HF_TOKEN")
    
    try:
        # Download from HuggingFace
        console.print("\n[cyan]Downloading from Hugging Face...[/cyan]")
        
        downloaded_file = download_model_from_hf(
            repo_id=model_metadata["repo_id"],
            filename=model_metadata["filename"],
            local_dir=model_dir,
            token=hf_token,
        )
        
        # Rename to model.gguf if needed
        if downloaded_file.name != "model.gguf":
            final_path = model_dir / "model.gguf"
            if final_path.exists():
                final_path.unlink()
            downloaded_file.rename(final_path)
            console.print(f"[green]✓[/green] Renamed to model.gguf")
        
        # Generate config.json
        console.print("[cyan]Generating configuration...[/cyan]")
        config_path = create_model_config(
            model_name=model,
            model_dir=model_dir,
            catalog_metadata=model_metadata,
        )
        console.print(f"[green]✓[/green] Config saved to {config_path.name}")
        
        # Verify installation
        from app.models.registry import get_registry
        registry = get_registry()
        registry.scan_models()
        
        # Success message
        console.print(f"\n[green]✓[/green] Model '{model}' installed successfully!")
        console.print(f"\n[dim]Try it out:[/dim]")
        console.print(f"  [cyan]myllm run {model}[/cyan]")
        console.print(f"  [cyan]myllm show {model}[/cyan]")
    
    except DownloadError as e:
        console.print(f"\n[red]✗[/red] Download failed: {e}")
        
        # Cleanup on failure
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
            console.print("[dim]Cleaned up partial download[/dim]")
        
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"\n[red]✗[/red] Unexpected error: {e}")
        raise typer.Exit(1)
