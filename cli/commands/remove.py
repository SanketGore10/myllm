"""
Remove command - delete models from local storage.

Provides safe model deletion with confirmation prompts.
"""

import typer
import shutil
from pathlib import Path
from rich.console import Console
from rich.prompt import Confirm

from app.core.config import get_settings
from app.models.registry import get_registry


console = Console()


def remove_command(
    model: str = typer.Argument(..., help="Model name to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
):
    """
    Remove a model from local storage.
    
    This will delete the model files and remove it from the registry.
    Use --force to skip the confirmation prompt.
    """
    settings = get_settings()
    registry = get_registry()
    
    # Check if model exists
    try:
        model_config = registry.get_model_config(model)
    except ValueError:
        console.print(f"[red]✗[/red] Model '{model}' not found")
        console.print("\n[dim]Available models:[/dim]")
        
        # Show available models
        models = registry.list_models()
        if models:
            for m in models:
                console.print(f"  - {m['name']}")
        else:
            console.print("  [dim]No models installed[/dim]")
        
        raise typer.Exit(1)
    
    # Get model directory
    model_dir = settings.models_dir / model
    
    if not model_dir.exists():
        console.print(f"[yellow]⚠[/yellow] Model directory not found: {model_dir}")
        console.print("[yellow]Model is in registry but files are missing[/yellow]")
        raise typer.Exit(1)
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    # Show model info
    console.print(f"\n[cyan]Model:[/cyan] {model}")
    console.print(f"[cyan]Family:[/cyan] {model_config.family}")
    console.print(f"[cyan]Size:[/cyan] {size_mb:.1f} MB")
    console.print(f"[cyan]Location:[/cyan] {model_dir}")
    
    # Confirm deletion (unless --force)
    if not force:
        console.print()
        confirmed = Confirm.ask(
            f"[yellow]Are you sure you want to remove '{model}'?[/yellow]",
            default=False
        )
        
        if not confirmed:
            console.print("\n[dim]Removal cancelled[/dim]")
            raise typer.Exit(0)
    
    # Remove model directory
    try:
        console.print(f"\n[cyan]Removing model files...[/cyan]")
        shutil.rmtree(model_dir)
        console.print(f"[green]✓[/green] Removed {model_dir}")
        
        # Rescan registry to update
        console.print(f"[cyan]Updating registry...[/cyan]")
        registry.scan_models()
        
        console.print(f"\n[green]✓[/green] Model '{model}' removed successfully!")
        console.print(f"[dim]Freed {size_mb:.1f} MB of disk space[/dim]")
    
    except Exception as e:
        console.print(f"\n[red]✗[/red] Failed to remove model: {e}")
        raise typer.Exit(1)
