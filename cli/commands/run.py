"""
Run command - interactive chat.
"""

import typer
import asyncio
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from app.models.schemas import Message, InferenceOptions
from app.core.runtime import get_runtime
from app.core.config import get_settings
from app.storage.database import init_database
from app.utils.errors import ModelNotFoundError


async def interactive_chat(model_name: str, temperature: float):
    """Run interactive chat session."""
    console = Console()
    
    # Initialize database
    await init_database()
    
    # Get runtime
    runtime = get_runtime()
    
    # Create new session
    session_id = None
    
    # Welcome message
    console.print(Panel(
        f"[cyan]MyLLM Interactive Chat[/cyan]\n\n"
        f"Model: [yellow]{model_name}[/yellow]\n"
        f"Temperature: [yellow]{temperature}[/yellow]\n\n"
        f"Commands:\n"
        f"  /exit - Exit chat\n"
        f"  /clear - Clear history\n"
        f"  /help - Show help",
        title="Welcome",
        border_style="blue",
    ))
    
    while True:
        try:
            # Get user input
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
            
            # Handle commands
            if user_input.strip() == "/exit":
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            elif user_input.strip() == "/clear":
                session_id = None
                console.print("[yellow]History cleared. Starting new session.[/yellow]")
                continue
            
            elif user_input.strip() == "/help":
                console.print(Panel(
                    "Commands:\n"
                    "  /exit - Exit chat\n"
                    "  /clear - Clear history\n"
                    "  /help - Show this help",
                    title="Help",
                ))
                continue
            
            elif not user_input.strip():
                continue
            
            # Prepare message
            messages = [Message(role="user", content=user_input)]
            
            # Generate response
            console.print("\n[cyan]Assistant:[/cyan] ", end="")
            
            options = InferenceOptions(temperature=temperature)
            
            token_generator, session_id = await runtime.chat(
                model_name=model_name,
                messages=messages,
                session_id=session_id,
                options=options,
                stream=True,
            )
            
            # Stream and display tokens (sanitization happens in engine layer)
            response_text = ""
            
            for token in token_generator:
                console.print(token, end="")
                response_text += token
            
            console.print()  # New line after response
            
            # Save conversation turn
            await runtime.save_assistant_response(
                session_id=session_id,
                user_message=user_input,
                assistant_message=response_text,
            )
            
            # Add visual separator
            console.print("[dim]" + "─" * 80 + "[/dim]")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Use /exit to quit.[/yellow]")
        
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def run_command(
    model: str = typer.Argument(..., help="Model name to use"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
):
    """
    Run an interactive chat session with a model.
    
    Start a REPL-style conversation with the specified model.
    History is maintained across messages within the session.
    """
    console = Console()
    
    try:
        # Verify model exists
        from app.models.registry import get_registry
        registry = get_registry()
        registry.get_model(model)
        
        # Run interactive chat
        asyncio.run(interactive_chat(model, temperature))
    
    except ModelNotFoundError:
        console.print(f"[red]Error: Model '{model}' not found[/red]")
        console.print("\nAvailable models:")
        
        from app.models.registry import get_registry
        registry = get_registry()
        models = registry.list_models()
        
        for m in models:
            console.print(f"  - {m.name}")
        
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
