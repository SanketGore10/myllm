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
from app.utils.context import quiet_mode


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
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
            
            # Handle empty input
            if not user_input.strip():
                continue
            
            # Handle commands
            if user_input.strip().lower() == "/exit":
                console.print("\n[dim]Goodbye! 👋[/dim]")
                break
            
            if user_input.strip().lower() == "/clear":
                session_id = None
                console.print("\n[dim]History cleared[/dim]")
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
            options = InferenceOptions(
                temperature=0.7,
                max_tokens=512,
            )
            
            # Use quiet_mode for clean output (suppress logs + stderr)
            # Only in non-debug mode
            use_quiet = not debug
            
            if use_quiet:
                # Clean UX mode (Ollama-style)
                with quiet_mode():
                    token_generator, session_id = await runtime.chat(
                        model_name=model_name,
                        messages=messages,
                        session_id=session_id,
                        options=options,
                        stream=True,
                    )
            else:
                # Debug mode - show all logs
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
    model: str = typer.Argument(..., help="Model name to run"),
    system_prompt: str = typer.Option(
        "You are a helpful AI assistant.",
        "--system",
        "-s",
        help="System prompt"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable verbose logging (shows model loading, inference details)"
    ),
):
    """
    Run interactive chat with a model.
    
    Provides a clean, Ollama-style chat experience.
    Use --debug to see internal logs and diagnostics.
    """
    import os
    
    # Set log level based on debug flag
    if debug:
        os.environ["MYLLM_LOG_LEVEL"] = "INFO"
        console.print("[dim]Debug mode enabled - verbose logs will show[/dim]\n")
    else:
        os.environ["MYLLM_LOG_LEVEL"] = "WARNING"
    
    # Re-setup logging with new level
    from app.utils.logging import setup_logging
    setup_logging()
    
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
