#!/usr/bin/env python3
"""
CS Tutor CLI Application.

A command-line interface for the CS Tutor LLM.

Usage:
    tutor explain "binary search tree" --level beginner
    tutor practice --topic graphs --count 3 --difficulty medium
    tutor quiz --topic sorting --questions 5
    tutor grade --question-file q.txt --answer-file a.txt
    tutor chat
"""

import sys
from pathlib import Path
from typing import Optional
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    import argparse


# Initialize app
if HAS_RICH:
    app = typer.Typer(
        name="tutor",
        help="🎓 CS Tutor - Your local AI computer science tutor",
        add_completion=False,
    )
    console = Console()


class DifficultyChoice(str, Enum):
    """Difficulty level choices."""
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class TopicChoice(str, Enum):
    """Topic choices."""
    data_structures = "data_structures"
    algorithms = "algorithms"
    operating_systems = "operating_systems"
    discrete_math = "discrete_math"
    machine_learning = "machine_learning"
    networks = "networks"
    databases = "databases"
    programming = "programming"


# Global engine (lazy loaded)
_engine = None


def get_engine(model_path: str = "outputs/final_model"):
    """Get or create the inference engine."""
    global _engine
    if _engine is None:
        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Loading model...", total=None)
                from src.inference import CSTutorEngine
                _engine = CSTutorEngine.from_pretrained(model_path)
        else:
            print("Loading model...")
            from src.inference import CSTutorEngine
            _engine = CSTutorEngine.from_pretrained(model_path)
    return _engine


if HAS_RICH:
    @app.command()
    def explain(
        concept: str = typer.Argument(..., help="The CS concept to explain"),
        level: DifficultyChoice = typer.Option(
            DifficultyChoice.intermediate,
            "--level", "-l",
            help="Explanation difficulty level",
        ),
        model: str = typer.Option(
            "outputs/final_model",
            "--model", "-m",
            help="Path to model directory",
        ),
        no_examples: bool = typer.Option(
            False,
            "--no-examples",
            help="Skip code examples",
        ),
    ):
        """
        📚 Explain a CS concept clearly.
        
        Examples:
            tutor explain "binary search tree"
            tutor explain "quicksort" --level advanced
            tutor explain "TCP/IP" --level beginner
        """
        console.print(f"\n[bold blue]📚 Explaining: {concept}[/bold blue]\n")
        
        engine = get_engine(model)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Generating explanation...", total=None)
            explanation = engine.explain(
                concept,
                level=level.value,
                include_examples=not no_examples,
            )
        
        console.print(Panel(
            Markdown(explanation),
            title=f"[bold]{concept}[/bold]",
            subtitle=f"Level: {level.value}",
            border_style="blue",
        ))
    
    
    @app.command()
    def practice(
        topic: TopicChoice = typer.Option(
            ...,
            "--topic", "-t",
            help="Topic for practice problems",
        ),
        count: int = typer.Option(
            1,
            "--count", "-n",
            help="Number of problems to generate",
        ),
        difficulty: DifficultyChoice = typer.Option(
            DifficultyChoice.intermediate,
            "--difficulty", "-d",
            help="Problem difficulty",
        ),
        model: str = typer.Option(
            "outputs/final_model",
            "--model", "-m",
            help="Path to model directory",
        ),
        show_solution: bool = typer.Option(
            True,
            "--solution/--no-solution",
            help="Show solutions",
        ),
    ):
        """
        💪 Generate practice problems.
        
        Examples:
            tutor practice --topic algorithms --count 3
            tutor practice -t data_structures -n 2 -d advanced
        """
        console.print(f"\n[bold green]💪 Generating {count} {topic.value} problems[/bold green]\n")
        
        engine = get_engine(model)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Generating problems...", total=None)
            problems = engine.generate_problems(
                topic.value.replace("_", " "),
                difficulty=difficulty.value,
                n=count,
                include_solution=show_solution,
            )
        
        for i, problem in enumerate(problems, 1):
            console.print(Panel(
                Markdown(f"**Problem:**\n\n{problem.problem}\n\n"
                        + (f"**Hints:**\n" + "\n".join(f"- {h}" for h in problem.hints) + "\n\n" if problem.hints else "")
                        + (f"**Solution:**\n\n{problem.solution}\n\n" if show_solution and problem.solution else "")
                        + (f"**Explanation:**\n\n{problem.explanation}" if problem.explanation else "")),
                title=f"[bold]Problem {i}[/bold]",
                subtitle=f"{topic.value} | {difficulty.value}",
                border_style="green",
            ))
            console.print()
    
    
    @app.command()
    def quiz(
        topic: TopicChoice = typer.Option(
            ...,
            "--topic", "-t",
            help="Quiz topic",
        ),
        questions: int = typer.Option(
            5,
            "--questions", "-n",
            help="Number of questions",
        ),
        difficulty: DifficultyChoice = typer.Option(
            DifficultyChoice.intermediate,
            "--difficulty", "-d",
            help="Quiz difficulty",
        ),
        model: str = typer.Option(
            "outputs/final_model",
            "--model", "-m",
            help="Path to model directory",
        ),
        interactive: bool = typer.Option(
            True,
            "--interactive/--show-all",
            help="Interactive mode vs show all at once",
        ),
    ):
        """
        📝 Take a quiz on a topic.
        
        Examples:
            tutor quiz --topic sorting
            tutor quiz -t data_structures -n 10 --difficulty advanced
        """
        console.print(f"\n[bold yellow]📝 Quiz: {topic.value}[/bold yellow]\n")
        
        engine = get_engine(model)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Generating quiz...", total=None)
            quiz_questions = engine.quiz_me(
                topic.value.replace("_", " "),
                difficulty=difficulty.value,
                num_questions=questions,
            )
        
        if interactive:
            score = 0
            for i, q in enumerate(quiz_questions, 1):
                console.print(Panel(
                    f"[bold]{q.question}[/bold]\n\n"
                    + "\n".join(f"  {chr(65+j)}. {opt}" for j, opt in enumerate(q.options)),
                    title=f"[bold]Question {i}/{len(quiz_questions)}[/bold]",
                    border_style="yellow",
                ))
                
                answer = typer.prompt("Your answer (A/B/C/D)").upper().strip()
                answer_idx = ord(answer) - ord('A') if answer in 'ABCD' else -1
                
                if answer_idx == q.correct_index:
                    console.print("[bold green]✓ Correct![/bold green]\n")
                    score += 1
                else:
                    correct_letter = chr(65 + q.correct_index)
                    console.print(f"[bold red]✗ Incorrect. The answer is {correct_letter}.[/bold red]\n")
                
                console.print(f"[dim]{q.explanation}[/dim]\n")
            
            # Final score
            percentage = (score / len(quiz_questions)) * 100
            console.print(Panel(
                f"[bold]Score: {score}/{len(quiz_questions)} ({percentage:.0f}%)[/bold]",
                title="Quiz Complete",
                border_style="cyan",
            ))
        else:
            for i, q in enumerate(quiz_questions, 1):
                console.print(Panel(
                    Markdown(f"**{q.question}**\n\n"
                            + "\n".join(f"- **{chr(65+j)}.** {opt}" for j, opt in enumerate(q.options))
                            + f"\n\n**Correct Answer:** {chr(65+q.correct_index)}\n\n"
                            + f"**Explanation:** {q.explanation}"),
                    title=f"[bold]Question {i}[/bold]",
                    border_style="yellow",
                ))
                console.print()
    
    
    @app.command()
    def grade(
        question: Optional[str] = typer.Option(
            None,
            "--question", "-q",
            help="The question (inline)",
        ),
        answer: Optional[str] = typer.Option(
            None,
            "--answer", "-a",
            help="Student answer (inline)",
        ),
        question_file: Optional[Path] = typer.Option(
            None,
            "--question-file",
            help="File containing the question",
        ),
        answer_file: Optional[Path] = typer.Option(
            None,
            "--answer-file",
            help="File containing the answer",
        ),
        reference_file: Optional[Path] = typer.Option(
            None,
            "--reference-file",
            help="File containing reference solution",
        ),
        model: str = typer.Option(
            "outputs/final_model",
            "--model", "-m",
            help="Path to model directory",
        ),
    ):
        """
        ✅ Grade a student's answer.
        
        Examples:
            tutor grade --question "What is a BST?" --answer "A tree structure..."
            tutor grade --question-file q.txt --answer-file a.txt
        """
        # Load from files if provided
        if question_file:
            question = question_file.read_text()
        if answer_file:
            answer = answer_file.read_text()
        
        reference = None
        if reference_file:
            reference = reference_file.read_text()
        
        if not question or not answer:
            console.print("[red]Error: Please provide both question and answer[/red]")
            raise typer.Exit(1)
        
        console.print("\n[bold magenta]✅ Grading Answer[/bold magenta]\n")
        
        engine = get_engine(model)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Analyzing answer...", total=None)
            result = engine.grade_answer(question, answer, reference)
        
        # Create score color
        if result.score >= 0.8:
            score_color = "green"
        elif result.score >= 0.6:
            score_color = "yellow"
        else:
            score_color = "red"
        
        console.print(Panel(
            f"[bold {score_color}]Score: {int(result.score * 100)}%[/bold {score_color}]\n\n"
            + f"**Feedback:**\n{result.feedback}\n\n"
            + "**Strengths:**\n" + "\n".join(f"  ✓ {s}" for s in result.strengths) + "\n\n"
            + "**Areas for Improvement:**\n" + "\n".join(f"  → {i}" for i in result.improvements),
            title="[bold]Grading Result[/bold]",
            border_style="magenta",
        ))
    
    
    @app.command()
    def chat(
        model: str = typer.Option(
            "outputs/final_model",
            "--model", "-m",
            help="Path to model directory",
        ),
    ):
        """
        💬 Start an interactive chat session.
        
        Ask questions about any CS topic!
        Type 'quit' or 'exit' to end the session.
        """
        console.print(Panel(
            "[bold]Welcome to CS Tutor Chat![/bold]\n\n"
            "Ask me anything about computer science.\n"
            "Type 'quit' or 'exit' to end the session.",
            title="💬 CS Tutor",
            border_style="cyan",
        ))
        
        engine = get_engine(model)
        history = []
        
        while True:
            try:
                user_input = typer.prompt("\n[You]")
            except (KeyboardInterrupt, EOFError):
                break
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print("\n[dim]Goodbye! Happy learning! 📚[/dim]\n")
                break
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Thinking...", total=None)
                response = engine.chat(user_input, history)
            
            console.print(f"\n[bold cyan][Tutor][/bold cyan]")
            console.print(Markdown(response))
            
            history.append({"user": user_input, "assistant": response})
    
    
    @app.command()
    def trace(
        algorithm: str = typer.Argument(..., help="Algorithm to trace"),
        input_data: str = typer.Argument(..., help="Input data to trace through"),
        model: str = typer.Option(
            "outputs/final_model",
            "--model", "-m",
            help="Path to model directory",
        ),
    ):
        """
        🔍 Trace through an algorithm step-by-step.
        
        Examples:
            tutor trace "binary search" "[1,3,5,7,9], target=5"
            tutor trace "quicksort" "[3,1,4,1,5,9,2,6]"
        """
        console.print(f"\n[bold]🔍 Tracing: {algorithm}[/bold]\n")
        
        engine = get_engine(model)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Generating trace...", total=None)
            trace_output = engine.step_by_step(algorithm, input_data)
        
        console.print(Panel(
            Markdown(trace_output),
            title=f"[bold]{algorithm}[/bold]",
            subtitle=f"Input: {input_data}",
            border_style="blue",
        ))
    
    
    @app.command()
    def version():
        """Show version information."""
        console.print(Panel(
            "[bold]CS Tutor LLM[/bold]\n\n"
            "Version: 1.0.0\n"
            "A local, offline CS tutor powered by LLM.\n\n"
            "GitHub: https://github.com/yourusername/cs-tutor-llm",
            title="About",
            border_style="blue",
        ))


def main():
    """Main entry point."""
    if HAS_RICH:
        app()
    else:
        # Fallback to argparse if typer/rich not available
        parser = argparse.ArgumentParser(
            description="CS Tutor - Your local AI computer science tutor"
        )
        subparsers = parser.add_subparsers(dest="command")
        
        # Explain command
        explain_parser = subparsers.add_parser("explain", help="Explain a concept")
        explain_parser.add_argument("concept", help="Concept to explain")
        explain_parser.add_argument("--level", "-l", default="intermediate")
        explain_parser.add_argument("--model", "-m", default="outputs/final_model")
        
        # Practice command
        practice_parser = subparsers.add_parser("practice", help="Generate problems")
        practice_parser.add_argument("--topic", "-t", required=True)
        practice_parser.add_argument("--count", "-n", type=int, default=1)
        practice_parser.add_argument("--difficulty", "-d", default="intermediate")
        practice_parser.add_argument("--model", "-m", default="outputs/final_model")
        
        args = parser.parse_args()
        
        if args.command == "explain":
            engine = get_engine(args.model)
            result = engine.explain(args.concept, level=args.level)
            print(f"\n=== {args.concept} ===\n")
            print(result)
        elif args.command == "practice":
            engine = get_engine(args.model)
            problems = engine.generate_problems(
                args.topic, difficulty=args.difficulty, n=args.count
            )
            for i, p in enumerate(problems, 1):
                print(f"\n=== Problem {i} ===\n")
                print(p.problem)
                if p.solution:
                    print(f"\nSolution:\n{p.solution}")
        else:
            parser.print_help()


if __name__ == "__main__":
    main()

