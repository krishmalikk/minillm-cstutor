"""
Dataset Schema for CS Tutor LLM.

Defines the structure for instruction-tuning examples
covering various CS topics and task types.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import json


class TaskType(Enum):
    """Types of tutoring tasks."""
    CONCEPT_EXPLANATION = "concept_explanation"
    PRACTICE_PROBLEM = "practice_problem"
    QUIZ_QUESTION = "quiz_question"
    ANSWER_GRADING = "answer_grading"
    CODE_REVIEW = "code_review"
    DEBUGGING_HELP = "debugging_help"
    STEP_BY_STEP = "step_by_step"


class Topic(Enum):
    """CS topics covered by the tutor."""
    DATA_STRUCTURES = "data_structures"
    ALGORITHMS = "algorithms"
    OPERATING_SYSTEMS = "operating_systems"
    DISCRETE_MATH = "discrete_math"
    MACHINE_LEARNING = "machine_learning"
    NETWORKS = "networks"
    DATABASES = "databases"
    PROGRAMMING_LANGUAGES = "programming_languages"
    COMPUTER_ARCHITECTURE = "computer_architecture"
    SOFTWARE_ENGINEERING = "software_engineering"


class Difficulty(Enum):
    """Difficulty levels for content."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class InstructionExample:
    """
    Single instruction-tuning example.
    
    JSONL Schema:
    {
        "instruction": str,      # The task instruction
        "input": str,            # Additional context (can be empty)
        "output": str,           # Expected response
        "task_type": str,        # Type of tutoring task
        "topic": str,            # CS topic
        "subtopic": str,         # Specific subtopic
        "difficulty": str,       # Difficulty level
        "metadata": dict         # Additional metadata
    }
    """
    instruction: str
    input: str
    output: str
    task_type: str = TaskType.CONCEPT_EXPLANATION.value
    topic: str = Topic.DATA_STRUCTURES.value
    subtopic: str = ""
    difficulty: str = Difficulty.INTERMEDIATE.value
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_jsonl(self) -> str:
        """Convert to JSONL string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstructionExample":
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_jsonl(cls, line: str) -> "InstructionExample":
        """Create from JSONL string."""
        return cls.from_dict(json.loads(line))
    
    def format_prompt(self, include_output: bool = True) -> str:
        """
        Format as training prompt.
        
        Args:
            include_output: Whether to include the output (for inference, set False)
        
        Returns:
            Formatted prompt string
        """
        prompt = f"### Instruction:\n{self.instruction}\n\n"
        
        if self.input:
            prompt += f"### Input:\n{self.input}\n\n"
        
        prompt += "### Response:\n"
        
        if include_output:
            prompt += self.output
        
        return prompt


@dataclass
class QuizQuestion:
    """Schema for quiz questions."""
    question: str
    options: List[str]
    correct_answer: int  # Index of correct option
    explanation: str
    topic: str
    difficulty: str
    
    def to_instruction_example(self) -> InstructionExample:
        """Convert to InstructionExample format."""
        options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(self.options))
        
        instruction = f"Quiz question on {self.topic}:\n\n{self.question}\n\n{options_text}"
        
        output = f"The correct answer is {chr(65+self.correct_answer)}. {self.options[self.correct_answer]}\n\n"
        output += f"Explanation: {self.explanation}"
        
        return InstructionExample(
            instruction=instruction,
            input="",
            output=output,
            task_type=TaskType.QUIZ_QUESTION.value,
            topic=self.topic,
            difficulty=self.difficulty,
        )


@dataclass
class PracticeProblem:
    """Schema for practice problems."""
    problem_statement: str
    topic: str
    subtopic: str
    difficulty: str
    hints: List[str]
    solution: str
    explanation: str
    test_cases: Optional[List[Dict[str, str]]] = None
    
    def to_instruction_example(self) -> InstructionExample:
        """Convert to InstructionExample format."""
        instruction = f"Solve the following {self.topic} problem:\n\n{self.problem_statement}"
        
        output = f"## Solution\n\n{self.solution}\n\n"
        output += f"## Explanation\n\n{self.explanation}\n\n"
        
        if self.hints:
            output += "## Hints (if needed)\n\n"
            for i, hint in enumerate(self.hints, 1):
                output += f"{i}. {hint}\n"
        
        return InstructionExample(
            instruction=instruction,
            input="",
            output=output,
            task_type=TaskType.PRACTICE_PROBLEM.value,
            topic=self.topic,
            subtopic=self.subtopic,
            difficulty=self.difficulty,
            metadata={"hints": self.hints, "test_cases": self.test_cases},
        )


@dataclass
class GradingExample:
    """Schema for answer grading examples."""
    question: str
    student_answer: str
    reference_solution: str
    feedback: str
    score: float  # 0.0 to 1.0
    strengths: List[str]
    improvements: List[str]
    topic: str
    
    def to_instruction_example(self) -> InstructionExample:
        """Convert to InstructionExample format."""
        instruction = "Grade the following student answer and provide detailed feedback."
        
        input_text = f"## Question\n{self.question}\n\n"
        input_text += f"## Reference Solution\n{self.reference_solution}\n\n"
        input_text += f"## Student Answer\n{self.student_answer}"
        
        output = f"## Grade: {int(self.score * 100)}%\n\n"
        output += f"## Feedback\n{self.feedback}\n\n"
        
        if self.strengths:
            output += "## Strengths\n"
            for s in self.strengths:
                output += f"- {s}\n"
            output += "\n"
        
        if self.improvements:
            output += "## Areas for Improvement\n"
            for i in self.improvements:
                output += f"- {i}\n"
        
        return InstructionExample(
            instruction=instruction,
            input=input_text,
            output=output,
            task_type=TaskType.ANSWER_GRADING.value,
            topic=self.topic,
            metadata={"score": self.score},
        )


# Prompt templates for different task types
PROMPT_TEMPLATES = {
    TaskType.CONCEPT_EXPLANATION: """Explain the concept of {concept} in {topic}.
Level: {level}
Include:
- Clear definition
- Key properties
- Real-world analogy
- Common use cases
- Example (if applicable)""",

    TaskType.PRACTICE_PROBLEM: """Create a {difficulty} practice problem about {subtopic} in {topic}.
Include:
- Problem statement
- Input/output format
- Constraints
- Example test cases
- Solution with explanation""",

    TaskType.QUIZ_QUESTION: """Generate a {difficulty} multiple-choice quiz question about {subtopic} in {topic}.
Include:
- Clear question
- 4 answer options (one correct)
- Detailed explanation of the correct answer""",

    TaskType.ANSWER_GRADING: """Grade the following answer about {topic}.
Provide:
- Numerical score (0-100)
- Detailed feedback
- Strengths identified
- Specific improvements needed""",

    TaskType.CODE_REVIEW: """Review the following {language} code implementing {algorithm}.
Analyze:
- Correctness
- Time complexity
- Space complexity
- Code style
- Potential improvements""",

    TaskType.DEBUGGING_HELP: """Help debug this {language} code for {problem}.
Identify:
- The bug(s)
- Root cause
- Step-by-step fix
- How to prevent similar bugs""",

    TaskType.STEP_BY_STEP: """Walk through {algorithm} step by step on this input: {input}.
Show:
- Each step clearly
- State changes
- Key decisions made
- Final result""",
}

