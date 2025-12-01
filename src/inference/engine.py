"""
Inference Engine for CS Tutor LLM.

Provides high-level tutoring functions:
- explain(concept, level)
- generate_problems(topic, difficulty, n)
- quiz_me(topic, difficulty)
- grade_answer(question, student_answer, reference_solution)
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

import torch
import torch.nn.functional as F


class DifficultyLevel(Enum):
    """Difficulty levels for content generation."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Topic(Enum):
    """CS topics supported by the tutor."""
    DATA_STRUCTURES = "data structures"
    ALGORITHMS = "algorithms"
    OPERATING_SYSTEMS = "operating systems"
    DISCRETE_MATH = "discrete math"
    MACHINE_LEARNING = "machine learning"
    NETWORKS = "networks"
    DATABASES = "databases"
    PROGRAMMING = "programming"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class QuizQuestion:
    """A generated quiz question."""
    question: str
    options: List[str]
    correct_index: int
    explanation: str


@dataclass
class PracticeProblem:
    """A generated practice problem."""
    problem: str
    hints: List[str]
    solution: str
    explanation: str


@dataclass
class GradingResult:
    """Result of answer grading."""
    score: float  # 0.0 to 1.0
    feedback: str
    strengths: List[str]
    improvements: List[str]


class CSTutorEngine:
    """
    Main inference engine for the CS Tutor LLM.
    
    Provides high-level API for tutoring tasks:
    - Concept explanations
    - Practice problem generation
    - Quiz generation
    - Answer grading
    
    Usage:
        engine = CSTutorEngine.from_pretrained("outputs/final_model")
        
        # Explain a concept
        explanation = engine.explain("binary search tree", level="beginner")
        
        # Generate practice problems
        problems = engine.generate_problems("graphs", difficulty="intermediate", n=3)
        
        # Quiz
        quiz = engine.quiz_me("sorting", num_questions=5)
        
        # Grade an answer
        result = engine.grade_answer(question, student_answer, reference_solution)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "auto",
        generation_config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize the inference engine.
        
        Args:
            model: The CS Tutor LLM model
            tokenizer: Tokenizer for the model
            device: Device to run inference on ("auto", "cuda", "mps", "cpu")
            generation_config: Default generation settings
        """
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config or GenerationConfig()
        
        # Setup device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"CSTutorEngine initialized on {self.device}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "auto",
        **kwargs,
    ) -> "CSTutorEngine":
        """
        Load engine from a pretrained model directory.
        
        Args:
            model_path: Path to model directory
            device: Device for inference
        
        Returns:
            CSTutorEngine instance
        """
        from src.model import CSTutorLLM, ModelConfig, CSTutorTokenizer
        
        model_path = Path(model_path)
        
        # Load config and model
        config = ModelConfig.from_pretrained(str(model_path))
        model = CSTutorLLM(config)
        
        weights_path = model_path / "model.pt"
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        # Load tokenizer
        tokenizer_path = model_path / "tokenizer"
        if tokenizer_path.exists():
            tokenizer = CSTutorTokenizer.from_pretrained(str(tokenizer_path))
        else:
            tokenizer = CSTutorTokenizer(vocab_size=config.vocab_size)
        
        return cls(model, tokenizer, device, **kwargs)
    
    def _generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
        
        Returns:
            Generated text
        """
        config = config or self.generation_config
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            add_bos=True,
            add_eos=False,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate
        with torch.no_grad():
            if config.do_sample:
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                )
            else:
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=config.max_new_tokens,
                    temperature=0,
                )
        
        # Decode (skip input tokens)
        generated_ids = output_ids[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
        
        return response.strip()
    
    def explain(
        self,
        concept: str,
        level: Union[str, DifficultyLevel] = DifficultyLevel.INTERMEDIATE,
        include_examples: bool = True,
        include_analogy: bool = True,
    ) -> str:
        """
        Explain a CS concept clearly.
        
        Args:
            concept: The concept to explain (e.g., "binary search tree", "quicksort")
            level: Explanation level (beginner, intermediate, advanced)
            include_examples: Include code examples
            include_analogy: Include real-world analogy
        
        Returns:
            Clear explanation of the concept
        """
        if isinstance(level, DifficultyLevel):
            level = level.value
        
        prompt = f"""### Instruction:
Explain the concept of "{concept}" in computer science.

Level: {level}

Please include:
- Clear definition
- Key properties and characteristics
- Time/space complexity (if applicable)
{"- Real-world analogy" if include_analogy else ""}
{"- Code example with explanation" if include_examples else ""}
- Common use cases

Make the explanation clear and educational.

### Response:
"""
        
        return self._generate(prompt)
    
    def generate_problems(
        self,
        topic: Union[str, Topic],
        difficulty: Union[str, DifficultyLevel] = DifficultyLevel.INTERMEDIATE,
        n: int = 1,
        include_hints: bool = True,
        include_solution: bool = True,
    ) -> List[PracticeProblem]:
        """
        Generate practice problems on a topic.
        
        Args:
            topic: CS topic for problems
            difficulty: Problem difficulty
            n: Number of problems to generate
            include_hints: Include hints
            include_solution: Include solution
        
        Returns:
            List of PracticeProblem objects
        """
        if isinstance(topic, Topic):
            topic = topic.value
        if isinstance(difficulty, DifficultyLevel):
            difficulty = difficulty.value
        
        problems = []
        
        for i in range(n):
            prompt = f"""### Instruction:
Create a {difficulty} practice problem about {topic}.

Include:
- Clear problem statement
- Input/output format
- Constraints
- Example test cases
{"- 2-3 progressive hints" if include_hints else ""}
{"- Complete solution with explanation" if include_solution else ""}

Make sure the problem is original and educational.

### Response:
"""
            
            response = self._generate(prompt)
            
            # Parse response into PracticeProblem
            problem = self._parse_problem(response)
            problems.append(problem)
        
        return problems
    
    def _parse_problem(self, response: str) -> PracticeProblem:
        """Parse generated text into PracticeProblem structure."""
        # Simple parsing - in production, use more robust parsing
        sections = {
            "problem": "",
            "hints": [],
            "solution": "",
            "explanation": "",
        }
        
        current_section = "problem"
        lines = response.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if "hint" in line_lower and ":" in line:
                current_section = "hints"
                sections["hints"].append(line.split(":", 1)[-1].strip())
            elif "solution" in line_lower:
                current_section = "solution"
            elif "explanation" in line_lower:
                current_section = "explanation"
            else:
                if current_section == "hints" and line.strip():
                    if line.strip().startswith(("-", "*", "•")) or line.strip()[0].isdigit():
                        sections["hints"].append(line.strip().lstrip("-*•0123456789. "))
                elif current_section in sections:
                    if isinstance(sections[current_section], str):
                        sections[current_section] += line + "\n"
        
        return PracticeProblem(
            problem=sections["problem"].strip(),
            hints=sections["hints"],
            solution=sections["solution"].strip(),
            explanation=sections["explanation"].strip(),
        )
    
    def quiz_me(
        self,
        topic: Union[str, Topic],
        difficulty: Union[str, DifficultyLevel] = DifficultyLevel.INTERMEDIATE,
        num_questions: int = 5,
    ) -> List[QuizQuestion]:
        """
        Generate a quiz on a topic.
        
        Args:
            topic: Topic for the quiz
            difficulty: Quiz difficulty
            num_questions: Number of questions
        
        Returns:
            List of QuizQuestion objects
        """
        if isinstance(topic, Topic):
            topic = topic.value
        if isinstance(difficulty, DifficultyLevel):
            difficulty = difficulty.value
        
        questions = []
        
        for i in range(num_questions):
            prompt = f"""### Instruction:
Create a {difficulty} multiple-choice quiz question about {topic}.

Format:
Question: [Your question here]

A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]

Correct Answer: [Letter]

Explanation: [Why the correct answer is correct and why others are wrong]

Make sure:
- The question tests understanding, not just memorization
- All options are plausible
- The explanation is educational

### Response:
"""
            
            response = self._generate(prompt)
            question = self._parse_quiz(response)
            questions.append(question)
        
        return questions
    
    def _parse_quiz(self, response: str) -> QuizQuestion:
        """Parse quiz response into QuizQuestion."""
        lines = response.split('\n')
        
        question = ""
        options = []
        correct_index = 0
        explanation = ""
        
        in_explanation = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.lower().startswith("question:"):
                question = line.split(":", 1)[1].strip()
            elif line.startswith(("A.", "A)")):
                options.append(line[2:].strip())
            elif line.startswith(("B.", "B)")):
                options.append(line[2:].strip())
            elif line.startswith(("C.", "C)")):
                options.append(line[2:].strip())
            elif line.startswith(("D.", "D)")):
                options.append(line[2:].strip())
            elif "correct answer" in line.lower():
                answer_part = line.split(":")[-1].strip().upper()
                if "A" in answer_part:
                    correct_index = 0
                elif "B" in answer_part:
                    correct_index = 1
                elif "C" in answer_part:
                    correct_index = 2
                elif "D" in answer_part:
                    correct_index = 3
            elif "explanation" in line.lower():
                in_explanation = True
                explanation = line.split(":", 1)[-1].strip()
            elif in_explanation:
                explanation += " " + line
        
        # Ensure we have 4 options
        while len(options) < 4:
            options.append("[Option not parsed]")
        
        return QuizQuestion(
            question=question or response[:100],
            options=options[:4],
            correct_index=correct_index,
            explanation=explanation.strip(),
        )
    
    def grade_answer(
        self,
        question: str,
        student_answer: str,
        reference_solution: Optional[str] = None,
    ) -> GradingResult:
        """
        Grade a student's answer.
        
        Args:
            question: The original question
            student_answer: Student's response
            reference_solution: Optional reference solution
        
        Returns:
            GradingResult with score, feedback, strengths, improvements
        """
        reference_part = f"\n\nReference Solution:\n{reference_solution}" if reference_solution else ""
        
        prompt = f"""### Instruction:
Grade the following student answer and provide detailed feedback.

Question:
{question}
{reference_part}

Student Answer:
{student_answer}

Please provide:
1. A score from 0-100%
2. Detailed feedback on the answer
3. 2-3 specific strengths
4. 2-3 specific areas for improvement

Format your response as:
Score: [X]%

Feedback:
[Your detailed feedback]

Strengths:
- [Strength 1]
- [Strength 2]

Areas for Improvement:
- [Improvement 1]
- [Improvement 2]

### Response:
"""
        
        response = self._generate(prompt)
        return self._parse_grading(response)
    
    def _parse_grading(self, response: str) -> GradingResult:
        """Parse grading response into GradingResult."""
        score = 0.7  # Default
        feedback = ""
        strengths = []
        improvements = []
        
        current_section = "feedback"
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            
            if "score:" in line_lower or "grade:" in line_lower:
                # Extract score
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    score = min(100, max(0, int(numbers[0]))) / 100.0
            elif "strength" in line_lower:
                current_section = "strengths"
            elif "improvement" in line_lower or "area" in line_lower:
                current_section = "improvements"
            elif "feedback" in line_lower:
                current_section = "feedback"
            elif line.startswith(("-", "*", "•")):
                item = line.lstrip("-*• ").strip()
                if current_section == "strengths":
                    strengths.append(item)
                elif current_section == "improvements":
                    improvements.append(item)
            elif current_section == "feedback":
                feedback += line + " "
        
        return GradingResult(
            score=score,
            feedback=feedback.strip(),
            strengths=strengths or ["Understanding of core concepts"],
            improvements=improvements or ["Add more detail to explanations"],
        )
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Free-form chat about CS topics.
        
        Args:
            message: User's message
            history: Previous conversation history
        
        Returns:
            Assistant response
        """
        # Build conversation context
        context = ""
        if history:
            for turn in history[-5:]:  # Keep last 5 turns
                context += f"User: {turn.get('user', '')}\n"
                context += f"Assistant: {turn.get('assistant', '')}\n\n"
        
        prompt = f"""### Instruction:
You are a helpful CS tutor. Answer the student's question clearly and educationally.

{f"Previous conversation:{chr(10)}{context}" if context else ""}

Student: {message}

### Response:
"""
        
        return self._generate(prompt)
    
    def step_by_step(
        self,
        algorithm: str,
        input_data: str,
    ) -> str:
        """
        Walk through an algorithm step by step.
        
        Args:
            algorithm: Name of the algorithm
            input_data: Input to trace through
        
        Returns:
            Step-by-step walkthrough
        """
        prompt = f"""### Instruction:
Walk through the {algorithm} algorithm step by step with this input: {input_data}

For each step, show:
1. The current state
2. The operation being performed
3. How the state changes
4. Key decisions made

Make it clear and educational.

### Response:
"""
        
        return self._generate(prompt)

