# CS Tutor LLM

A **fully local, offline** Large Language Model for teaching computer science concepts. Build, train, and run your own AI tutor that explains CS topics, generates practice problems, runs quizzes, and grades answers.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- Concept Explanations
- Practice Problems
- Interactive Quizzes
- Answer Grading
- Fully Offline
- Efficient

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CS Tutor LLM System                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   CLI App   │  │  Inference  │  │   Model     │          │
│  │   (Typer)   │─▶│   Engine    │─▶│  (GPT-2)    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         │                │                │                  │
│         ▼                ▼                ▼                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │    Rich     │  │  Tokenizer  │  │ Quantized   │          │
│  │   Output    │  │    (BPE)    │  │  Weights    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Installation

```bash
# Clone the repository
git clone https://github.com/krishmalikk/minillm-cstutor.git
cd minillm-cstutor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Using the CLI

```bash
# Explain a concept
tutor explain "binary search tree" --level beginner

# Generate practice problems
tutor practice --topic algorithms --count 3 --difficulty intermediate

# Take a quiz
tutor quiz --topic sorting --questions 5

# Grade an answer
tutor grade --question "What is a hash table?" --answer "A data structure..."

# Interactive chat
tutor chat
```

## Dataset Format

### Concept Explanation

```json
{
  "instruction": "Explain the concept of recursion in programming.",
  "input": "",
  "output": "Recursion is a programming technique where a function calls itself...",
  "task_type": "concept_explanation",
  "topic": "programming",
  "difficulty": "intermediate"
}
```

### Practice Problem

```json
{
  "instruction": "Solve the following problem: Implement binary search.",
  "input": "",
  "output": "## Solution\n\n```python\ndef binary_search(arr, target):\n    ...",
  "task_type": "practice_problem",
  "topic": "algorithms",
  "difficulty": "intermediate"
}
```

### Quiz Question

```json
{
  "instruction": "Quiz: What is the time complexity of quicksort?\n\nA. O(n)\nB. O(n log n)\nC. O(n²)\nD. O(log n)",
  "input": "",
  "output": "The correct answer is B. O(n log n)\n\nExplanation: ...",
  "task_type": "quiz_question",
  "topic": "algorithms",
  "difficulty": "intermediate"
}
```

### Answer Grading

```json
{
  "instruction": "Grade the following student answer.",
  "input": "Question: What is a hash table?\n\nStudent Answer: A hash table stores data...",
  "output": "## Grade: 85%\n\n## Feedback\n...\n\n## Strengths\n- ...\n\n## Improvements\n- ...",
  "task_type": "answer_grading",
  "topic": "data_structures"
}
```


### Concept Explanation

```
tutor explain "binary search" --level beginner

Binary Search

Binary search is a fast searching algorithm that finds items in a sorted list.

## How It Works
Think of it like finding a word in a dictionary:
1. Open to the middle
2. If your word comes before, look in the first half
3. If after, look in the second half
4. Repeat until found

## Time Complexity
- O(log n) - much faster than checking every item!

## Code Example
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

### Quiz

```
tutor quiz --topic sorting --questions 1

Question 1/1

What is the average time complexity of Merge Sort?

  A. O(n)
  B. O(n log n)
  C. O(n^2)
  D. O(log n)

Your answer (A/B/C/D): B

Correct!

Explanation: Merge Sort always divides the array in half (log n levels)
and at each level, all n elements are processed during the merge step.
This gives O(n * log n) = O(n log n).
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


- Inspired by LLaMA, GPT-2, and modern transformer architectures
- Built with PyTorch, Typer, and Rich
- Training techniques from "Scaling Language Models" research
