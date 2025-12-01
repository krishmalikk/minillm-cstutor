#!/usr/bin/env python3
"""
Dataset Generation Script for CS Tutor LLM.

Helps create and expand instruction-tuning datasets.

Usage:
    # Generate concept explanations
    python scripts/generate_dataset.py concepts \
        --topics "sorting,graphs,trees" \
        --output data/generated/concepts.jsonl
    
    # Generate practice problems
    python scripts/generate_dataset.py problems \
        --topics "algorithms,data_structures" \
        --count 50 \
        --output data/generated/problems.jsonl
    
    # Generate quizzes
    python scripts/generate_dataset.py quizzes \
        --topics "all" \
        --questions-per-topic 10 \
        --output data/generated/quizzes.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))


# Template examples for each topic
CONCEPT_TEMPLATES = {
    "data_structures": {
        "subtopics": [
            "array", "linked list", "stack", "queue", "hash table",
            "binary tree", "binary search tree", "AVL tree", "red-black tree",
            "B-tree", "heap", "priority queue", "graph", "trie",
            "disjoint set", "segment tree", "fenwick tree"
        ],
        "template": """## {concept}

### Definition
{definition}

### Key Properties
{properties}

### Operations and Complexity
| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
{complexity_table}

### Real-World Analogy
{analogy}

### Use Cases
{use_cases}

### Example Implementation
```python
{code_example}
```

### Common Interview Questions
{interview_questions}"""
    },
    "algorithms": {
        "subtopics": [
            "binary search", "linear search", "bubble sort", "insertion sort",
            "merge sort", "quick sort", "heap sort", "counting sort",
            "radix sort", "BFS", "DFS", "Dijkstra's algorithm",
            "Bellman-Ford", "Floyd-Warshall", "Kruskal's algorithm",
            "Prim's algorithm", "dynamic programming", "greedy algorithms",
            "divide and conquer", "backtracking", "two pointers",
            "sliding window", "topological sort"
        ],
        "template": """## {algorithm}

### Overview
{overview}

### Algorithm Steps
{steps}

### Time Complexity
- Best Case: {best_case}
- Average Case: {average_case}
- Worst Case: {worst_case}

### Space Complexity
{space_complexity}

### When to Use
{when_to_use}

### Example Trace
{trace_example}

### Implementation
```python
{code_example}
```

### Practice Problems
{practice_problems}"""
    },
    "operating_systems": {
        "subtopics": [
            "process", "thread", "process scheduling", "CPU scheduling",
            "memory management", "virtual memory", "paging", "segmentation",
            "deadlock", "synchronization", "mutex", "semaphore",
            "file systems", "I/O management", "interrupts",
            "system calls", "kernel", "user mode vs kernel mode"
        ],
    },
    "networks": {
        "subtopics": [
            "OSI model", "TCP/IP model", "TCP", "UDP", "HTTP", "HTTPS",
            "DNS", "DHCP", "ARP", "IP addressing", "subnetting",
            "routing", "switching", "firewalls", "load balancing",
            "socket programming", "REST API", "WebSocket"
        ],
    },
    "discrete_math": {
        "subtopics": [
            "sets", "functions", "relations", "logic", "proofs",
            "mathematical induction", "recursion", "counting",
            "permutations", "combinations", "probability",
            "graph theory", "trees", "Boolean algebra"
        ],
    },
    "machine_learning": {
        "subtopics": [
            "supervised learning", "unsupervised learning", "reinforcement learning",
            "linear regression", "logistic regression", "decision trees",
            "random forests", "SVM", "neural networks", "CNN", "RNN", "LSTM",
            "transformers", "attention mechanism", "gradient descent",
            "backpropagation", "regularization", "cross-validation",
            "bias-variance tradeoff", "overfitting", "underfitting"
        ],
    },
}


# Quiz question templates
QUIZ_TEMPLATES = [
    {
        "type": "time_complexity",
        "template": "What is the time complexity of {operation} in a {data_structure}?",
        "options_type": "complexity"
    },
    {
        "type": "comparison",
        "template": "Which of the following is true about {concept_a} compared to {concept_b}?",
        "options_type": "statements"
    },
    {
        "type": "definition",
        "template": "What best describes {concept}?",
        "options_type": "definitions"
    },
    {
        "type": "application",
        "template": "Which data structure/algorithm would be best for {scenario}?",
        "options_type": "structures"
    },
]


def generate_concept_explanation(topic: str, subtopic: str, difficulty: str) -> Dict:
    """Generate a concept explanation example."""
    return {
        "instruction": f"Explain the concept of {subtopic} in {topic}.",
        "input": "",
        "output": f"## {subtopic.title()}\n\n[Detailed explanation of {subtopic} would go here...]\n\n"
                 f"### Key Points\n- Point 1\n- Point 2\n- Point 3\n\n"
                 f"### Example\n```python\n# Example code for {subtopic}\n```\n\n"
                 f"### Use Cases\n- Use case 1\n- Use case 2",
        "task_type": "concept_explanation",
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "metadata": {}
    }


def generate_practice_problem(topic: str, subtopic: str, difficulty: str) -> Dict:
    """Generate a practice problem example."""
    return {
        "instruction": f"Solve the following {topic} problem about {subtopic}.",
        "input": f"[Problem statement about {subtopic}]\n\nInput: [description]\nOutput: [description]\n\nConstraints:\n- Constraint 1\n- Constraint 2",
        "output": f"## Solution\n\n```python\ndef solve_{subtopic.replace(' ', '_')}(...):\n    # Solution code\n    pass\n```\n\n"
                 f"## Explanation\n\n[Step by step explanation...]\n\n"
                 f"## Complexity\n- Time: O(...)\n- Space: O(...)\n\n"
                 f"## Hints\n1. Hint 1\n2. Hint 2",
        "task_type": "practice_problem",
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "metadata": {"has_solution": True, "has_hints": True}
    }


def generate_quiz_question(topic: str, subtopic: str, difficulty: str) -> Dict:
    """Generate a quiz question example."""
    return {
        "instruction": f"Quiz question on {topic}:\n\n[Question about {subtopic}]\n\n"
                      f"A. Option A\nB. Option B\nC. Option C\nD. Option D",
        "input": "",
        "output": f"The correct answer is B. Option B\n\n"
                 f"**Explanation:** [Detailed explanation of why B is correct and others are wrong]",
        "task_type": "quiz_question",
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "metadata": {"correct_answer": 1, "options": ["A", "B", "C", "D"]}
    }


def generate_grading_example(topic: str, subtopic: str) -> Dict:
    """Generate a grading example."""
    return {
        "instruction": "Grade the following student answer and provide detailed feedback.",
        "input": f"## Question\n[Question about {subtopic}]\n\n"
                f"## Reference Solution\n[Reference solution]\n\n"
                f"## Student Answer\n[Student's answer]",
        "output": f"## Grade: X%\n\n"
                 f"## Feedback\n[Detailed feedback]\n\n"
                 f"## Strengths\n- Strength 1\n- Strength 2\n\n"
                 f"## Areas for Improvement\n- Area 1\n- Area 2",
        "task_type": "answer_grading",
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": "intermediate",
        "metadata": {"score": 0.75}
    }


def main():
    parser = argparse.ArgumentParser(description="Generate CS Tutor dataset")
    subparsers = parser.add_subparsers(dest="command")
    
    # Concepts command
    concepts_parser = subparsers.add_parser("concepts", help="Generate concept explanations")
    concepts_parser.add_argument("--topics", type=str, default="all", help="Comma-separated topics or 'all'")
    concepts_parser.add_argument("--output", type=str, default="data/generated/concepts.jsonl")
    concepts_parser.add_argument("--difficulties", type=str, default="beginner,intermediate,advanced")
    
    # Problems command
    problems_parser = subparsers.add_parser("problems", help="Generate practice problems")
    problems_parser.add_argument("--topics", type=str, default="all")
    problems_parser.add_argument("--count", type=int, default=50)
    problems_parser.add_argument("--output", type=str, default="data/generated/problems.jsonl")
    
    # Quizzes command
    quizzes_parser = subparsers.add_parser("quizzes", help="Generate quiz questions")
    quizzes_parser.add_argument("--topics", type=str, default="all")
    quizzes_parser.add_argument("--questions-per-topic", type=int, default=10)
    quizzes_parser.add_argument("--output", type=str, default="data/generated/quizzes.jsonl")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("--data-dir", type=str, default="data/examples")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Get topics
    all_topics = list(CONCEPT_TEMPLATES.keys())
    
    if args.command == "stats":
        # Show statistics
        data_dir = Path(args.data_dir)
        total = 0
        by_type = {}
        by_topic = {}
        
        for jsonl_file in data_dir.glob("**/*.jsonl"):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        total += 1
                        
                        task_type = data.get("task_type", "unknown")
                        topic = data.get("topic", "unknown")
                        
                        by_type[task_type] = by_type.get(task_type, 0) + 1
                        by_topic[topic] = by_topic.get(topic, 0) + 1
        
        print("\n📊 Dataset Statistics")
        print("=" * 40)
        print(f"Total examples: {total}")
        
        print("\nBy Task Type:")
        for t, count in sorted(by_type.items()):
            print(f"  {t}: {count}")
        
        print("\nBy Topic:")
        for t, count in sorted(by_topic.items()):
            print(f"  {t}: {count}")
        
        return
    
    # Parse topics
    if args.topics == "all":
        topics = all_topics
    else:
        topics = [t.strip() for t in args.topics.split(",")]
    
    # Generate data
    examples = []
    
    if args.command == "concepts":
        difficulties = [d.strip() for d in args.difficulties.split(",")]
        
        for topic in topics:
            if topic in CONCEPT_TEMPLATES:
                subtopics = CONCEPT_TEMPLATES[topic].get("subtopics", [topic])
                for subtopic in subtopics:
                    for difficulty in difficulties:
                        examples.append(generate_concept_explanation(topic, subtopic, difficulty))
        
        print(f"Generated {len(examples)} concept explanations")
    
    elif args.command == "problems":
        for topic in topics:
            if topic in CONCEPT_TEMPLATES:
                subtopics = CONCEPT_TEMPLATES[topic].get("subtopics", [topic])
                for subtopic in subtopics[:args.count // len(topics) + 1]:
                    for difficulty in ["beginner", "intermediate", "advanced"]:
                        examples.append(generate_practice_problem(topic, subtopic, difficulty))
        
        examples = examples[:args.count]
        print(f"Generated {len(examples)} practice problems")
    
    elif args.command == "quizzes":
        for topic in topics:
            if topic in CONCEPT_TEMPLATES:
                subtopics = CONCEPT_TEMPLATES[topic].get("subtopics", [topic])
                for subtopic in subtopics[:args.questions_per_topic]:
                    for difficulty in ["beginner", "intermediate"]:
                        examples.append(generate_quiz_question(topic, subtopic, difficulty))
        
        print(f"Generated {len(examples)} quiz questions")
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"Saved to {output_path}")
    print("\n⚠️  Note: These are template placeholders. You should:")
    print("   1. Edit the generated file to add real content")
    print("   2. Or use a larger LLM to generate high-quality examples")
    print("   3. Or manually curate examples from textbooks/courses")


if __name__ == "__main__":
    main()

