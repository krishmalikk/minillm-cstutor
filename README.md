# 🎓 CS Tutor LLM

A **fully local, offline** Large Language Model for teaching computer science concepts. Build, train, and run your own AI tutor that explains CS topics, generates practice problems, runs quizzes, and grades answers.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 📚 **Concept Explanations** - Clear, level-appropriate explanations of CS topics
- 💪 **Practice Problems** - Generate custom problems with hints and solutions
- 📝 **Interactive Quizzes** - Test your knowledge with generated quiz questions
- ✅ **Answer Grading** - Get detailed feedback on your answers
- 🔒 **Fully Offline** - Runs entirely on your local machine
- 🚀 **Efficient** - Supports 4-bit/8-bit quantization for fast inference

## 🏗️ Architecture

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

## 📦 Project Structure

```
minillm/
├── configs/                    # Configuration files
│   ├── model_configs.yaml      # Model architecture configs
│   ├── training_config.yaml    # Training hyperparameters
│   └── quantization_config.yaml
├── data/
│   └── examples/               # Example training data
│       ├── concept_explanations.jsonl
│       ├── practice_problems.jsonl
│       ├── quizzes.jsonl
│       └── grading.jsonl
├── scripts/
│   ├── train.py               # Full training script
│   ├── finetune_lora.py       # LoRA fine-tuning
│   └── quantize.py            # Quantization script
├── src/
│   ├── model/                 # Model architecture
│   │   ├── config.py          # Model configuration
│   │   ├── layers.py          # Transformer layers
│   │   ├── model.py           # Main model class
│   │   └── tokenizer.py       # BPE tokenizer
│   ├── data/                  # Data processing
│   │   ├── schema.py          # Dataset schema
│   │   └── dataset.py         # PyTorch datasets
│   ├── training/              # Training pipeline
│   │   └── trainer.py         # Trainer class
│   ├── lora/                  # LoRA fine-tuning
│   │   └── lora.py            # LoRA implementation
│   ├── quantization/          # Quantization
│   │   └── quantize.py        # INT8/INT4/GGUF
│   ├── inference/             # Inference engine
│   │   └── engine.py          # High-level API
│   └── cli/                   # CLI application
│       └── app.py             # Typer CLI
├── tutor.py                   # CLI entry point
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cs-tutor-llm.git
cd cs-tutor-llm

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

## 🧠 Training Your Own Model

### 1. Prepare Dataset

Create JSONL files with instruction-tuning examples:

```json
{
  "instruction": "Explain what a binary search tree is.",
  "input": "",
  "output": "A binary search tree is a hierarchical data structure...",
  "task_type": "concept_explanation",
  "topic": "data_structures",
  "difficulty": "beginner"
}
```

### 2. Train the Model

```bash
# Train a 125M parameter model
python scripts/train.py \
    --model-size 125m \
    --data-dir data/examples \
    --output-dir outputs \
    --epochs 3 \
    --batch-size 4 \
    --gradient-accumulation 8

# For larger models (300M, 1B)
python scripts/train.py --model-size 300m ...
python scripts/train.py --model-size 1b ...
```

### 3. Fine-tune with LoRA

```bash
# Fine-tune on specific topics with LoRA
python scripts/finetune_lora.py \
    --base-model outputs/final_model \
    --data-dir data/my_custom_data \
    --lora-rank 16 \
    --lora-alpha 32 \
    --output-dir outputs/lora_finetuned
```

### 4. Quantize for Efficiency

```bash
# 8-bit quantization
python scripts/quantize.py \
    --model-path outputs/final_model \
    --output-dir outputs/quantized \
    --quant-type int8

# 4-bit quantization
python scripts/quantize.py --quant-type nf4

# Export to GGUF (for llama.cpp)
python scripts/quantize.py --export-gguf --gguf-quant q4_0
```

## 📊 Model Configurations

| Size | Params | Hidden | Layers | Heads | VRAM (FP16) | VRAM (4-bit) |
|------|--------|--------|--------|-------|-------------|--------------|
| 125M | 125M   | 768    | 12     | 12    | ~500 MB     | ~100 MB      |
| 300M | 300M   | 1024   | 24     | 16    | ~1.2 GB     | ~250 MB      |
| 1B   | 1B     | 2048   | 24     | 32    | ~4 GB       | ~800 MB      |

## 📝 Dataset Format

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

## 🔧 API Reference

### CSTutorEngine

```python
from src.inference import CSTutorEngine

# Load model
engine = CSTutorEngine.from_pretrained("outputs/final_model")

# Explain a concept
explanation = engine.explain(
    "quicksort",
    level="intermediate",
    include_examples=True
)

# Generate practice problems
problems = engine.generate_problems(
    topic="graphs",
    difficulty="advanced",
    n=3
)

# Quiz
questions = engine.quiz_me(
    topic="sorting",
    num_questions=5
)

# Grade an answer
result = engine.grade_answer(
    question="Explain Big O notation",
    student_answer="Big O describes algorithm efficiency...",
    reference_solution="..."
)

# Interactive chat
response = engine.chat("How does binary search work?")
```

## ⚙️ Configuration

### Model Config (YAML)

```yaml
model_125m:
  name: "cs-tutor-125m"
  vocab_size: 32000
  max_seq_length: 2048
  hidden_size: 768
  intermediate_size: 3072
  num_hidden_layers: 12
  num_attention_heads: 12
  position_embedding_type: "rotary"
  hidden_activation: "silu"
```

### Training Config

```yaml
training:
  learning_rate: 3.0e-4
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  warmup_ratio: 0.03
  weight_decay: 0.1
  bf16: true
  gradient_checkpointing: true
```

## 🏃 Performance Tips

1. **Use Quantization**: 4-bit quantization reduces memory by 4x with minimal quality loss
2. **Enable Flash Attention**: Set `use_flash_attention: true` for faster training
3. **Gradient Checkpointing**: Trades compute for memory on larger models
4. **LoRA Fine-tuning**: Fine-tune with only ~1% of parameters

## 📈 Example Outputs

### Concept Explanation

```
tutor explain "binary search" --level beginner

📚 Binary Search

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
```python
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
```

### Quiz

```
tutor quiz --topic sorting --questions 1

📝 Question 1/1

What is the average time complexity of Merge Sort?

  A. O(n)
  B. O(n log n)
  C. O(n²)
  D. O(log n)

Your answer (A/B/C/D): B

✓ Correct!

Explanation: Merge Sort always divides the array in half (log n levels)
and at each level, all n elements are processed during the merge step.
This gives O(n × log n) = O(n log n).
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by LLaMA, GPT-2, and modern transformer architectures
- Built with PyTorch, Typer, and Rich
- Training techniques from "Scaling Language Models" research

---

Made with ❤️ for CS education

