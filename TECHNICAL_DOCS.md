# Kubernetes v1.28 Knowledge Distillation Pipeline - Technical Documentation

## Executive Summary

This document provides a comprehensive technical overview of the Knowledge Distillation Pipeline designed for Kubernetes v1.28 documentation. The system implements vertical knowledge distillation where GPT-4o-mini (teacher) transfers knowledge to Phi-2 (student) through fine-tuning, improving domain-specific performance.

## 1. System Architecture

### Core Components
- **Teacher Model**: GPT-4o-mini via OpenAI API
- **Student Model**: Microsoft Phi-2 (2.7B parameters)
- **Evaluation**: RAGAS framework with answer_relevancy metric
- **Optimization**: 4-bit QLoRA quantization

### Pipeline Stages
1. Question Generation (GPT-4o-mini)
2. Baseline Answer Generation (Untrained Phi-2)
3. Reference Answer Generation (GPT-4o-mini)
4. Baseline Evaluation (RAGAS)
5. Fine-tuning (QLoRA)
6. Fine-tuned Evaluation (RAGAS)
7. Performance Comparison

## 2. Technical Implementation

### Knowledge Base
- **Source**: Kubernetes v1.28 documentation
- **Content**: 1,754 characters of structured knowledge
- **Topics**: 20 Kubernetes domains (Pods, Services, Storage, etc.)

### Question Generation
```
System Prompt: "You are an expert Kubernetes question generator specializing in Kubernetes v1.28."
Task: Generate questions based on selected topics
Context: Kubernetes v1.28 knowledge base
Output: Domain-specific, practical questions
```

### Answer Generation Parameters
- **max_new_tokens**: 500
- **temperature**: 0.8
- **top_p**: 0.95
- **top_k**: 50
- **repetition_penalty**: 1.1
- **max_time**: 300 seconds

### Fine-tuning Configuration (QLoRA)
- **Quantization**: 4-bit precision
- **LoRA Rank**: 16
- **Target Modules**: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
- **Learning Rate**: 2e-4
- **Batch Size**: 1 with gradient accumulation
- **Training Steps**: 100

### Training Data Format
```
<|im_start|>system
You are a Kubernetes expert. Answer the following question about Kubernetes v1.28.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{reference_answer}
<|im_end|>
```

## 3. Memory Management

### GPU Memory Optimization
- **Target**: 4GB VRAM systems
- **Strategy**: 4-bit quantization + aggressive memory clearing
- **Fallback**: CPU mode when GPU memory insufficient

### Optimization Techniques
- Gradient checkpointing
- Mixed precision training (FP16)
- Automatic device mapping
- Memory limits (3GB max allocation)

## 4. Evaluation Framework

### RAGAS Answer Relevancy
- **Method**: Semantic similarity between student and reference answers
- **Process**: Embedding generation → Cosine similarity → Score normalization
- **Scale**: 0.0-1.0 (higher is better)

### Performance Metrics
- **Baseline Score**: Untrained model performance
- **Fine-tuned Score**: Trained model performance
- **Improvement**: Absolute and percentage improvement
- **Success Rate**: Percentage of successful cycles

## 5. Pipeline Workflow

### Cycle Structure
1. **Initialization**: Load config, initialize models, setup logging
2. **Question Generation**: Select topics, generate questions, save checkpoint
3. **Baseline Evaluation**: Generate answers, create references, evaluate
4. **Fine-tuning**: Prepare dataset, configure QLoRA, execute training
5. **Fine-tuned Evaluation**: Load model, generate answers, evaluate
6. **Results Analysis**: Calculate metrics, generate report, save results

### Checkpointing System
- **Format**: JSON-based checkpoint files
- **Frequency**: After each major step
- **Recovery**: Resume from any checkpoint
- **Structure**: Questions, answers, metrics, improvement data

## 6. User Interface

### Gradio Web Interface
- **Framework**: Gradio for web-based interaction
- **Features**: Configuration sliders, real-time progress, results visualization
- **Options**: Question count (1-50), cycle count (1-10), topic selection

## 7. Performance Analysis

### Expected Results
- **Baseline Performance**: 60-80% answer relevancy
- **Fine-tuned Performance**: 80-95% answer relevancy
- **Typical Improvement**: 10-30% increase
- **Training Time**: 5-30 minutes per cycle

### Success Metrics
- Answer quality (RAGAS relevancy)
- Consistency across cycles
- Memory efficiency
- Completion reliability

## 8. Technical Requirements

### System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: 4GB VRAM minimum, 6GB+ recommended
- **Storage**: 10GB free space
- **Python**: 3.8+

### Key Dependencies
- PyTorch, Transformers, BitsAndBytes
- TRL (Training and RL library)
- RAGAS (Evaluation framework)
- Gradio (Web interface)
- OpenAI (API client)

## 9. Deployment Process

### Installation Steps
1. Clone repository: `git clone https://github.com/atheno-ai/data-prep.git`
2. Create virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Set API key: `echo "OPENAI_API_KEY=your_key" > .env`
5. Download model: `python download_phi2.py`
6. Launch UI: `python run_ui.py`

### Usage Workflow
1. Configure parameters in web interface
2. Select Kubernetes topics
3. Start pipeline execution
4. Monitor real-time progress
5. Review results and metrics

## 10. Troubleshooting

### Common Issues
- **CUDA Memory Errors**: Automatic CPU fallback
- **API Rate Limits**: Built-in retry mechanisms
- **Model Loading Failures**: Robust error handling
- **Training Interruptions**: Checkpoint recovery

### Maintenance
- Regular dependency updates
- Model and knowledge base refreshes
- Performance monitoring
- Checkpoint and adapter backups

## 11. Future Enhancements

### Potential Improvements
- Multi-modal support (diagrams, code examples)
- Advanced RAGAS metrics
- Distributed training (multi-GPU)
- Further model compression

### Scalability
- Batch processing for larger datasets
- Cloud deployment integration
- RESTful API endpoints
- Advanced monitoring and analytics

## Conclusion

The Kubernetes v1.28 Knowledge Distillation Pipeline successfully demonstrates effective knowledge transfer from large to small models. Through 4-bit QLoRA optimization, comprehensive evaluation, and user-friendly interface, it provides a robust solution for domain-specific model improvement.

The modular architecture and detailed documentation make it suitable for both research and practical applications in knowledge distillation and domain-specific optimization.

---

**Document Version**: 1.0  
**Last Updated**: July 2024  
**Technical Documentation for Kubernetes v1.28 Knowledge Distillation Pipeline** 