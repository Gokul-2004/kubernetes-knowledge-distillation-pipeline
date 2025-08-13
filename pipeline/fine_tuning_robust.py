#!/usr/bin/env python3
"""
Robust fine-tuning module that handles CUDA environment issues
Falls back to CPU if GPU is not available
"""
import torch
import logging
import os
import gc
import json
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

from .config import STUDENT_MODEL_NAME

logger = logging.getLogger(__name__)

def reset_cuda_environment():
    """Reset CUDA environment to fix initialization issues"""
    try:
        # Clear all CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Reset CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            logger.info("CUDA environment reset successfully")
        else:
            logger.info("CUDA not available, using CPU")
            
    except Exception as e:
        logger.warning(f"CUDA reset failed: {e}, continuing with CPU")

def load_model_robust():
    """Load model with robust error handling and fallback"""
    try:
        # Reset CUDA environment first
        reset_cuda_environment()
        
        # Try GPU first
        if torch.cuda.is_available():
            try:
                logger.info("Attempting GPU loading...")
                
                # Conservative quantization for GPU
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    STUDENT_MODEL_NAME,
                    trust_remote_code=True,
                    padding_side="right",
                    local_files_only=True
                )
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model on GPU
                model = AutoModelForCausalLM.from_pretrained(
                    STUDENT_MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    max_memory={0: "4GB", "cpu": "8GB"},
                    low_cpu_mem_usage=True,
                    local_files_only=True
                )
                
                logger.info("Model loaded successfully on GPU")
                return model, tokenizer, "gpu"
                
            except Exception as gpu_error:
                logger.warning(f"GPU loading failed: {gpu_error}, falling back to CPU")
                reset_cuda_environment()
        
        # Fallback to CPU
        logger.info("Loading model on CPU...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            STUDENT_MODEL_NAME,
            trust_remote_code=True,
            padding_side="right",
            local_files_only=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model on CPU with minimal memory
        model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        
        logger.info("Model loaded successfully on CPU")
        return model, tokenizer, "cpu"
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def prepare_training_data(questions, reference_answers):
    """Prepare training data in Alpaca format"""
    try:
        training_data = []
        for question, answer in zip(questions, reference_answers):
            # Format for SFTTrainer with text field
            formatted_text = format_prompt(question, "", answer)
            training_example = {
                "text": formatted_text
            }
            training_data.append(training_example)
        
        # Convert to dataset
        dataset = Dataset.from_list(training_data)
        logger.info(f"Prepared {len(training_data)} training examples")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to prepare training data: {str(e)}")
        raise

def format_prompt(instruction, input_text, output):
    """Format prompt in Alpaca style"""
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

def fine_tune_robust(questions, reference_answers, checkpoint_name="robust_fine_tuning"):
    """Robust fine-tuning with GPU/CPU fallback"""
    try:
        logger.info("Starting robust fine-tuning...")
        
        # Load model and tokenizer
        model, tokenizer, device_type = load_model_robust()
        
        # Prepare training data
        dataset = prepare_training_data(questions, reference_answers)
        
        # Configure training based on device
        if device_type == "gpu":
            # GPU training settings
            training_args = TrainingArguments(
                output_dir=f"checkpoints/{checkpoint_name}",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                warmup_steps=2,
                max_steps=15,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=1,
                save_steps=15,
                save_strategy="steps",
                load_best_model_at_end=False,
                report_to=None,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                gradient_checkpointing=True,
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                weight_decay=0.01,
                max_grad_norm=1.0,
                seed=42,
            )
        else:
            # CPU training settings (more conservative)
            training_args = TrainingArguments(
                output_dir=f"checkpoints/{checkpoint_name}",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=1,
                max_steps=10,
                learning_rate=1e-4,
                fp16=False,  # No fp16 on CPU
                logging_steps=1,
                save_steps=10,
                save_strategy="steps",
                load_best_model_at_end=False,
                report_to=None,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                gradient_checkpointing=False,  # No gradient checkpointing on CPU
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                weight_decay=0.01,
                max_grad_norm=1.0,
                seed=42,
            )
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Create trainer (compatible with trl 0.19.1)
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
        )
        
        # Train
        logger.info(f"Starting training on {device_type.upper()}...")
        trainer.train()
        
        # Save adapter
        adapter_path = f"checkpoints/{checkpoint_name}_adapter"
        trainer.save_model(adapter_path)
        logger.info(f"Adapter saved to {adapter_path}")
        
        # Save training info
        training_info = {
            "device_type": device_type,
            "training_steps": training_args.max_steps,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps
        }
        
        with open(f"checkpoints/{checkpoint_name}_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        # Clear memory
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "success": True,
            "adapter_path": adapter_path,
            "device_type": device_type,
            "training_steps": training_args.max_steps,
            "checkpoint_name": checkpoint_name
        }
        
    except Exception as e:
        logger.error(f"Robust fine-tuning failed: {str(e)}")
        # Clear memory on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "success": False,
            "error": str(e),
            "checkpoint_name": checkpoint_name
        }

if __name__ == "__main__":
    # Test robust fine-tuning
    logging.basicConfig(level=logging.INFO)
    
    test_questions = ["What is Kubernetes?"]
    test_answers = ["Kubernetes is an open-source container orchestration platform."]
    
    result = fine_tune_robust(test_questions, test_answers)
    print(f"Fine-tuning result: {result}") 