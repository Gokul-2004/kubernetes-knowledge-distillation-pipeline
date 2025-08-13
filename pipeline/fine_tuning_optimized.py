#!/usr/bin/env python3
"""
Optimized fine-tuning module for minimal memory usage
Specifically designed for 1 question, 1 cycle scenarios
"""
import torch
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer
from datasets import Dataset
import os
import gc
from peft import LoraConfig

from .config import STUDENT_MODEL_NAME, QLORA_CONFIG, TRAINING_CONFIG

logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """Clear GPU memory aggressively"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        logger.info("GPU memory cleared")

def load_model_minimal_memory():
    """Load model with absolute minimal memory usage"""
    try:
        # Clear memory first
        clear_gpu_memory()
        
        # Ultra-conservative quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer first (use local cache)
        tokenizer = AutoTokenizer.from_pretrained(
            STUDENT_MODEL_NAME,
            trust_remote_code=True,
            padding_side="right",
            local_files_only=True  # Use local cache only
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with minimal memory (use local cache)
        model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            max_memory={0: "3GB", "cpu": "8GB"},  # Very conservative
            low_cpu_mem_usage=True,
            local_files_only=True  # Use local cache only
        )
        
        logger.info("Model loaded with minimal memory settings")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model with minimal memory: {str(e)}")
        raise

def prepare_training_data_minimal(questions, reference_answers):
    """Prepare training data with minimal processing"""
    try:
        # Create simple training examples
        training_data = []
        for question, answer in zip(questions, reference_answers):
            # Simple prompt format
            prompt = f"Question: {question}\nAnswer: {answer}"
            training_data.append({"text": prompt})
        
        # Create dataset
        dataset = Dataset.from_list(training_data)
        logger.info(f"Prepared {len(training_data)} training examples")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to prepare training data: {str(e)}")
        raise

def fine_tune_minimal_memory(questions, reference_answers, checkpoint_name="minimal_fine_tuning"):
    """Fine-tune with absolute minimal memory usage"""
    try:
        logger.info("Starting minimal memory fine-tuning...")
        
        # Clear memory
        clear_gpu_memory()
        
        # Load model and tokenizer
        model, tokenizer = load_model_minimal_memory()
        
        # Prepare data
        dataset = prepare_training_data_minimal(questions, reference_answers)
        
        # Ultra-minimal training config
        training_args = TrainingArguments(
            output_dir=f"checkpoints/{checkpoint_name}",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,  # Very high to simulate larger batch
            warmup_steps=2,  # Minimal warmup
            max_steps=10,  # Very few steps for 1 question
            learning_rate=1e-4,  # Lower learning rate
            fp16=True,
            logging_steps=1,
            save_steps=10,  # Save at the end
            save_strategy="steps",
            load_best_model_at_end=False,  # Disable to avoid evaluation requirement
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
        
        # Use a real LoraConfig object for peft_config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            peft_config=lora_config
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save adapter
        adapter_path = f"checkpoints/{checkpoint_name}_adapter"
        trainer.save_model(adapter_path)
        logger.info(f"Adapter saved to {adapter_path}")
        
        # Clear memory
        clear_gpu_memory()
        
        return {
            "success": True,
            "adapter_path": adapter_path,
            "training_steps": 10,
            "checkpoint_name": checkpoint_name
        }
        
    except Exception as e:
        logger.error(f"Minimal memory fine-tuning failed: {str(e)}")
        clear_gpu_memory()
        return {
            "success": False,
            "error": str(e),
            "checkpoint_name": checkpoint_name
        }

if __name__ == "__main__":
    # Test minimal fine-tuning
    logging.basicConfig(level=logging.INFO)
    
    test_questions = ["What is Kubernetes?"]
    test_answers = ["Kubernetes is an open-source container orchestration platform."]
    
    result = fine_tune_minimal_memory(test_questions, test_answers)
    print(f"Fine-tuning result: {result}") 