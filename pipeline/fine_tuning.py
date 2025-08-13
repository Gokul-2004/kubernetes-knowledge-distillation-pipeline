#!/usr/bin/env python3
"""
Fine-tuning module using QLoRA with trl.SFTTrainer
Performs adapter-based fine-tuning on student model using generated Q&A data
"""
import json
import logging
import os
from typing import List, Dict, Any, Optional
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import datasets
from datasets import Dataset

from .config import (
    STUDENT_MODEL_NAME, QLORA_CONFIG, TRAINING_CONFIG,
    MODEL_SAVE_DIR, ADAPTER_SAVE_DIR
)
from .data_utils import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)

class TrainingCallback(TrainerCallback):
    """Custom callback for training progress tracking"""
    
    def __init__(self, checkpoint_name: str):
        self.checkpoint_name = checkpoint_name
        self.step_count = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        if self.step_count % 10 == 0:  # Log every 10 steps
            logger.info(f"Training step {self.step_count}, Loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}")

def prepare_training_data(questions: List[str], reference_answers: List[str]) -> Dataset:
    """Prepare training data in Alpaca format"""
    training_data = []
    
    for question, answer in zip(questions, reference_answers):
        # Format in Alpaca style
        formatted_data = {
            "instruction": question,
            "input": "",
            "output": answer
        }
        training_data.append(formatted_data)
    
    dataset = Dataset.from_list(training_data)
    return dataset

def load_student_model_for_training(model_path: Optional[str] = None) -> tuple:
    """
    Load student model with QLoRA configuration for training
    
    Args:
        model_path: Path to model (if None, downloads from HuggingFace)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        logger.info(f"Loading student model: {STUDENT_MODEL_NAME}")
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load model with quantization for limited VRAM
        model = AutoModelForCausalLM.from_pretrained(
            model_path or STUDENT_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            max_memory={0: "4GB", "cpu": "8GB"} if torch.cuda.is_available() else None
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path or STUDENT_MODEL_NAME,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA configuration
        lora_config = LoraConfig(**QLORA_CONFIG)
        model = get_peft_model(model, lora_config)
        
        logger.info("Student model loaded successfully with QLoRA configuration")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load student model: {str(e)}")
        raise

def fine_tune_student_model(questions: List[str], reference_answers: List[str], 
                          checkpoint_name: str = "fine_tuning") -> Dict[str, Any]:
    """
    Fine-tune student model using QLoRA with SFTTrainer
    
    Args:
        questions: List of questions for training
        reference_answers: List of reference answers for training
        checkpoint_name: Name for checkpointing results
    
    Returns:
        Dictionary containing training results
    """
    try:
        logger.info(f"Starting fine-tuning with {len(questions)} training examples")
        
        # Prepare training data
        training_dataset = prepare_training_data(questions, reference_answers)
        logger.info(f"Training dataset prepared with {len(training_dataset)} examples")
        
        # Load model and tokenizer
        model, tokenizer = load_student_model_for_training()
        
        # Create training arguments
        training_args = TrainingArguments(
            **TRAINING_CONFIG,
            output_dir=MODEL_SAVE_DIR,
            logging_dir=f"{MODEL_SAVE_DIR}/logs"
        )
        
        # Create SFTTrainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=training_dataset,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=512,
            packing=False,
            dataset_text_field="instruction"  # Use instruction field for training
        )
        
        # Add custom callback
        callback = TrainingCallback(checkpoint_name)
        trainer.add_callback(callback)
        
        # Start training
        logger.info("Starting training...")
        training_result = trainer.train()
        
        # Save the fine-tuned model
        logger.info("Saving fine-tuned model...")
        os.makedirs(ADAPTER_SAVE_DIR, exist_ok=True)
        
        # Save adapter weights
        model.save_pretrained(ADAPTER_SAVE_DIR, safe_serialization=True)
        tokenizer.save_pretrained(ADAPTER_SAVE_DIR)
        
        # Save adapter config
        adapter_config_path = os.path.join(ADAPTER_SAVE_DIR, "adapter_config.json")
        with open(adapter_config_path, 'w') as f:
            json.dump(QLORA_CONFIG, f, indent=2)
        
        # Prepare results
        results = {
            "training_complete": True,
            "num_training_examples": len(questions),
            "final_loss": training_result.training_loss,
            "adapter_save_path": ADAPTER_SAVE_DIR,
            "training_steps": training_result.global_step
        }
        
        # Save checkpoint
        checkpoint_data = {
            "results": results,
            "questions": questions,
            "reference_answers": reference_answers,
            "training_dataset": training_dataset.to_dict()
        }
        
        save_checkpoint(checkpoint_data, checkpoint_name)
        logger.info(f"Fine-tuning completed. Model saved to: {ADAPTER_SAVE_DIR}")
        
        return results
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}")
        
        # Save error checkpoint
        try:
            error_results = {
                "training_complete": False,
                "num_training_examples": len(questions),
                "error": str(e)
            }
            
            checkpoint_data = {
                "results": error_results,
                "questions": questions,
                "reference_answers": reference_answers,
                "error": str(e)
            }
            
            save_checkpoint(checkpoint_data, f"{checkpoint_name}_error")
            logger.info("Error checkpoint saved")
            
        except Exception as save_error:
            logger.error(f"Failed to save error checkpoint: {str(save_error)}")
        
        raise

def load_fine_tuned_model(adapter_path: str = None) -> tuple:
    """
    Load fine-tuned model with adapter weights
    
    Args:
        adapter_path: Path to adapter weights (if None, uses default)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        adapter_path = adapter_path or ADAPTER_SAVE_DIR
        
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        
        logger.info(f"Loading fine-tuned model from: {adapter_path}")
        
        # Load base model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Load adapter weights
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            STUDENT_MODEL_NAME,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Fine-tuned model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load fine-tuned model: {str(e)}")
        raise

def load_fine_tuning_results(checkpoint_name: str) -> Dict[str, Any]:
    """Load fine-tuning results from checkpoint"""
    try:
        checkpoint_data = load_checkpoint(checkpoint_name)
        return checkpoint_data
    except Exception as e:
        logger.error(f"Failed to load fine-tuning results: {str(e)}")
        return None

if __name__ == "__main__":
    # Test fine-tuning with sample data
    logging.basicConfig(level=logging.INFO)
    
    test_questions = [
        "What is a Kubernetes Pod?",
        "How do you scale a deployment?",
        "What is a Service in Kubernetes?"
    ]
    
    test_reference_answers = [
        "A Pod is the smallest and simplest unit in the Kubernetes object model that you create or deploy. A Pod represents a running process on your cluster and can contain one or more containers.",
        "You can scale a deployment using kubectl scale deployment <deployment-name> --replicas=<number> or by editing the deployment YAML file.",
        "A Service is an abstract way to expose an application running on a set of Pods as a network service."
    ]
    
    try:
        results = fine_tune_student_model(test_questions, test_reference_answers, "test_fine_tuning")
        print(f"Test fine-tuning successful: {results}")
    except Exception as e:
        print(f"Test fine-tuning failed: {str(e)}") 