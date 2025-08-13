import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pipeline.config import ALPACA_DATA_PATH
from pipeline.data_utils import load_alpaca_qa, save_alpaca_qa
from loguru import logger
import os

def load_student_model_cpu():
    """Load a small model that works efficiently on CPU"""
    try:
        # Use a smaller model that works well on CPU
        model_name = "microsoft/DialoGPT-small"  # Only 117M parameters
        
        # Load tokenizer and model for CPU
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"Successfully loaded {model_name} for CPU inference")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load student model: {e}")
        return None, None

def generate_student_answers_cpu():
    """Generate answers for questions using CPU-only model"""
    model, tokenizer = load_student_model_cpu()
    if not model or not tokenizer:
        return []
    
    qa_data = load_alpaca_qa()
    if not qa_data:
        logger.error("No question data found")
        return []
    
    logger.info(f"Generating answers for {len(qa_data)} questions using CPU")
    
    for i, item in enumerate(qa_data):
        question = item["instruction"]
        
        # Create simple prompt for DialoGPT
        prompt = f"Question: {question}\nAnswer:"
        
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            
            # Generate answer on CPU
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,  # Short answers for speed
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode answer
            answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            # Update the data
            qa_data[i]["output"] = answer
            
            logger.info(f"Generated answer for question {i+1}/{len(qa_data)}")
            
        except Exception as e:
            logger.error(f"Error generating answer for question {i+1}: {e}")
            qa_data[i]["output"] = "Error generating answer"
    
    # Save updated data
    save_alpaca_qa(qa_data)
    logger.info("Student answers generated and saved")
    return qa_data

if __name__ == "__main__":
    generate_student_answers_cpu() 