import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pipeline.config import STUDENT_MODEL, ALPACA_DATA_PATH
from pipeline.data_utils import load_alpaca_qa, save_alpaca_qa
from loguru import logger
import os

def generate_mock_answers():
    """Generate mock answers for testing pipeline flow without large model download"""
    qa_data = load_alpaca_qa()
    if not qa_data:
        logger.error("No question data found")
        return []
    
    logger.info(f"Generating mock answers for {len(qa_data)} questions")
    
    mock_answers = [
        "Kubernetes v1.28 introduced expanded version skew policy allowing node components to be up to three minor versions behind control plane components.",
        "Recovery from non-graceful node shutdown is now generally available, improving failover for stateful workloads.",
        "CEL (Common Expression Language) allows more complex CRD validation without webhooks.",
        "ValidatingAdmissionPolicies graduated to beta, enabling in-process validation of API requests.",
        "Beta support for enabling swap space on Linux nodes allows controlled use of swap memory.",
        "Mixed version proxy allows API servers at different versions to proxy requests during upgrades.",
        "Control plane components source code was reorganized for better maintainability.",
        "CDI injection into containers is currently in alpha stage.",
        "New pod replacement policy for Jobs was introduced for better job management.",
        "CSI Migration deprecations were announced for certain in-tree plugins."
    ]
    
    for i, item in enumerate(qa_data):
        if i < len(mock_answers):
            qa_data[i]["output"] = mock_answers[i]
        else:
            qa_data[i]["output"] = "Mock answer for testing purposes."
    
    save_alpaca_qa(qa_data)
    logger.info("Mock student answers generated and saved")
    return qa_data

def load_student_model():
    """Load Phi-2 model with ultra-minimal 4-bit quantization for 6GB VRAM"""
    try:
        # Ultra-minimal 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        
        # Load tokenizer and model with minimal memory usage
        tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            offload_folder="offload"
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"Successfully loaded {STUDENT_MODEL} with ultra-minimal 4-bit quantization")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load student model: {e}")
        return None, None

def generate_student_answers():
    """Generate answers for questions using Phi-2 student model"""
    model, tokenizer = load_student_model()
    if not model or not tokenizer:
        return []
    
    qa_data = load_alpaca_qa()
    if not qa_data:
        logger.error("No question data found")
        return []
    
    logger.info(f"Generating answers for {len(qa_data)} questions")
    
    for i, item in enumerate(qa_data):
        question = item["instruction"]
        
        # Create prompt for Phi-2
        prompt = f"Question: {question}\n\nAnswer:"
        
        try:
            # Tokenize input with minimal memory usage
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate answer with minimal compute
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,  # Reduced for minimal compute
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    early_stopping=True
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
    # Use real Phi-2 model with minimal compute
    generate_student_answers() 