import torch
import gc
import os
import subprocess
import sys
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from loguru import logger
import logging
import signal
from contextlib import contextmanager

def clear_gpu_memory():
    """Clear GPU memory aggressively"""
    import gc
    import torch
    
    # Clear PyTorch cache multiple times
    if torch.cuda.is_available():
        for _ in range(5):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # Multiple garbage collection passes
    for _ in range(15):
        gc.collect()
    
    # Force CUDA memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Try to reset CUDA memory allocator
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            # Force memory defragmentation
            torch.cuda.empty_cache()
        except:
            pass
    
    logger.info("GPU memory cleared aggressively")

def restart_process_if_needed():
    """Restart the current process to clear GPU memory completely"""
    logger.warning("GPU memory fragmented. Restarting process to clear memory...")
    # Save current state
    os.environ['RESTART_AFTER_FINE_TUNING'] = '1'
    # Restart the process
    subprocess.Popen([sys.executable] + sys.argv)
    sys.exit(0)

def force_gpu_reset():
    """Force a complete GPU reset by restarting the process"""
    logger.warning("Forcing GPU reset due to severe memory fragmentation...")
    os.environ['FORCE_GPU_RESET'] = '1'
    subprocess.Popen([sys.executable] + sys.argv)
    sys.exit(0)

def load_phi2_minimal(adapter_path=None, max_retries=5):
    """Load Phi-2 model with minimal memory settings and retry logic"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from accelerate import infer_auto_device_map, init_empty_weights
    import gc
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading Phi-2 (attempt {attempt + 1}/{max_retries})")
            
            # Clear memory before loading
            clear_gpu_memory()
            
            # Wait longer for memory to settle
            time.sleep(5)
            
            if torch.cuda.is_available():
                logger.info("Using CUDA for Phi-2")
                device_map = "auto"
                torch_dtype = torch.float16
            else:
                logger.info("Using CPU for Phi-2")
                device_map = "cpu"
                torch_dtype = torch.float32
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/Phi-2",
                trust_remote_code=True,
                padding_side="left"
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with even more minimal memory settings
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-2",
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                max_memory={0: "3.0GB", "cpu": "8GB"},  # Reduced from 3.5GB
                offload_folder="offload",
                offload_state_dict=True,
                low_cpu_mem_usage=True
            )
            
            # Load adapter if provided
            if adapter_path and os.path.exists(adapter_path):
                logger.info(f"Loading adapter from {adapter_path}")
                model = PeftModel.from_pretrained(model, adapter_path)
                model.eval()
            
            logger.info("Successfully loaded Phi-2 with minimal settings on cuda")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load Phi-2: {str(e)}")
            
            # Clear memory and try again
            clear_gpu_memory()
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Increasing wait time
                logger.info(f"Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                # If this is after fine-tuning and we still can't load, restart process
                if "CUDA out of memory" in str(e) and os.environ.get('AFTER_FINE_TUNING') == '1':
                    logger.warning("Severe GPU memory fragmentation after fine-tuning. Restarting process...")
                    force_gpu_reset()
                else:
                    # Last resort: try CPU mode
                    logger.warning("All GPU attempts failed. Trying CPU mode as last resort...")
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            "microsoft/Phi-2",
                            torch_dtype=torch.float32,
                            device_map="cpu",
                            trust_remote_code=True,
                            load_in_4bit=False,
                            low_cpu_mem_usage=True
                        )
                        logger.info("Successfully loaded Phi-2 on CPU")
                        return model, tokenizer
                    except Exception as cpu_error:
                        logger.error(f"CPU loading also failed: {str(cpu_error)}")
                        raise e

def find_latest_adapter():
    """Find the most recent fine-tuned adapter"""
    import glob
    import os
    
    try:
        # Look for adapter directories
        adapter_patterns = [
            "checkpoints/fine_tuning_cycle_*_adapter",
            "checkpoints/robust_fine_tuning_adapter"
        ]
        
        latest_adapter = None
        latest_time = 0
        
        for pattern in adapter_patterns:
            for adapter_path in glob.glob(pattern):
                if os.path.isdir(adapter_path):
                    mod_time = os.path.getmtime(adapter_path)
                    if mod_time > latest_time:
                        latest_time = mod_time
                        latest_adapter = adapter_path
        
        if latest_adapter:
            logger.info(f"Found latest adapter: {latest_adapter}")
            return latest_adapter
        else:
            logger.warning("No fine-tuned adapter found")
            return None
            
    except Exception as e:
        logger.error(f"Error finding latest adapter: {e}")
        return None

def load_fine_tuned_model(adapter_path):
    """Load the fine-tuned model with adapter"""
    try:
        logger.info(f"Loading fine-tuned model with adapter: {adapter_path}")
        
        # Clear memory before loading
        clear_gpu_memory()
        time.sleep(2)
        
        # Load base model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True,
            local_files_only=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Successfully loaded fine-tuned model with adapter")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load fine-tuned model: {e}")
        return None, None

def generate_student_answers(questions, adapter_path=None, max_retries=3, use_fine_tuned=False):
    """
    Generate answers using Phi-2 model with enhanced memory management and timeout
    """
    logger = logging.getLogger(__name__)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Generating Phi-2 answers for {len(questions)} questions (attempt {attempt + 1}/{max_retries})")
            
            # Load model - check for fine-tuned adapter
            if use_fine_tuned and adapter_path is None:
                # Look for the most recent fine-tuned adapter
                adapter_path = find_latest_adapter()
            
            model, tokenizer = load_phi2_minimal(adapter_path)
            
            if model is None or tokenizer is None:
                logger.error("Could not load Phi-2 model")
                return []
            
            answers = []
            
            for i, question in enumerate(questions):
                logger.info(f"Generating Phi-2 answer {i+1}/{len(questions)}")
                
                try:
                    # Prepare input with better prompt format
                    prompt = f"Please provide a detailed answer to the following question about Kubernetes:\n\nQuestion: {question}\n\nDetailed Answer:"
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    
                    # Move to same device as model
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Generate with better settings for longer, more detailed answers
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=500,  # Increased for more detailed answers
                            do_sample=True,
                            temperature=0.8,  # Slightly higher for more creative responses
                            top_p=0.95,
                            top_k=50,
                            repetition_penalty=1.1,  # Prevent repetitive text
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            early_stopping=False,  # Allow longer generation
                            max_time=300.0  # 5 minutes max generation time
                        )
                    
                    # Decode the generated text
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    answer = generated_text[len(prompt):].strip()
                    
                    if not answer:
                        answer = "I apologize, but I couldn't generate a proper answer for this question."
                    
                    answers.append(answer)
                    logger.info(f"Generated Phi-2 answer {i+1}/{len(questions)}")
                    
                except Exception as e:
                    logger.error(f"Error generating answer for question {i+1}: {e}")
                    answers.append(f"Error generating answer: {str(e)}")
            
            # Clear memory
            del model, tokenizer
            clear_gpu_memory()
            
            logger.info("Phi-2 answers generated")
            return answers
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
                clear_gpu_memory()
            else:
                logger.error("All retry attempts failed")
                return ["Error: Could not generate answers after multiple attempts"] * len(questions) 