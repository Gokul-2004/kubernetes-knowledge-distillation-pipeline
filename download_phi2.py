#!/usr/bin/env python3
"""
Robust Phi-2 model downloader with retry logic and progress tracking
"""
import os
import time
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_phi2_with_retry(max_retries=5, retry_delay=10):
    """Download Phi-2 model with retry logic"""
    model_name = "microsoft/Phi-2"
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to download {model_name}")
            
            # Download model files
            logger.info("Downloading model files...")
            model_path = snapshot_download(
                repo_id=model_name,
                local_dir="./models/phi2",
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info(f"Model downloaded successfully to: {model_path}")
            
            # Test loading the model
            logger.info("Testing model loading...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully!")
            logger.info(f"Model device: {next(model.parameters()).device}")
            logger.info(f"Model dtype: {next(model.parameters()).dtype}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("All download attempts failed")
                raise

if __name__ == "__main__":
    try:
        # Create models directory
        os.makedirs("./models", exist_ok=True)
        
        # Download Phi-2
        model_path = download_phi2_with_retry()
        print(f"\n✅ Phi-2 model successfully downloaded to: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Failed to download Phi-2: {str(e)}")
        exit(1) 