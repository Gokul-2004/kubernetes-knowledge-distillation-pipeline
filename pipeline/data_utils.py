import os
import json
from loguru import logger
from pipeline.config import NOTES_PATH, ALPACA_DATA_PATH, CHECKPOINT_DIR

def load_notes():
    try:
        with open(NOTES_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load notes: {e}")
        return None

def save_alpaca_qa(data):
    try:
        with open(ALPACA_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save Alpaca Q&A: {e}")

def load_alpaca_qa():
    try:
        with open(ALPACA_DATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load Alpaca Q&A: {e}")
        return []

def save_checkpoint(data, name):
    """Save checkpoint data with a given name"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # Sanitize filename to avoid issues with special characters
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_')
    path = os.path.join(CHECKPOINT_DIR, f"{safe_name}.json")
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Checkpoint saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint {name}: {e}")

def load_checkpoint(name):
    """Load checkpoint data with a given name"""
    # Sanitize filename to avoid issues with special characters
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_')
    path = os.path.join(CHECKPOINT_DIR, f"{safe_name}.json")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Checkpoint loaded: {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load checkpoint {name}: {e}")
        return None 