import os
from dotenv import load_dotenv

load_dotenv()

# Model and API config
TEACHER_MODEL = "gpt-4o-mini"  # OpenAI API
STUDENT_MODEL_NAME = "microsoft/Phi-2"  # Phi-2 with QLoRA for minimal compute

# Paths
NOTES_PATH = "data/k8s_notes.txt"
ALPACA_DATA_PATH = "data/alpaca_qa.json"
CHECKPOINT_DIR = "checkpoints/"
MODEL_SAVE_DIR = "models/"
ADAPTER_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "phi2_adapter")
LOG_PATH = os.path.join(CHECKPOINT_DIR, "distill.log")

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Training
BATCH_SIZE = 1
NUM_EPOCHS = 1
QUANTIZATION = 4  # bits
EVAL_BATCH_SIZE = 4

# UI
DEFAULT_NUM_QUESTIONS = 10
DEFAULT_NUM_CYCLES = 2
MAX_QUESTIONS = 50
MAX_CYCLES = 10

# Kubernetes Topics for Question Generation
KUBERNETES_TOPICS = [
    "Kubernetes Architecture",
    "Pods and Containers",
    "Services and Networking",
    "Storage and Volumes",
    "ConfigMaps and Secrets",
    "Deployments and ReplicaSets",
    "StatefulSets and DaemonSets",
    "Jobs and CronJobs",
    "Namespaces and RBAC",
    "Ingress and Load Balancing",
    "Helm Charts and Package Management",
    "Monitoring and Logging",
    "Security and Policies",
    "Resource Management",
    "Scaling and Autoscaling",
    "Backup and Disaster Recovery",
    "Multi-cluster Management",
    "Service Mesh (Istio)",
    "Kubernetes API",
    "Troubleshooting and Debugging"
]

# RAGAS
RAGAS_METRIC = "answer_relevancy"

# QLoRA Configuration
QLORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Training Configuration
TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,  # Increased for memory efficiency
    "warmup_steps": 5,  # Reduced for faster training
    "max_steps": 50,  # Reduced for testing
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 5,
    "save_steps": 25,
    "eval_steps": 25,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "load_best_model_at_end": True,
    "report_to": None,  # Disable wandb
    "remove_unused_columns": False,
    "dataloader_pin_memory": False,  # Reduce memory usage
    "gradient_checkpointing": True,  # Enable gradient checkpointing
} 