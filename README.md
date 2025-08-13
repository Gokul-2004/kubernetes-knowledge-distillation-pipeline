# 🤖 Kubernetes v1.28 Knowledge Distillation Pipeline

A complete vertical knowledge distillation pipeline that transfers knowledge from GPT-4o-mini (teacher) to Phi-2 (student) model, specifically designed for Kubernetes v1.28 documentation.

## 🎯 What This Project Does

This pipeline demonstrates **knowledge distillation** - a technique where a large, powerful model (GPT-4o-mini) teaches a smaller, efficient model (Phi-2) to perform better on specific tasks. In this case, we're teaching Phi-2 to answer Kubernetes v1.28 questions more accurately.

### The Process:
1. **Generate Questions** from Kubernetes v1.28 documentation
2. **Get Baseline Answers** from untrained Phi-2
3. **Get Reference Answers** from expert GPT-4o-mini
4. **Evaluate Performance** using RAGAS framework
5. **Fine-tune Phi-2** using QLoRA (4-bit quantization)
6. **Re-evaluate** to see improvement
7. **Compare Results** - baseline vs fine-tuned

## 🚀 Quick Setup Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/atheno-ai/data-prep.git
cd data-prep
git checkout final-repo
```

### Step 2: Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up OpenAI API Key
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```
**Important:** Replace `your_openai_api_key_here` with your actual OpenAI API key.

### Step 5: Download Phi-2 Model (First Time Only)
```bash
python download_phi2.py
```
This downloads the Phi-2 model (~5GB) to your local system.

### Step 6: Run the Pipeline
```bash
python run_ui.py
```

### Step 7: Open the Web Interface
Open your browser and go to: **http://127.0.0.1:7860**

## 🎮 Using the Web Interface

1. **Configure Settings:**
   - **Number of Questions**: Start with 1-2 for testing
   - **Number of Cycles**: Start with 1 for testing
   - **Select Topics**: Choose Kubernetes topics you're interested in

2. **Start the Pipeline:**
   - Click "🚀 Start Pipeline"
   - Watch the progress in real-time
   - Results will appear automatically

3. **View Results:**
   - See baseline vs fine-tuned performance
   - Compare answer quality
   - Track improvement metrics

## 📊 Understanding the Results

### Performance Metrics:
- **Answer Relevancy**: 0.0-1.0 (higher is better)
- **Improvement**: How much the model improved after fine-tuning
- **Success Rate**: Percentage of successful cycles

### Example Results:
```
Baseline Performance: 60.5% relevancy
Fine-tuned Performance: 88.2% relevancy
Improvement: +27.7%
```

## 🛠️ System Requirements

### Minimum Requirements:
- **RAM**: 8GB
- **Storage**: 10GB free space
- **GPU**: 4GB VRAM (or CPU-only mode)
- **Python**: 3.8 or higher

### Recommended:
- **RAM**: 16GB
- **GPU**: 6GB+ VRAM
- **Storage**: 20GB free space

## 🔧 Troubleshooting

### Common Issues and Solutions:

#### 1. "CUDA out of memory" Error
**Solution:**
- The pipeline automatically falls back to CPU mode
- Close other GPU applications
- Reduce number of questions in UI

#### 2. "OpenAI API Key Error"
**Solution:**
```bash
# Check if .env file exists
ls -la .env

# Create .env file if missing
echo "OPENAI_API_KEY=your_actual_key_here" > .env
```

#### 3. "Module not found" Errors
**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

#### 4. "Model download failed"
**Solution:**
```bash
# Check internet connection
# Try downloading again
python download_phi2.py
```

#### 5. "Port 7860 already in use"
**Solution:**
```bash
# Kill existing process
pkill -f "python run_ui.py"

# Or use different port
python run_ui.py --port 7861
```

## 📁 Project Structure

```
data-prep/
├── pipeline/                 # Core pipeline code
│   ├── config.py            # Configuration settings
│   ├── main_pipeline.py     # Main orchestrator
│   ├── gradio_ui.py         # Web interface
│   └── ...                  # Other modules
├── data/
│   └── k8s_notes.txt        # Kubernetes knowledge base
├── models/                  # Downloaded models
├── checkpoints/             # Training checkpoints
├── requirements.txt         # Python dependencies
├── run_ui.py               # UI launcher
└── README.md               # This file
```

## 🧪 Testing the Pipeline

### Quick Test (1 question, 1 cycle):
1. Open the web interface
2. Set "Number of Questions" to 1
3. Set "Number of Cycles" to 1
4. Click "🚀 Start Pipeline"
5. Wait for completion (~5-10 minutes)

### Full Test (10 questions, 2 cycles):
1. Set "Number of Questions" to 10
2. Set "Number of Cycles" to 2
3. Click "🚀 Start Pipeline"
4. Wait for completion (~30-60 minutes)


### What You'll See:
- Questions about Kubernetes v1.28 features
- Baseline answers (shorter, less detailed)
- Reference answers (comprehensive, expert-level)
- Fine-tuned answers (improved, more detailed)
- Performance metrics and comparisons

## 🔍 Advanced Usage

### Command Line Interface:
```bash
# Run pipeline without UI
python -m pipeline.main_pipeline

# Test individual components
python test_pipeline.py

# View results
python final_summary.py
```

### Customization:
Edit `pipeline/config.py` to modify:
- Training parameters
- Model settings
- Evaluation metrics

