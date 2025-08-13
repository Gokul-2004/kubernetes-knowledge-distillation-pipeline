#!/usr/bin/env python3
"""
Evaluation module using RAGAS framework
Evaluates student answers against reference answers using answer_relevancy metric
"""
import json
import logging
from typing import List, Dict, Any
from ragas import evaluate
from ragas.metrics import answer_relevancy
from datasets import Dataset
import pandas as pd

from .config import EVAL_BATCH_SIZE
from .data_utils import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)

def prepare_evaluation_dataset(questions: List[str], student_answers: List[str], 
                             reference_answers: List[str]) -> Dataset:
    """Prepare dataset for RAGAS evaluation"""
    data = {
        "question": questions,
        "answer": student_answers,
        "ground_truth": reference_answers
    }
    
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    return dataset

def evaluate_answers(questions: List[str], student_answers: List[str], 
                   reference_answers: List[str], checkpoint_name: str = "evaluation") -> Dict[str, Any]:
    """
    Evaluate student answers against reference answers using RAGAS
    
    Args:
        questions: List of questions
        student_answers: List of student-generated answers
        reference_answers: List of reference/teacher answers
        checkpoint_name: Name for checkpointing results
    
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        logger.info(f"Evaluating {len(questions)} question-answer pairs")
        
        # Prepare dataset for RAGAS
        dataset = prepare_evaluation_dataset(questions, student_answers, reference_answers)
        
        # Run evaluation with answer_relevancy metric
        logger.info("Running RAGAS evaluation with answer_relevancy metric...")
        results = evaluate(
            dataset,
            metrics=[answer_relevancy],
            batch_size=EVAL_BATCH_SIZE
        )
        
        # Extract metrics - handle both list and float results
        answer_relevancy_value = results["answer_relevancy"]
        if isinstance(answer_relevancy_value, list):
            answer_relevancy_value = float(answer_relevancy_value[0]) if answer_relevancy_value else 0.0
        else:
            answer_relevancy_value = float(answer_relevancy_value)
        
        metrics = {
            "answer_relevancy": answer_relevancy_value,
            "num_questions": len(questions),
            "evaluation_complete": True
        }
        
        logger.info(f"Evaluation complete. Answer Relevancy: {metrics['answer_relevancy']:.4f}")
        
        # Save evaluation results
        checkpoint_data = {
            "metrics": metrics,
            "questions": questions,
            "student_answers": student_answers,
            "reference_answers": reference_answers,
            "dataset": dataset.to_dict()
        }
        
        save_checkpoint(checkpoint_data, checkpoint_name)
        logger.info(f"Evaluation results saved to checkpoint: {checkpoint_name}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        
        # Try to save partial results
        try:
            partial_metrics = {
                "answer_relevancy": 0.0,
                "num_questions": len(questions),
                "evaluation_complete": False,
                "error": str(e)
            }
            
            checkpoint_data = {
                "metrics": partial_metrics,
                "questions": questions,
                "student_answers": student_answers,
                "reference_answers": reference_answers,
                "error": str(e)
            }
            
            save_checkpoint(checkpoint_data, f"{checkpoint_name}_error")
            logger.info("Partial results saved to error checkpoint")
            
        except Exception as save_error:
            logger.error(f"Failed to save error checkpoint: {str(save_error)}")
        
        raise

def load_evaluation_results(checkpoint_name: str) -> Dict[str, Any]:
    """Load evaluation results from checkpoint"""
    try:
        checkpoint_data = load_checkpoint(checkpoint_name)
        return checkpoint_data
    except Exception as e:
        logger.error(f"Failed to load evaluation results: {str(e)}")
        return None

def compare_evaluations(baseline_metrics: Dict[str, Any], 
                       improved_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two evaluation results"""
    comparison = {
        "baseline_answer_relevancy": baseline_metrics.get("answer_relevancy", 0.0),
        "improved_answer_relevancy": improved_metrics.get("answer_relevancy", 0.0),
        "improvement": improved_metrics.get("answer_relevancy", 0.0) - baseline_metrics.get("answer_relevancy", 0.0),
        "improvement_percentage": 0.0
    }
    
    if baseline_metrics.get("answer_relevancy", 0.0) > 0:
        comparison["improvement_percentage"] = (
            comparison["improvement"] / baseline_metrics["answer_relevancy"]
        ) * 100
    
    return comparison

if __name__ == "__main__":
    # Test evaluation with sample data
    logging.basicConfig(level=logging.INFO)
    
    test_questions = [
        "What is a Kubernetes Pod?",
        "How do you scale a deployment?"
    ]
    
    test_student_answers = [
        "A Pod is the smallest deployable unit in Kubernetes that contains one or more containers.",
        "You can scale a deployment using kubectl scale command."
    ]
    
    test_reference_answers = [
        "A Pod is the smallest and simplest unit in the Kubernetes object model that you create or deploy. A Pod represents a running process on your cluster and can contain one or more containers.",
        "You can scale a deployment using kubectl scale deployment <deployment-name> --replicas=<number> or by editing the deployment YAML file."
    ]
    
    try:
        metrics = evaluate_answers(test_questions, test_student_answers, test_reference_answers, "test_eval")
        print(f"Test evaluation successful: {metrics}")
    except Exception as e:
        print(f"Test evaluation failed: {str(e)}") 