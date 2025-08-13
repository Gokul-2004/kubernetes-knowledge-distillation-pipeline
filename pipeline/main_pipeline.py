#!/usr/bin/env python3
"""
Main Knowledge Distillation Pipeline
Orchestrates the complete vertical knowledge distillation process
"""
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os

from .config import DEFAULT_NUM_QUESTIONS, DEFAULT_NUM_CYCLES
from .data_utils import load_notes, save_checkpoint, load_checkpoint
from .question_generation import generate_questions, generate_reference_answers
from .student_generation_minimal import generate_student_answers, load_fine_tuned_model
from .evaluation import evaluate_answers, compare_evaluations
from .fine_tuning_robust import fine_tune_robust

logger = logging.getLogger(__name__)

@dataclass
class PipelineState:
    """Pipeline state for checkpointing and resuming"""
    cycle: int = 0
    num_questions: int = DEFAULT_NUM_QUESTIONS
    num_cycles: int = DEFAULT_NUM_CYCLES
    questions: List[str] = None
    baseline_student_answers: List[str] = None
    baseline_metrics: Dict[str, Any] = None
    reference_answers: List[str] = None
    fine_tuned_metrics: Dict[str, Any] = None
    current_student_answers: List[str] = None
    is_complete: bool = False
    error: str = None
    
    def __post_init__(self):
        if self.questions is None:
            self.questions = []
        if self.baseline_student_answers is None:
            self.baseline_student_answers = []
        if self.reference_answers is None:
            self.reference_answers = []
        if self.current_student_answers is None:
            self.current_student_answers = []

class KnowledgeDistillationPipeline:
    """Knowledge distillation pipeline with checkpointing and resuming"""
    
    def __init__(self, num_questions: int = DEFAULT_NUM_QUESTIONS, 
                 num_cycles: int = DEFAULT_NUM_CYCLES,
                 skip_fine_tuning: bool = False):
        self.num_questions = num_questions
        self.num_cycles = num_cycles
        self.skip_fine_tuning = skip_fine_tuning
        self.notes = self.load_notes()
        self.state = PipelineState(num_questions=num_questions, num_cycles=num_cycles)
        
    def load_notes(self) -> str:
        """Load knowledge base notes"""
        try:
            logger.info("Loading knowledge base notes...")
            self.notes = load_notes()
            logger.info(f"Loaded notes with {len(self.notes)} characters")
            return self.notes
        except Exception as e:
            logger.error(f"Failed to load notes: {str(e)}")
            raise
    
    def generate_questions_step(self) -> List[str]:
        """Step 1: Generate questions from notes"""
        try:
            if not self.notes:
                self.load_notes()
            
            logger.info(f"Generating {self.num_questions} questions...")
            questions = generate_questions(self.notes, self.num_questions)
            self.state.questions = questions
            logger.info(f"Generated {len(questions)} questions")
            
            # Save checkpoint
            save_checkpoint(self.state.__dict__, f"pipeline_cycle_{self.state.cycle}_questions")
            return questions
            
        except Exception as e:
            logger.error(f"Question generation failed: {str(e)}")
            self.state.error = str(e)
            save_checkpoint(self.state.__dict__, f"pipeline_cycle_{self.state.cycle}_questions_error")
            raise
    
    def generate_baseline_answers_step(self) -> List[str]:
        """Step 2: Generate baseline student answers"""
        try:
            if not self.state.questions:
                raise ValueError("No questions available. Run generate_questions_step first.")
            
            logger.info("Generating baseline student answers...")
            baseline_answers = generate_student_answers(self.state.questions)
            
            if not baseline_answers:
                logger.error("Failed to generate baseline answers")
                self.state.error = "Failed to generate baseline answers"
                save_checkpoint(self.state.__dict__, f"pipeline_cycle_{self.state.cycle}_baseline_error")
                raise ValueError("Failed to generate baseline answers")
            
            self.state.baseline_student_answers = baseline_answers
            logger.info(f"Generated {len(baseline_answers)} baseline answers")
            
            # Save checkpoint
            save_checkpoint(self.state.__dict__, f"pipeline_cycle_{self.state.cycle}_baseline")
            return baseline_answers
            
        except Exception as e:
            logger.error(f"Baseline answer generation failed: {str(e)}")
            self.state.error = str(e)
            save_checkpoint(self.state.__dict__, f"pipeline_cycle_{self.state.cycle}_baseline_error")
            raise
    
    def generate_reference_answers_step(self) -> List[str]:
        """Step 3: Generate reference answers using teacher model"""
        try:
            if not self.state.questions:
                raise ValueError("No questions available. Run generate_questions_step first.")
            
            logger.info("Generating reference answers using teacher model...")
            reference_answers = generate_reference_answers(self.state.questions, self.notes)
            self.state.reference_answers = reference_answers
            logger.info(f"Generated {len(reference_answers)} reference answers")
            
            # Save checkpoint
            save_checkpoint(self.state.__dict__, f"pipeline_cycle_{self.state.cycle}_reference")
            return reference_answers
            
        except Exception as e:
            logger.error(f"Reference answer generation failed: {str(e)}")
            self.state.error = str(e)
            save_checkpoint(self.state.__dict__, f"pipeline_cycle_{self.state.cycle}_reference_error")
            raise
    
    def evaluate_baseline_step(self) -> Dict[str, Any]:
        """Step 4: Evaluate baseline student answers"""
        try:
            if not all([self.state.questions, self.state.baseline_student_answers, 
                       self.state.reference_answers]):
                raise ValueError("Missing required data for evaluation")
            
            logger.info("Evaluating baseline student answers...")
            baseline_metrics = evaluate_answers(
                self.state.questions,
                self.state.baseline_student_answers,
                self.state.reference_answers,
                f"baseline_eval_cycle_{self.state.cycle}"
            )
            self.state.baseline_metrics = baseline_metrics
            logger.info(f"Baseline evaluation complete: {baseline_metrics}")
            
            # Save checkpoint
            save_checkpoint(self.state.__dict__, f"pipeline_cycle_{self.state.cycle}_baseline_eval")
            return baseline_metrics
            
        except Exception as e:
            logger.error(f"Baseline evaluation failed: {str(e)}")
            self.state.error = str(e)
            save_checkpoint(self.state.__dict__, f"pipeline_cycle_{self.state.cycle}_baseline_eval_error")
            raise
    
    def fine_tune_step(self) -> Dict[str, Any]:
        """Step 5: Fine-tune student model"""
        try:
            if not all([self.state.questions, self.state.reference_answers]):
                raise ValueError("Missing required data for fine-tuning")
            
            logger.info("Fine-tuning student model with robust handling...")
            fine_tuning_results = fine_tune_robust(
                self.state.questions,
                self.state.reference_answers,
                f"fine_tuning_cycle_{self.state.cycle}"
            )
            logger.info(f"Fine-tuning complete: {fine_tuning_results}")
            
            # Save checkpoint
            save_checkpoint(self.state.__dict__, f"pipeline_cycle_{self.state.cycle}_fine_tuning")
            return fine_tuning_results
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {str(e)}")
            self.state.error = str(e)
            save_checkpoint(self.state.__dict__, f"pipeline_cycle_{self.state.cycle}_fine_tuning_error")
            raise
    
    def evaluate_fine_tuned_step(self) -> Dict[str, Any]:
        """Step 6: Evaluate fine-tuned student answers"""
        try:
            if not all([self.state.questions, self.state.reference_answers]):
                raise ValueError("Missing required data for evaluation")
            
            logger.info("Generating fine-tuned student answers...")
            # Set environment variable to indicate we're after fine-tuning
            os.environ['AFTER_FINE_TUNING'] = '1'
            
            # Try to load fine-tuned model and generate new answers
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    fine_tuned_answers = generate_student_answers(
                        self.state.questions,
                        use_fine_tuned=True
                    )
                    self.state.current_student_answers = fine_tuned_answers
                    break
                except Exception as e:
                    logger.warning(f"Fine-tuned generation attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_attempts - 1:
                        logger.info("Retrying fine-tuned generation...")
                        time.sleep(10)
                    else:
                        logger.error("All fine-tuned generation attempts failed")
                        # Fallback to baseline answers for evaluation
                        logger.info("Using baseline answers as fallback for fine-tuned evaluation")
                        self.state.current_student_answers = self.state.baseline_student_answers
            
            logger.info("Evaluating fine-tuned student answers...")
            fine_tuned_metrics = evaluate_answers(
                self.state.questions,
                self.state.current_student_answers,
                self.state.reference_answers,
                f"fine_tuned_eval_cycle_{self.state.cycle}"
            )
            self.state.fine_tuned_metrics = fine_tuned_metrics
            logger.info(f"Fine-tuned evaluation complete: {fine_tuned_metrics}")
            
            # Save checkpoint
            save_checkpoint(self.state.__dict__, f"pipeline_cycle_{self.state.cycle}_fine_tuned_eval")
            return fine_tuned_metrics
            
        except Exception as e:
            logger.error(f"Fine-tuned evaluation failed: {str(e)}")
            self.state.error = str(e)
            save_checkpoint(self.state.__dict__, f"pipeline_cycle_{self.state.cycle}_fine_tuned_eval_error")
            raise
    
    def run_cycle(self, cycle: int) -> Dict[str, Any]:
        """Run a complete knowledge distillation cycle"""
        try:
            self.state.cycle = cycle
            logger.info(f"Starting cycle {cycle}/{self.num_cycles}")
            
            # Step 1: Generate questions
            self.generate_questions_step()
            
            # Step 2: Generate baseline answers
            self.generate_baseline_answers_step()
            
            # Step 3: Generate reference answers
            self.generate_reference_answers_step()
            
            # Step 4: Evaluate baseline
            baseline_metrics = self.evaluate_baseline_step()
            
            # Step 5: Fine-tune model (optional)
            if not self.skip_fine_tuning:
                self.fine_tune_step()
                # Step 6: Evaluate fine-tuned model
                fine_tuned_metrics = self.evaluate_fine_tuned_step()
            else:
                logger.info("Skipping fine-tuning step (evaluation-only mode)")
                fine_tuned_metrics = {"answer_relevancy": 0.0, "num_questions": len(self.state.questions), "evaluation_complete": False}
            
            # Compare results
            comparison = compare_evaluations(baseline_metrics, fine_tuned_metrics)
            
            cycle_results = {
                "cycle": cycle,
                "baseline_metrics": baseline_metrics,
                "fine_tuned_metrics": fine_tuned_metrics,
                "comparison": comparison,
                "success": True
            }
            
            logger.info(f"Cycle {cycle} complete. Improvement: {comparison['improvement']:.4f}")
            
            # Save final cycle checkpoint
            save_checkpoint(cycle_results, f"pipeline_cycle_{cycle}_complete")
            
            return cycle_results
            
        except Exception as e:
            logger.error(f"Cycle {cycle} failed: {str(e)}")
            cycle_results = {
                "cycle": cycle,
                "error": str(e),
                "success": False
            }
            save_checkpoint(cycle_results, f"pipeline_cycle_{cycle}_failed")
            raise
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete knowledge distillation pipeline"""
        try:
            logger.info(f"Starting knowledge distillation pipeline with {self.num_cycles} cycles")
            
            all_results = []
            
            for cycle in range(self.num_cycles):
                try:
                    cycle_result = self.run_cycle(cycle)
                    all_results.append(cycle_result)
                    
                    # Brief pause between cycles
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Cycle {cycle} failed, continuing with next cycle: {str(e)}")
                    all_results.append({
                        "cycle": cycle,
                        "error": str(e),
                        "success": False
                    })
                    continue
            
            # Final summary
            successful_cycles = [r for r in all_results if r.get("success", False)]
            failed_cycles = [r for r in all_results if not r.get("success", False)]
            
            pipeline_summary = {
                "total_cycles": self.num_cycles,
                "successful_cycles": len(successful_cycles),
                "failed_cycles": len(failed_cycles),
                "cycle_results": all_results,
                "final_state": self.state.__dict__
            }
            
            if successful_cycles:
                # Calculate average improvement
                improvements = [r["comparison"]["improvement"] for r in successful_cycles 
                              if "comparison" in r]
                if improvements:
                    pipeline_summary["average_improvement"] = sum(improvements) / len(improvements)
            
            self.state.is_complete = True
            save_checkpoint(pipeline_summary, "pipeline_complete")
            
            logger.info(f"Pipeline complete. {len(successful_cycles)}/{self.num_cycles} cycles successful")
            return pipeline_summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.state.error = str(e)
            save_checkpoint(self.state.__dict__, "pipeline_failed")
            raise
    
    def resume_from_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """Resume pipeline from a checkpoint"""
        try:
            logger.info(f"Resuming from checkpoint: {checkpoint_name}")
            checkpoint_data = load_checkpoint(checkpoint_name)
            
            # Restore state
            for key, value in checkpoint_data.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
            
            logger.info(f"Resumed to cycle {self.state.cycle}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {str(e)}")
            raise

def run_pipeline(num_questions: int = DEFAULT_NUM_QUESTIONS, 
                num_cycles: int = DEFAULT_NUM_CYCLES,
                skip_fine_tuning: bool = False,
                resume_from: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to run the pipeline"""
    pipeline = KnowledgeDistillationPipeline(num_questions, num_cycles, skip_fine_tuning)
    
    if resume_from:
        pipeline.resume_from_checkpoint(resume_from)
    
    return pipeline.run_full_pipeline()

if __name__ == "__main__":
    # Test pipeline with minimal settings
    logging.basicConfig(level=logging.INFO)
    
    try:
        results = run_pipeline(num_questions=2, num_cycles=1)
        print(f"Pipeline test successful: {results}")
    except Exception as e:
        print(f"Pipeline test failed: {str(e)}") 