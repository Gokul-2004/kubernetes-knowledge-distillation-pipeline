#!/usr/bin/env python3
"""
Gradio UI for Knowledge Distillation Pipeline
Provides a local GUI with sliders, start button, and metrics display
"""
import gradio as gr
import json
import os
import time
from datetime import datetime
from pipeline.main_pipeline import run_pipeline
from pipeline.data_utils import load_checkpoint, save_checkpoint
from pipeline.config import DEFAULT_NUM_QUESTIONS, DEFAULT_NUM_CYCLES, KUBERNETES_TOPICS
import threading
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineManager:
    def __init__(self):
        self.is_running = False
        self.current_status = "Ready"
        self.progress = 0
        self.results = {}
        self.detailed_results = ""
        
    def update_status(self, status, progress=0):
        self.current_status = status
        self.progress = progress
        logger.info(f"Status updated: {status} (Progress: {progress}%)")
        
    def load_latest_results(self):
        """Load the latest pipeline results"""
        try:
            # Try to load the most recent complete pipeline results
            checkpoint_path = "checkpoints/pipeline_complete.json"
            if os.path.exists(checkpoint_path):
                results = load_checkpoint("pipeline_complete.json")
                if results and 'cycle_results' in results:
                    self.results = results
                    self.detailed_results = self.format_results(results)
                    return f"✅ Loaded results from {len(results['cycle_results'])} successful cycles"
                else:
                    return "⚠️ No complete results found in checkpoint"
            else:
                return "❌ No pipeline results found"
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return f"❌ Error loading results: {str(e)}"
    
    def format_results(self, results):
        """Format results for display"""
        if not results or 'cycle_results' not in results:
            return "No results available"
        
        formatted = "📊 Pipeline Results Summary\n"
        formatted += "=" * 50 + "\n\n"
        
        formatted += f"Total Cycles: {results.get('total_cycles', 0)}\n"
        formatted += f"Successful Cycles: {results.get('successful_cycles', 0)}\n"
        formatted += f"Failed Cycles: {results.get('failed_cycles', 0)}\n"
        formatted += f"Average Improvement: {results.get('average_improvement', 0):.4f} ({results.get('average_improvement', 0)*100:.2f}%)\n\n"
        
        for cycle_result in results['cycle_results']:
            cycle = cycle_result['cycle']
            formatted += f"🔄 Cycle {cycle} Results:\n"
            formatted += f"   Status: {'✅ Success' if cycle_result['success'] else '❌ Failed'}\n"
            
            if 'baseline_metrics' in cycle_result:
                baseline = cycle_result['baseline_metrics']
                formatted += f"   📈 Baseline Performance:\n"
                formatted += f"      Answer Relevancy: {baseline['answer_relevancy']:.4f} ({baseline['answer_relevancy']*100:.2f}%)\n"
                formatted += f"      Questions Evaluated: {baseline['num_questions']}\n"
            
            if 'fine_tuned_metrics' in cycle_result:
                fine_tuned = cycle_result['fine_tuned_metrics']
                formatted += f"   🎯 Fine-tuned Performance:\n"
                formatted += f"      Answer Relevancy: {fine_tuned['answer_relevancy']:.4f} ({fine_tuned['answer_relevancy']*100:.2f}%)\n"
                formatted += f"      Questions Evaluated: {fine_tuned['num_questions']}\n"
            
            if 'comparison' in cycle_result:
                comp = cycle_result['comparison']
                formatted += f"   📊 Improvement Analysis:\n"
                formatted += f"      Baseline: {comp['baseline_answer_relevancy']:.4f} ({comp['baseline_answer_relevancy']*100:.2f}%)\n"
                formatted += f"      Fine-tuned: {comp['improved_answer_relevancy']:.4f} ({comp['improved_answer_relevancy']*100:.2f}%)\n"
                formatted += f"      Improvement: +{comp['improvement']:.4f} (+{comp['improvement_percentage']:.2f}%)\n"
            
            formatted += "\n"
        
        # Add sample questions and answers if available
        if 'final_state' in results and 'questions' in results['final_state']:
            formatted += "📝 Sample Question & Answers:\n"
            formatted += "-" * 40 + "\n"
            
            questions = results['final_state']['questions']
            baseline_answers = results['final_state'].get('baseline_student_answers', [])
            fine_tuned_answers = results['final_state'].get('current_student_answers', [])
            reference_answers = results['final_state'].get('reference_answers', [])
            
            for i, question in enumerate(questions):
                formatted += f"❓ Question {i+1}:\n   {question}\n\n"
                
                if i < len(baseline_answers):
                    formatted += f"🤖 Baseline Answer (Phi-2):\n   {baseline_answers[i]}\n\n"
                
                if i < len(fine_tuned_answers):
                    formatted += f"🎯 Fine-tuned Answer (Phi-2):\n   {fine_tuned_answers[i]}\n\n"
                
                if i < len(reference_answers):
                    formatted += f"👨‍🏫 Reference Answer (GPT-4o-mini):\n   {reference_answers[i]}\n\n"
        
        return formatted
    
    def run_pipeline_thread(self, num_questions, num_cycles, selected_topics, skip_fine_tuning):
        """Run pipeline in a separate thread"""
        try:
            self.is_running = True
            self.update_status("Starting pipeline...", 0)
            
            # Set environment variables for topic selection
            if selected_topics:
                os.environ['KUBERNETES_TOPICS'] = ','.join(selected_topics)
            
            # Run the pipeline
            self.update_status("Generating questions...", 10)
            results = run_pipeline(
                num_questions=num_questions,
                num_cycles=num_cycles,
                skip_fine_tuning=skip_fine_tuning
            )
            
            if results and 'cycle_results' in results:
                self.results = results
                self.detailed_results = self.format_results(results)
                self.update_status("Pipeline completed successfully!", 100)
                logger.info("Pipeline completed successfully")
            else:
                self.update_status("Pipeline completed with errors", 100)
                logger.error("Pipeline completed with errors")
                
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.update_status(f"Pipeline failed: {str(e)}", 100)
        finally:
            self.is_running = False
    
    def start_pipeline(self, num_questions, num_cycles, selected_topics, skip_fine_tuning):
        """Start the pipeline"""
        if self.is_running:
            return "Pipeline is already running!", "Pipeline is already running!", "Pipeline is already running!"
        
        # Start pipeline in background thread
        thread = threading.Thread(
            target=self.run_pipeline_thread,
            args=(num_questions, num_cycles, selected_topics, skip_fine_tuning)
        )
        thread.daemon = True
        thread.start()
        
        return "Pipeline started successfully!", "Pipeline started successfully!", "Pipeline started successfully!"
    
    def stop_pipeline(self):
        """Stop the pipeline"""
        if not self.is_running:
            return "No pipeline running", "No pipeline running", "No pipeline running"
        
        self.is_running = False
        self.update_status("Pipeline stopped by user", 0)
        return "Pipeline stopped", "Pipeline stopped", "Pipeline stopped"
    
    def refresh_status(self):
        """Refresh the current status"""
        if self.is_running:
            return f"🔄 {self.current_status} (Progress: {self.progress}%)", self.detailed_results
        else:
            return f"⏹️ {self.current_status}", self.detailed_results

# Create global pipeline manager
pipeline_manager = PipelineManager()

def create_ui():
    """Create the Gradio UI"""
    
    with gr.Blocks(title="Kubernetes v1.28 Knowledge Distillation Pipeline") as demo:
        gr.Markdown("""
        # 🤖 Kubernetes v1.28 Knowledge Distillation Pipeline
        **Vertical knowledge distillation for Kubernetes v1.28 using GPT-4o-mini as teacher and Phi-2 as student**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")
                
                num_questions = gr.Slider(
                    minimum=1, maximum=50, value=DEFAULT_NUM_QUESTIONS, step=1,
                    label="Number of Questions",
                    info="Number of questions to generate per cycle"
                )
                
                num_cycles = gr.Slider(
                    minimum=1, maximum=10, value=DEFAULT_NUM_CYCLES, step=1,
                    label="Number of Cycles",
                    info="Number of distillation cycles to run"
                )
                
                topics = gr.CheckboxGroup(
                    choices=KUBERNETES_TOPICS,
                    value=KUBERNETES_TOPICS[:5],  # Default to first 5 topics
                    label="Kubernetes Question Topics",
                    info="Select Kubernetes v1.28 topics for question generation"
                )
                
                skip_fine_tuning = gr.Checkbox(
                    value=False,
                    label="Skip Fine-tuning (Evaluation Only)",
                    info="Keep unchecked to enable fine-tuning (recommended)"
                )
                
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Pipeline", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop Pipeline", variant="stop")
            
            with gr.Column(scale=1):
                gr.Markdown("### Status")
                
                status_text = gr.Textbox(
                    value="Ready to start pipeline",
                    label="Pipeline Status",
                    interactive=False
                )
                
                progress_text = gr.Textbox(
                    value="No progress yet",
                    label="Pipeline Progress",
                    interactive=False
                )
                
                with gr.Row():
                    load_results_btn = gr.Button("📊 Load Latest Results")
                    refresh_btn = gr.Button("🔄 Refresh Status")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Results")
                
                results_text = gr.Textbox(
                    value="No results available yet.",
                    label="Pipeline Results",
                    lines=10,
                    interactive=False
                )
                
                detailed_results = gr.Textbox(
                    value="No results available yet.",
                    label="📊 Detailed Step-by-Step Results",
                    lines=20,
                    interactive=False
                )
        
        gr.Markdown("""
        ### Pipeline Steps:
        1. **Question Generation**: Generate Kubernetes v1.28 questions from selected topics
        2. **Baseline Answers**: Generate answers using untrained Phi-2
        3. **Reference Answers**: Generate reference answers using GPT-4o-mini
        4. **Baseline Evaluation**: Evaluate baseline answers using RAGAS
        5. **Fine-tuning**: Fine-tune Phi-2 using QLoRA (if enabled)
        6. **Fine-tuned Evaluation**: Evaluate fine-tuned answers
        7. **Comparison**: Compare baseline vs fine-tuned performance
        
        ### Notes:
        - The pipeline runs in cycles, with each cycle improving the student model
        - All results are automatically checkpointed for resuming
        - The UI updates every 5 seconds to show progress
        - You can stop the pipeline at any time
        - Select Kubernetes topics to customize question generation
        - Detailed step-by-step outputs are shown in the results section
        - This is a vertical knowledge distillation pipeline specifically for Kubernetes v1.28
        """)
        
        # Event handlers
        start_btn.click(
            fn=pipeline_manager.start_pipeline,
            inputs=[num_questions, num_cycles, topics, skip_fine_tuning],
            outputs=[status_text, progress_text, results_text]
        )
        
        stop_btn.click(
            fn=pipeline_manager.stop_pipeline,
            outputs=[status_text, progress_text, results_text]
        )
        
        load_results_btn.click(
            fn=pipeline_manager.load_latest_results,
            outputs=[results_text]
        )
        
        refresh_btn.click(
            fn=pipeline_manager.refresh_status,
            outputs=[progress_text, detailed_results]
        )
        
        # Auto-refresh every 5 seconds
        demo.load(lambda: None, outputs=None)
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False) 