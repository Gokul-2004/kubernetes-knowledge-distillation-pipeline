#!/usr/bin/env python3
"""
Launcher script for the Knowledge Distillation Pipeline UI
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.gradio_ui import create_ui

if __name__ == "__main__":
    print("🚀 Starting Knowledge Distillation Pipeline UI...")
    print("📊 The UI will be available at: http://127.0.0.1:7860")
    print("⏹️  Press Ctrl+C to stop the server")
    
    try:
        demo = create_ui()
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n👋 UI stopped by user")
    except Exception as e:
        print(f"❌ Error starting UI: {str(e)}")
        sys.exit(1) 