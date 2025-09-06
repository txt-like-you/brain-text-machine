from model import build_interface
import gradio as gr

demo = build_interface()

demo.launch(server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False)