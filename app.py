"""Hugging Face Space front-end (Gradio SDK). Add GROQ_API_KEY / GEMINI_API_KEY as Space secrets."""
import gradio as gr
from bot import run_pipeline


def generate(source: str):
    if not source.strip():
        raise gr.Error("Paste a product URL or description first.")
    video, script = run_pipeline(source, out_dir="output")
    caption = f"{script.get('post_caption', '')}\n{' '.join(script.get('hashtags', []))}"
    return str(video), caption, script


gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Product URL or description", lines=3),
    outputs=[gr.Video(label="Video"), gr.Textbox(label="Post caption"), gr.JSON(label="Script")],
    title="Free Social Video Bot",
    flagging_mode="never",
).launch()
