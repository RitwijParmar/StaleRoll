# StaleRoll demo

`staleroll_demo.mp4` is a short narrated walkthrough of the project. It uses a pen-and-marker metaphor for proposing and checking repairs, then shows the actual 100-task LoRA and frozen-Gemini measurements.

[![StaleRoll demo poster](staleroll_demo_poster.png)](staleroll_demo.mp4)

The narration is conversational and is generated locally with the macOS Samantha voice. The source script is in `narration.txt`; the visual and audio build is reproducible with:

```bash
python3 demo/generate_demo.py
```

The demo deliberately states the result as measured: frozen Gemini remains slightly stronger for asynchronous quality, while the LoRA path is trainable locally and has comparable verified throughput.
