"""Build the short narrated StaleRoll project demo.

The output is a self-contained MP4 with a conversational macOS voiceover.
The visuals use a pen-and-marker metaphor for proposing and checking repairs,
then switch to the measured 100-task experiment results.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, ImageSequenceClip


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / ".build"
OUTPUT = ROOT / "staleroll_demo.mp4"
WIDTH, HEIGHT = 1280, 720
FPS = 24


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = "/System/Library/Fonts/SFNS.ttf" if not bold else "/System/Library/Fonts/SFNS-Bold.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int, fill: str = "#17202A", bold: bool = False) -> None:
    draw.text(xy, value, font=font(size, bold), fill=fill)


def wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, width: int, size: int, fill: str = "#17202A") -> int:
    words = value.split()
    lines: list[str] = []
    current = ""
    probe = font(size)
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=probe)[2] <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    for index, line in enumerate(lines):
        text(draw, (xy[0], xy[1] + index * (size + 10)), line, size, fill)
    return len(lines) * (size + 10)


def base(title: str, eyebrow: str, accent: str = "#2364AA") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#F7F5EF")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 16), fill=accent)
    text(draw, (72, 60), eyebrow.upper(), 20, accent, True)
    text(draw, (72, 96), title, 44, "#17202A", True)
    draw.line((72, 170, WIDTH - 72, 170), fill="#D7D3C8", width=2)
    return image, draw


def pen_and_marker(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float = 1.0) -> None:
    # Pen: a blue proposal stroke.
    draw.rounded_rectangle((x, y, x + int(230 * scale), y + int(38 * scale)), radius=int(18 * scale), fill="#2364AA")
    draw.polygon([(x + int(230 * scale), y), (x + int(275 * scale), y + int(19 * scale)), (x + int(230 * scale), y + int(38 * scale))], fill="#17202A")
    draw.rectangle((x + int(34 * scale), y, x + int(52 * scale), y + int(38 * scale)), fill="#BFD8F1")
    # Marker: a warm verification stroke.
    mx = x + int(20 * scale)
    my = y + int(90 * scale)
    draw.rounded_rectangle((mx, my, mx + int(250 * scale), my + int(50 * scale)), radius=int(12 * scale), fill="#E58B3A")
    draw.rectangle((mx + int(190 * scale), my, mx + int(250 * scale), my + int(50 * scale)), fill="#F4C16E")
    draw.polygon([(mx + int(250 * scale), my), (mx + int(285 * scale), my + int(25 * scale)), (mx + int(250 * scale), my + int(50 * scale))], fill="#9D4E1D")


def scene_title() -> Image.Image:
    image, draw = base("StaleRoll", "A narrated project demo", "#2364AA")
    text(draw, (74, 218), "Async RL with verifiable code repair", 30, "#46515C")
    wrapped(draw, (74, 278), "A small, measurable system for learning when a rollout is still safe to use after the policy has moved on.", 650, 24, "#46515C")
    pen_and_marker(draw, 840, 248, 1.15)
    text(draw, (840, 480), "pen = propose", 22, "#2364AA", True)
    text(draw, (840, 520), "marker = verify", 22, "#B15E1D", True)
    text(draw, (74, 640), "100 execution-verified repair tasks · local LoRA · frozen Gemini reference", 18, "#68737D")
    return image


def scene_setup() -> Image.Image:
    image, draw = base("The simple question", "01 · Setup", "#2364AA")
    text(draw, (74, 220), "Can asynchronous workers move faster", 32, "#17202A", True)
    text(draw, (74, 266), "without quietly lowering answer quality?", 32, "#17202A", True)
    draw.rounded_rectangle((74, 352, 746, 580), radius=18, fill="#FFFFFF", outline="#D7D3C8", width=2)
    text(draw, (104, 382), "Task", 20, "#2364AA", True)
    text(draw, (104, 420), "Implement clamp(x, low, high)", 28, "#17202A", True)
    text(draw, (104, 478), "The model proposes five complete functions.", 22, "#46515C")
    text(draw, (104, 518), "Only one survives the hidden tests.", 22, "#46515C")
    pen_and_marker(draw, 874, 360, 0.8)
    text(draw, (866, 548), "proposal → check", 22, "#46515C", True)
    return image


def scene_verifier() -> Image.Image:
    image, draw = base("The verifier sees what the proxy misses", "02 · Execution evidence", "#B15E1D")
    for y, label, color, desc in [
        (232, "VISIBLE TESTS", "#E58B3A", "quick signal for the trainer"),
        (366, "HIDDEN TESTS", "#2364AA", "the score that counts"),
        (500, "PROXY HACK", "#8B3D35", "visible pass, hidden failure"),
    ]:
        draw.rounded_rectangle((86, y, 500, y + 82), radius=14, fill=color)
        text(draw, (116, y + 25), label, 23, "#FFFFFF", True)
        text(draw, (570, y + 25), desc, 26, "#46515C")
    pen_and_marker(draw, 930, 286, 0.85)
    text(draw, (884, 538), "The marker checks the work", 21, "#B15E1D", True)
    return image


def scene_lora() -> Image.Image:
    image, draw = base("The trainable part is small on purpose", "03 · LoRA training", "#2F7D5A")
    text(draw, (74, 220), "Frozen encoder", 24, "#46515C", True)
    draw.rounded_rectangle((74, 274, 406, 474), radius=18, fill="#DDE8F4", outline="#2364AA", width=3)
    for i in range(4):
        draw.rectangle((110, 310 + i * 32, 370, 326 + i * 32), fill="#8FB6DC")
    text(draw, (116, 500), "context + candidate", 18, "#2364AA")
    draw.line((430, 374, 560, 374), fill="#68737D", width=4)
    draw.polygon([(560, 374), (532, 360), (532, 388)], fill="#68737D")
    text(draw, (596, 220), "Trainable LoRA head", 24, "#2F7D5A", True)
    draw.rounded_rectangle((596, 274, 934, 474), radius=18, fill="#E0F0E8", outline="#2F7D5A", width=3)
    draw.rectangle((644, 326, 750, 422), fill="#80B99A")
    draw.rectangle((780, 326, 886, 422), fill="#4D956E")
    draw.line((750, 374, 780, 374), fill="#2F7D5A", width=5)
    text(draw, (680, 500), "1,032 parameters", 18, "#2F7D5A", True)
    for i, label in enumerate(["eval 0", "eval 5", "eval 10", "final"]):
        x = 1000 + (i % 2) * 128
        y = 284 + (i // 2) * 112
        draw.rounded_rectangle((x, y, x + 106, y + 56), radius=10, fill="#F4C16E")
        text(draw, (x + 15, y + 18), label, 15, "#6A3A16", True)
    text(draw, (74, 612), "The adapter learns from verified reward; the base representation stays fixed.", 21, "#46515C")
    return image


def scene_async() -> Image.Image:
    image, draw = base("Staleness becomes a measured decision", "04 · Controller", "#7B5BA7")
    labels = [("worker A", 246), ("worker B", 348), ("worker C", 450)]
    for label, y in labels:
        text(draw, (82, y - 8), label, 18, "#46515C", True)
        draw.line((220, y + 10, 1070, y + 10), fill="#C8C1D2", width=3)
    for x, y, color in [(292, 246, "#2364AA"), (506, 348, "#2364AA"), (714, 450, "#2364AA"), (612, 246, "#E58B3A"), (864, 348, "#2F7D5A")]:
        draw.ellipse((x, y - 10, x + 22, y + 12), fill=color)
    draw.line((292, 256, 612, 256), fill="#E58B3A", width=8)
    text(draw, (308, 214), "old snapshot", 18, "#B15E1D")
    text(draw, (790, 560), "accept only when lag and KL stay safe", 22, "#7B5BA7", True)
    pen_and_marker(draw, 1040, 220, 0.45)
    return image


def scene_results() -> Image.Image:
    image, draw = base("What the 100-task run actually showed", "05 · Results", "#2364AA")
    headers = [(80, "Arm"), (340, "LoRA pass"), (570, "Gemini pass"), (800, "LoRA / tick"), (1040, "Gemini / tick")]
    for x, label in headers:
        text(draw, (x, 220), label, 18, "#68737D", True)
    rows = [
        ("sync", "0.7951", "0.7839", "0.4180", "0.4143"),
        ("naive async", "0.7552", "0.7747", "0.5916", "0.5986"),
        ("stale filter", "0.7552", "0.7622", "0.5916", "0.5888"),
    ]
    for i, row in enumerate(rows):
        y = 286 + i * 88
        draw.rounded_rectangle((66, y - 12, 1194, y + 56), radius=10, fill="#FFFFFF" if i % 2 == 0 else "#EEEAE0")
        for (x, _), value in zip(headers, row):
            text(draw, (x, y + 10), value, 22, "#17202A", i == 0)
    wrapped(draw, (80, 580), "The local adapter nearly matched Gemini on this controller test. Gemini remained slightly stronger for asynchronous quality, while the LoRA path was faster and fully trainable locally.", 1090, 22, "#46515C")
    return image


def scene_close() -> Image.Image:
    image, draw = base("The point of the project", "06 · Takeaway", "#2F7D5A")
    wrapped(draw, (74, 230), "This is not a single benchmark number. It is a reproducible loop: propose, execute, measure, and keep the stale work that is still trustworthy.", 760, 31, "#17202A")
    pen_and_marker(draw, 910, 288, 0.85)
    text(draw, (74, 604), "Code, checkpoints, reports, and this video are included in the repository.", 21, "#46515C")
    return image


SCENES = [scene_title, scene_setup, scene_verifier, scene_lora, scene_async, scene_results, scene_close]
NARRATION = [
    "Here is StaleRoll. I use a simple pen and marker idea throughout: the pen proposes a repair, and the marker checks whether it actually works.",
    "The project asks one practical question. Can asynchronous rollout workers move faster without quietly lowering the quality of the answers they produce?",
    "Each task is a complete Python repair. Visible tests give a quick training signal, but hidden tests decide the real score. A visible-only success is recorded as proxy reward hacking.",
    "The trainable part is deliberately small. A frozen Transformer turns the task and candidate into a representation, and a rank-eight LoRA head learns from verified reward. The run saved evaluations every five updates and checkpoints every ten.",
    "Workers do not all finish at the same time. StaleRoll measures how old each sample is, compares the old and current policy, and only keeps stale work when the lag and KL are safe.",
    "On one hundred repair tasks, the local adapter reached a seventy-five point five percent verified pass rate in the stale-filter arm, with slightly higher verified throughput than frozen Gemini. Gemini was still a little stronger on asynchronous quality.",
    "The result is a complete, inspectable loop: propose, execute, measure, and keep the work that is still trustworthy. The repository contains the code, checkpoints, reports, and this demo.",
]


def main() -> None:
    BUILD.mkdir(exist_ok=True)
    frames: list[Image.Image] = []
    durations = [8, 7, 8, 9, 7, 9, 7]
    for index, scene in enumerate(SCENES):
        image = scene()
        frames.extend([image] * (durations[index] * FPS))
        image.save(BUILD / f"scene-{index:02d}.png")
    narration = ROOT / "narration.txt"
    narration.write_text("\n\n".join(NARRATION) + "\n", encoding="utf-8")
    audio_files: list[Path] = []
    for index, paragraph in enumerate(NARRATION):
        audio = BUILD / f"voice-{index:02d}.aiff"
        subprocess.run(["say", "-v", "Samantha", "-r", "174", "-o", str(audio), paragraph], check=True)
        audio_files.append(audio)
    audio = AudioFileClip(str(audio_files[0]))
    for path in audio_files[1:]:
        next_clip = AudioFileClip(str(path))
        audio = audio.with_duration(audio.duration + next_clip.duration)
        # MoviePy's concatenate_videoclips is not needed for voice; the audio
        # track is assembled by the ffmpeg writer from the individual files.
        next_clip.close()
    # Use the scene timings as the visual track and the first voice file as a
    # conservative fallback if a local MoviePy build cannot concatenate audio.
    # The voice files are concatenated explicitly below with ffmpeg through
    # MoviePy's audio concatenate helper.
    from moviepy import concatenate_audioclips

    voice = concatenate_audioclips([AudioFileClip(str(path)) for path in audio_files])
    video = ImageSequenceClip([str(BUILD / f"scene-{i:02d}.png") for i in range(len(SCENES))], durations=durations)
    video = video.with_audio(voice)
    video.write_videofile(str(OUTPUT), fps=FPS, codec="libx264", audio_codec="aac", bitrate="2200k", logger=None)
    video.close()
    voice.close()
    audio.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
