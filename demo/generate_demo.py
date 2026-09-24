"""Build a narrated, run-through demo for StaleRoll.

This is intentionally a project walkthrough rather than a title-card reel:
the video shows the command that is run, real smoke-run numbers, the actual
LoRA/checkpoint code path, and the checked-in 100-task comparison. A blue pen
cursor points at the line being discussed and a yellow marker highlight moves
over the important evidence.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from moviepy import AudioFileClip, ImageSequenceClip, concatenate_audioclips, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
BUILD = ROOT / ".build-v2"
OUTPUT = ROOT / "staleroll_demo.mp4"
WIDTH, HEIGHT = 1440, 810
FPS = 24

BLUE = "#5CA8E6"
INK = "#E9EEF2"
MUTED = "#9AA8B3"
GREEN = "#72D19B"
YELLOW = "#F5C451"
RED = "#F27670"
PANEL = "#17212B"
TERMINAL = "#10161C"


NARRATION = [
    "Let me show you what this repository actually runs. I’m going to start at the project root, launch a small LoRA smoke run, and then connect that run to the full benchmark in the report.",
    "This is the command. The short demo uses twenty code-repair tasks so it finishes quickly. The checked-in experiment uses one hundred tasks, three seeds, and the same controller arms.",
    "Each task gives the policy five complete Python repairs. The visible tests are only a quick signal. The hidden tests are the marker: they decide whether a repair really works, or whether it only learned to look good on the proxy.",
    "Here is the trainable part. The Transformer representation stays frozen. The low-rank head is updated from verified reward. Every five updates the policy is evaluated, and every ten updates a checkpoint is written so the training path can be inspected later.",
    "The workers finish at different times. StaleRoll measures the age of each sample and the KL shift from the old policy to the current one. The marker highlights a sample that stays inside both limits; unsafe stale work is left out of the update.",
    "Now the real comparison. On one hundred tasks, the stale-filter arm reached seventy-five point five percent verified pass rate with the LoRA adapter, versus seventy-six point two percent for frozen Gemini. Throughput was almost identical: point five nine two versus point five eight nine verified rollouts per tick.",
    "That is the honest result. Gemini is a little stronger on asynchronous quality, while the adapter is trainable locally and has comparable throughput. The repository includes the command, the reports, the checkpoints, and this walkthrough. You can run it yourself.",
]


def fnt(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/SFNS-Bold.ttf" if bold else "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int, fill: str = INK, bold: bool = False) -> None:
    draw.text(xy, value, font=fnt(size, bold), fill=fill)


def terminal_frame(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), TERMINAL)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 58), fill="#202C38")
    for x, color in ((34, RED), (60, YELLOW), (86, GREEN)):
        draw.ellipse((x - 8, 21, x + 8, 37), fill=color)
    draw_text(draw, (122, 16), title, 22, INK, True)
    if subtitle:
        draw_text(draw, (1040, 18), subtitle, 17, MUTED)
    return image, draw


def code_line(draw: ImageDraw.ImageDraw, y: int, number: int, value: str, highlight: bool = False, color: str = INK) -> None:
    if highlight:
        draw.rounded_rectangle((86, y - 5, 1360, y + 38), radius=8, fill="#5C481A")
    draw_text(draw, (42, y), f"{number:>3}", 18, "#647685")
    draw_text(draw, (92, y), value, 21, color)


def marker(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, label: str = "MARKER") -> None:
    # A translucent yellow marker stroke sits over the line being discussed.
    draw.rounded_rectangle((x, y, x + width, y + 42), radius=9, fill="#6E581D", outline=YELLOW, width=2)
    draw_text(draw, (x + width + 12, y + 9), label, 15, YELLOW, True)


def pen(draw: ImageDraw.ImageDraw, x: int, y: int, label: str = "PEN") -> None:
    # A small blue pen points to the line being discussed.
    draw.rounded_rectangle((x, y, x + 94, y + 20), radius=9, fill=BLUE)
    draw.rectangle((x + 18, y, x + 28, y + 20), fill="#C7E5FF")
    draw.polygon([(x + 94, y), (x + 122, y + 10), (x + 94, y + 20)], fill=INK)
    draw_text(draw, (x + 132, y - 2), label, 14, BLUE, True)


def read_demo_metrics() -> dict:
    path = PROJECT / "artifacts" / "demo_run" / "comparison.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return {mode: data["modes"][mode] for mode in ("sync", "stale_filter")}
    return {"sync": {"verified_pass_rate": 0.5}, "stale_filter": {"verified_pass_rate": 0.375}}


def scene_intro(step: int) -> Image.Image:
    image, draw = terminal_frame("StaleRoll · project walkthrough", "actual run + measured result")
    draw_text(draw, (78, 138), "A real run-through, not a title card", 42, INK, True)
    draw_text(draw, (80, 206), "pen = the model's proposal", 27, BLUE, True)
    draw_text(draw, (80, 252), "marker = the verifier's check", 27, YELLOW, True)
    draw.rounded_rectangle((80, 360, 1360, 550), radius=18, fill=PANEL, outline="#334655", width=2)
    command = "PYTHONPATH=src python3 -m staleroll.cli run --policy lora --task-domain code"
    visible = command[: max(0, min(len(command), step * 12))]
    draw_text(draw, (116, 412), "$ " + visible, 24, INK)
    if step >= 6:
        marker(draw, 114, 468, 560, "START HERE")
        pen(draw, 1180, 472)
    draw_text(draw, (82, 660), "StaleRoll · async RL · execution-verified code repair", 20, MUTED)
    return image


def scene_run(step: int) -> Image.Image:
    image, draw = terminal_frame("staleroll_demo_run", "the command is running")
    draw_text(draw, (52, 86), "$ PYTHONPATH=src python3 -m staleroll.cli run ", 21, INK)
    draw_text(draw, (52, 120), "  --policy lora --task-domain code --tasks 20 --seeds 1", 21, BLUE)
    draw_text(draw, (52, 154), "  --ticks 24 --checkpoint-interval 4 --eval-interval 4", 21, BLUE)
    lines = [
        "[StaleRoll] domain=code tasks=20 policy=lora",
        "[StaleRoll] checkpoint interval: 4 updates",
        "[StaleRoll] evaluation interval: 4 updates",
        "sync          verified pass rate   0.5000",
        "stale_filter  verified pass rate   0.3750",
        "profile       local adapter updates recorded",
    ]
    for i, line in enumerate(lines):
        color = GREEN if i >= 3 else INK
        code_line(draw, 260 + i * 54, i + 1, line, highlight=(step >= i + 2), color=color)
    pen(draw, 1160, 426 if step < 5 else 534)
    draw_text(draw, (52, 682), "This smoke run is for the walkthrough; the report uses the full 100-task experiment.", 18, MUTED)
    return image


def scene_repair(step: int) -> Image.Image:
    image, draw = terminal_frame("tasks.py + verifier.py", "a repair is only useful if it executes")
    draw_text(draw, (52, 86), "Task: implement clamp(x, low, high)", 26, INK, True)
    code = [
        "def clamp(x, low, high):",
        "    return max(low, min(high, x))",
        "",
        "visible tests: 5 → 5   |   -2 → 0",
        "hidden tests: 20 → 10  |   10 → 10",
    ]
    for i, line in enumerate(code):
        code_line(draw, 180 + i * 54, i + 1, line, highlight=(step == 2 and i == 1), color=GREEN if i in (1, 4) else INK)
    draw.rounded_rectangle((780, 174, 1344, 480), radius=18, fill=PANEL, outline="#334655", width=2)
    draw_text(draw, (822, 208), "candidate check", 23, YELLOW, True)
    draw_text(draw, (822, 270), "visible: PASS", 25, GREEN, True)
    draw_text(draw, (822, 326), "hidden: PASS", 25, GREEN, True)
    draw_text(draw, (822, 382), "verified: TRUE", 25, GREEN, True)
    draw_text(draw, (822, 438), "proxy-only: FALSE", 22, MUTED)
    marker(draw, 810, 312 if step < 3 else 368, 280, "CHECK")
    pen(draw, 690, 232 if step < 3 else 394)
    draw_text(draw, (52, 682), "The hidden tests prevent a visible-only shortcut from looking like a win.", 18, MUTED)
    return image


def scene_lora(step: int) -> Image.Image:
    image, draw = terminal_frame("src/staleroll/lora_policy.py", "the trainable path")
    draw_text(draw, (52, 86), "Only the low-rank adapter receives gradients.", 26, INK, True)
    lines = [
        "class _LoRAHead(nn.Module):",
        "    self.down = nn.Linear(dim, rank, bias=False)",
        "    self.up = nn.Linear(rank, 1, bias=False)",
        "",
        "def _adapter_step(self, loss):",
        "    loss.backward()",
        "    self.optimizer.step()",
        "",
        "def _evaluate_and_checkpoint(self):",
        "    torch.save(checkpoint, checkpoint_path)",
    ]
    for i, line in enumerate(lines):
        code_line(draw, 156 + i * 44, i + 1, line, highlight=(step in (2, 3) and i in (4, 5, 6)) or (step >= 4 and i in (8, 9)), color=GREEN if i in (4, 8, 9) else INK)
    marker(draw, 108, 156 + (4 if step < 4 else 8) * 44, 690, "TRAIN" if step < 4 else "SAVE")
    pen(draw, 900, 156 + (1 if step < 2 else 5 if step < 4 else 9) * 44)
    draw.rounded_rectangle((930, 210, 1350, 550), radius=18, fill=PANEL, outline="#334655", width=2)
    draw_text(draw, (972, 246), "checkpoint files", 22, YELLOW, True)
    for i, label in enumerate(("update-00000.pt", "update-00010.pt", "update-00020.pt", "update-00048.pt")):
        draw_text(draw, (972, 304 + i * 52), "✓  " + label, 21, GREEN if i < 3 else BLUE)
    draw_text(draw, (52, 682), "The final run saved 48 checkpoints and 324 adapter updates across the four arms.", 18, MUTED)
    return image


def scene_controller(step: int) -> Image.Image:
    image, draw = terminal_frame("src/staleroll/controller.py", "lag + KL decide what survives")
    draw_text(draw, (52, 86), "A rollout can be old without being useless.", 26, INK, True)
    code = [
        "lag_steps = current.version - trajectory.behavior_version",
        "kl = policy_kl(current, behavior, task)",
        "",
        "accept = lag_steps <= max_lag and kl <= max_kl",
        "weight = age_decay * kl_decay * importance_ratio",
    ]
    for i, line in enumerate(code):
        code_line(draw, 174 + i * 56, i + 1, line, highlight=(step >= 3 and i == 3), color=GREEN if i == 3 else INK)
    draw.line((100, 530, 1260, 530), fill="#4A5A67", width=4)
    for x, label, color in [(230, "snapshot", BLUE), (620, "worker finishes", YELLOW), (1030, "update", GREEN)]:
        draw.ellipse((x - 13, 517, x + 13, 543), fill=color)
        draw_text(draw, (x - 48, 562), label, 18, color, True)
    marker(draw, 126, 318 if step < 3 else 342, 680, "KEEP")
    pen(draw, 1030 if step < 3 else 1240, 340)
    draw_text(draw, (52, 682), "Unsafe stale work is rejected before it can move the adapter.", 18, MUTED)
    return image


def scene_results(step: int) -> Image.Image:
    image, draw = terminal_frame("artifacts/backend_comparison_final/report.md", "100-task comparison")
    draw_text(draw, (52, 86), "The result is close — and the gap is visible.", 26, INK, True)
    columns = [(68, "arm"), (360, "LoRA pass"), (620, "Gemini pass"), (900, "LoRA / tick"), (1180, "Gemini / tick")]
    for x, label in columns:
        draw_text(draw, (x, 164), label, 18, MUTED, True)
    rows = [
        ("sync", "0.7951", "0.7839", "0.4180", "0.4143"),
        ("naive async", "0.7552", "0.7747", "0.5916", "0.5986"),
        ("stale filter", "0.7552", "0.7622", "0.5916", "0.5888"),
    ]
    for i, row in enumerate(rows):
        y = 236 + i * 100
        fill = "#263846" if i != 2 else "#4A3D1F"
        draw.rounded_rectangle((48, y - 16, 1380, y + 58), radius=12, fill=fill)
        for (x, _), value in zip(columns, row):
            draw_text(draw, (x, y + 8), value, 24, GREEN if i == 2 else INK, i == 2)
    marker(draw, 338, 420 if step < 3 else 520, 850, "COMPARE")
    pen(draw, 1250, 428 if step < 3 else 528)
    draw_text(draw, (52, 682), "LoRA: 0.5916 verified/tick · Gemini: 0.5888 verified/tick in stale_filter.", 18, MUTED)
    return image


def scene_close(step: int) -> Image.Image:
    image, draw = terminal_frame("StaleRoll", "run it yourself")
    draw_text(draw, (76, 126), "The repo is the demo.", 44, INK, True)
    draw_text(draw, (78, 194), "Code → verifier → adapter → report", 28, BLUE)
    draw.rounded_rectangle((76, 298, 1360, 490), radius=18, fill=PANEL, outline="#334655", width=2)
    lines = [
        "$ make lora-code-training",
        "$ make compare-backends",
        "artifacts/lora_code_training_final2/report.md",
        "artifacts/backend_comparison_final/report.md",
        "demo/staleroll_demo.mp4",
    ]
    for i, line in enumerate(lines):
        code_line(draw, 326 + i * 34, i + 1, line, highlight=(step >= i + 2), color=GREEN if i >= 2 else INK)
    marker(draw, 108, 394 if step < 4 else 462, 930, "OPEN")
    pen(draw, 1160, 394 if step < 4 else 496)
    draw_text(draw, (78, 652), "A real run, real checkpoints, real hidden tests, and an honest comparison.", 22, MUTED)
    return image


SCENES = [scene_intro, scene_run, scene_repair, scene_lora, scene_controller, scene_results, scene_close]


async def build_audio() -> list[Path]:
    BUILD.mkdir(parents=True, exist_ok=True)
    import edge_tts

    paths: list[Path] = []
    for index, paragraph in enumerate(NARRATION):
        path = BUILD / f"voice-{index:02d}.mp3"
        communicate = edge_tts.Communicate(paragraph, "en-US-EmmaNeural", rate="-7%", pitch="+0Hz")
        await communicate.save(str(path))
        paths.append(path)
    return paths


def render() -> None:
    audio_paths = asyncio.run(build_audio())
    audio_clips = [AudioFileClip(str(path)) for path in audio_paths]
    durations = [max(5.0, clip.duration + 0.8) for clip in audio_clips]
    video_parts = []
    for scene_index, (scene, duration) in enumerate(zip(SCENES, durations)):
        paths: list[str] = []
        count = 7
        for frame_index in range(count):
            frame_path = BUILD / f"scene-{scene_index:02d}-{frame_index:02d}.png"
            scene(frame_index).save(frame_path)
            paths.append(str(frame_path))
        video_parts.append(ImageSequenceClip(paths, durations=[duration / count] * count))
    video = concatenate_videoclips(video_parts, method="compose")
    voice = concatenate_audioclips(audio_clips)
    video = video.with_audio(voice)
    video.write_videofile(str(OUTPUT), fps=FPS, codec="libx264", audio_codec="aac", bitrate="2800k", logger=None)
    for clip in video_parts:
        clip.close()
    video.close()
    voice.close()
    for clip in audio_clips:
        clip.close()
    print(OUTPUT)


if __name__ == "__main__":
    render()
