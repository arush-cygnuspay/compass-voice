import time
from faster_whisper import WhisperModel

audio_path = "models/tts/speaker_0.mp3"

# -----------------------------
# 1) Load model
# -----------------------------
t0 = time.perf_counter()

model = WhisperModel(
    "small.en",
    device="cuda",
    compute_type="float16"
)

t1 = time.perf_counter()
model_load_time = t1 - t0

print(f"[INFO] Model loaded in {model_load_time:.4f} seconds")


# -----------------------------
# 2) Run transcription
# IMPORTANT:
# model.transcribe() returns a generator-like segments object.
# Actual decoding happens when we iterate over it.
# So we must force it into a list to measure true inference time.
# -----------------------------
t2 = time.perf_counter()

segments, info = model.transcribe(
    audio_path,
    beam_size=5
)

segments = list(segments)   # Force full inference here

t3 = time.perf_counter()
inference_time = t3 - t2

print(f"[INFO] True inference completed in {inference_time:.4f} seconds")


# -----------------------------
# 3) Convert segments into final text
# This is the text you can pass to another API
# -----------------------------
t4 = time.perf_counter()

final_text = " ".join(s.text.strip() for s in segments).strip()

t5 = time.perf_counter()
text_assembly_time = t5 - t4

print(f"[INFO] Text assembly completed in {text_assembly_time:.6f} seconds")


# -----------------------------
# 4) Total time from start of transcribe
# until final text is ready for next API
# -----------------------------
total_stt_to_text_time = t5 - t2

# -----------------------------
# 5) Print per-segment timestamps
# -----------------------------
print("\n[SEGMENTS]")
for i, s in enumerate(segments, start=1):
    print(f"{i:02d}. [{s.start:.2f}s -> {s.end:.2f}s] {s.text.strip()}")


# -----------------------------
# 6) Print final text
# -----------------------------
print("\n[FINAL TEXT]")
print(final_text)


# -----------------------------
# 7) Metrics
# -----------------------------
audio_duration = info.duration if info.duration else 0.0
rtf = (inference_time / audio_duration) if audio_duration > 0 else 0.0

print("\n[PERFORMANCE METRICS]")
print(f"Audio Duration               : {audio_duration:.4f} seconds")
print(f"Model Load Time             : {model_load_time:.4f} seconds")
print(f"True Inference Time         : {inference_time:.4f} seconds")
print(f'Text Assembly Time          : {text_assembly_time:.6f} seconds')
print(f"STT -> Final Text Ready     : {total_stt_to_text_time:.4f} seconds")
print(f"Real-Time Factor (RTF)      : {rtf:.4f}")

if audio_duration > 0 and inference_time > 0:
    speed_multiple = audio_duration / inference_time
    print(f"Processing Speed            : {speed_multiple:.2f}x faster than real-time")