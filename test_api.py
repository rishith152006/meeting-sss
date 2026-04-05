"""
Direct API test — sends a real request to the running server to see what it receives.
Run while server.py is running.
"""
import requests
import wave, struct, math, tempfile, os

# Create a 2-second silent WAV file for testing
def make_test_wav():
    path = tempfile.mktemp(suffix=".wav")
    sample_rate = 16000
    duration    = 2
    frequency   = 440
    n_samples   = sample_rate * duration
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        # 440Hz tone
        for i in range(n_samples):
            val = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            f.writeframes(struct.pack("<h", val))
    return path

API_KEY = "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"

print("[1] Creating test audio file...")
wav_path = make_test_wav()
print(f"    Created: {wav_path}")

print("\n[2] Sending POST /api/start ...")
with open(wav_path, "rb") as f:
    resp = requests.post(
        "http://localhost:5000/api/start",
        files={"audio": ("test.wav", f, "audio/wav")},
        data={
            "api_key":      API_KEY,
            "model_size":   "tiny",
            "num_speakers": "1",
            "summary_type": "full",
        },
    )

print(f"    HTTP {resp.status_code}: {resp.text}")
os.remove(wav_path)

if resp.status_code == 200:
    print("\n[3] Pipeline started! Watch the server terminal for [DEBUG] lines.")
    print("    You can also poll: http://localhost:5000/api/status")
else:
    print("\n    ERROR - request was rejected before pipeline started.")
