"""
Module 6 — Flask Backend
Audio -> Whisper STT -> Speaker Diarization -> Groq Summarization
Run:  python server.py
"""

import os, json, time, threading, tempfile, hashlib, secrets, re
import smtplib
from email.message import EmailMessage
from datetime import datetime
from functools import wraps

import numpy as np
from flask import Flask, request, jsonify, send_from_directory, session, redirect
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

# ─── Persistent Secret Key ─────────────────────────────────────────────────────────
SECRET_KEY_FILE = "secret.key"
if os.path.exists(SECRET_KEY_FILE):
    with open(SECRET_KEY_FILE) as _f:
        app.secret_key = _f.read().strip()
else:
    _key = secrets.token_hex(32)
    with open(SECRET_KEY_FILE, "w") as _f:
        _f.write(_key)
    app.secret_key = _key

# ─── Data Storage ─────────────────────────────────────────────────────────────────
USERS_FILE = "users.json"
HISTORY_FILE = "history.json"
USER_DATA_DIR = "user_data"

os.makedirs(USER_DATA_DIR, exist_ok=True)

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_email" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        return f(*args, **kwargs)
    return decorated

# ─── Global Pipeline State ────────────────────────────────────────────────────
pipeline = {
    "status":            "idle",   # idle|preprocessing|transcribing|diarizing|merging|summarizing|done|error
    "log":               [],
    "transcript_text":   "",
    "stt_words":         [],
    "hyp_segments":      [],
    "merged_transcript": [],
    "summary":           "",
    "error":             "",
    "duration":          0.0,
    "running":           False,
}
pipeline_lock = threading.Lock()

PROMPT_TEMPLATES = {
    "full": """You are a professional meeting summarizer. Return a structured summary in EXACTLY this format:

**MEETING SUMMARY**

**Overview:**
[2-3 sentence overview]

**Key Points:**
- [point]

**Decisions Made:**
- [decision]

**Action Items:**
| Owner | Task | Deadline |
|-------|------|----------|
| [name] | [task] | [deadline or TBD] |

**Next Steps:**
- [step]

TRANSCRIPT:
{transcript}""",

    "keypoints": """Extract the KEY POINTS from this meeting transcript as a clear bullet list.

**Key Points:**
- [point]

TRANSCRIPT:
{transcript}""",

    "actions": """Extract all ACTION ITEMS from this meeting transcript.

| Owner | Task | Deadline |
|-------|------|----------|
| [name] | [task] | [deadline] |

TRANSCRIPT:
{transcript}""",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    with pipeline_lock:
        pipeline["log"].append(entry)
    print(entry)

def set_status(s: str):
    with pipeline_lock:
        pipeline["status"] = s

def reset_pipeline():
    with pipeline_lock:
        pipeline["status"]            = "idle"
        pipeline["log"]               = []
        pipeline["transcript_text"]   = ""
        pipeline["stt_words"]         = []
        pipeline["hyp_segments"]      = []
        pipeline["merged_transcript"] = []
        pipeline["summary"]           = ""
        pipeline["error"]             = ""
        pipeline["duration"]          = 0.0
        pipeline["running"]           = False

# ─── Pipeline Workers ─────────────────────────────────────────────────────────

def preprocess_audio(audio_bytes: bytes, suffix: str) -> str | None:
    try:
        import torch, torchaudio
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            raw_path = f.name

        waveform, orig_sr = torchaudio.load(raw_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_sr, 16000)(waveform)

        out_path = raw_path.replace(suffix, "_16k.wav")
        torchaudio.save(out_path, waveform, 16000)

        duration = waveform.shape[1] / 16000
        with pipeline_lock:
            pipeline["duration"] = duration
        log(f"Audio ready — {duration:.1f}s, mono 16kHz")
        return out_path
    except Exception as e:
        log(f"Preprocess error: {e}")
        return None


def stt_worker(path: str, model_size: str, done_event: threading.Event):
    """
    FIX: done_event is now passed in (not a global), so each pipeline
    run gets a fresh event — no stale-set from a previous run.
    """
    import whisper
    try:
        log(f"Whisper: loading '{model_size}' model...")
        model  = whisper.load_model(model_size)
        log("Whisper: transcribing audio...")
        result = model.transcribe(path, word_timestamps=True)

        text  = result["text"].strip()
        words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                words.append({
                    "word":  w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end":   round(w["end"],   3),
                })
        with pipeline_lock:
            pipeline["transcript_text"] = text
            pipeline["stt_words"]       = words
        log(f"STT done — {len(text.split())} words, {len(words)} word-timestamps")
    except Exception as e:
        log(f"STT error: {e}")
    finally:
        done_event.set()


def smooth_labels(labels, window=5):
    smoothed = labels.copy()
    for i in range(len(labels)):
        s = max(0, i - window // 2)
        e = min(len(labels), i + window // 2 + 1)
        smoothed[i] = np.bincount(labels[s:e]).argmax()
    return smoothed


def diarization_worker(path: str, num_speakers: int, done_event: threading.Event):
    try:
        # Imports are INSIDE try so any ImportError is caught
        # and done_event.set() in finally always runs.
        import torch, torchaudio
        from speechbrain.inference.speaker import EncoderClassifier
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize

        log("Diarization: loading ECAPA-TDNN encoder...")
        device  = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        encoder.eval()

        waveform, sr = torchaudio.load(path)
        waveform_np  = waveform.squeeze().numpy()

        win_samples = int(0.5 * sr)
        hop_samples = int(0.25 * sr)
        if win_samples >= len(waveform_np):
            win_samples = max(1, len(waveform_np) // 2)
            hop_samples = max(1, win_samples // 2)

        windows, wtimes = [], []
        i = 0
        while i + win_samples <= len(waveform_np):
            windows.append(waveform_np[i:i + win_samples])
            wtimes.append({"start": round(i / sr, 3), "end": round((i + win_samples) / sr, 3)})
            i += hop_samples

        if not windows:
            dur = len(waveform_np) / sr
            with pipeline_lock:
                pipeline["hyp_segments"] = [{"speaker": "speaker1", "start": 0.0, "end": dur}]
            log("Audio too short — single speaker assigned")
            return

        log(f"Diarization: extracting embeddings for {len(windows)} windows...")
        embeddings = []
        with torch.no_grad():
            for chunk in windows:
                t = torch.tensor(chunk).unsqueeze(0).float().to(device)
                embeddings.append(encoder.encode_batch(t).squeeze().cpu().numpy())
        embeddings = np.array(embeddings)

        k      = min(num_speakers, len(windows))
        emb_n  = normalize(embeddings)
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(emb_n)
        labels = smooth_labels(labels)

        segs = []
        seg_start, seg_label = wtimes[0]["start"], labels[0]
        for j in range(1, len(wtimes)):
            if labels[j] != seg_label:
                segs.append({"speaker": f"speaker{seg_label+1}", "start": seg_start, "end": wtimes[j-1]["end"]})
                seg_start, seg_label = wtimes[j]["start"], labels[j]
        segs.append({"speaker": f"speaker{seg_label+1}", "start": seg_start, "end": wtimes[-1]["end"]})

        merged = []
        for seg in segs:
            if seg["end"] - seg["start"] < 0.5 and merged:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(dict(seg))

        with pipeline_lock:
            pipeline["hyp_segments"] = merged
        log(f"Diarization done — {len(merged)} turns, {k} speaker(s)")
    except Exception as e:
        log(f"Diarization error: {e}")
    finally:
        done_event.set()


def merge_stt_diarization():
    with pipeline_lock:
        hyp   = list(pipeline["hyp_segments"])
        words = list(pipeline["stt_words"])
        dur   = pipeline["duration"]

    # If diarization failed entirely, create one segment covering full audio
    if not hyp and words:
        log("WARNING: Diarization produced no segments — assigning all words to speaker1")
        hyp = [{"speaker": "speaker1", "start": 0.0, "end": dur or words[-1]["end"]}]
        with pipeline_lock:
            pipeline["hyp_segments"] = hyp

    result = []
    for seg in hyp:
        # Use word midpoint for matching — robust against timing misalignment
        w_in = [
            w["word"] for w in words
            if seg["start"] <= (w["start"] + w["end"]) / 2 <= seg["end"]
        ]
        result.append({
            "speaker": seg["speaker"],
            "start":   seg["start"],
            "end":     seg["end"],
            "text":    " ".join(w_in).strip() or "[inaudible]",
        })

    # Fallback: if ALL segments are still inaudible but we have words,
    # distribute words evenly across segments
    all_inaudible = all(s["text"] == "[inaudible]" for s in result)
    if all_inaudible and words and result:
        log("WARNING: All segments inaudible after merge — distributing words by segment")
        n = len(result)
        chunk = max(1, len(words) // n)
        for idx, seg in enumerate(result):
            w_slice = words[idx * chunk : (idx + 1) * chunk]
            seg["text"] = " ".join(w["word"] for w in w_slice).strip() or "[inaudible]"
        leftover = words[n * chunk :]
        if leftover and result:
            result[-1]["text"] += " " + " ".join(w["word"] for w in leftover)
            result[-1]["text"] = result[-1]["text"].strip()

    with pipeline_lock:
        pipeline["merged_transcript"] = result

    non_inaudible = sum(1 for s in result if s["text"] != "[inaudible]")
    log(f"Merge done — {len(result)} segments, {non_inaudible} with speech")


def summarization_worker(api_key: str, summary_type: str):
    """
    FIX 1: Read merged_transcript under lock into a local variable.
    FIX 2: Check for empty/all-inaudible transcript before calling API.
    FIX 3: Use llama-3.3-70b-versatile — faster and smarter than 3.1-8b-instant
            for summarization quality. Falls back to 3.1-8b-instant on quota error.
    FIX 4: Detailed error logging so you can see exactly what Groq returns.
    """
    from groq import Groq

    with pipeline_lock:
        merged = list(pipeline["merged_transcript"])

    # Build labelled transcript — skip inaudible segments
    labelled_lines = [
        f"{s['speaker']}: {s['text']}"
        for s in merged
        if s["text"] != "[inaudible]"
    ]

    if not labelled_lines:
        # Fallback: use raw transcript text if diarization gave nothing useful
        with pipeline_lock:
            raw_text = pipeline["transcript_text"]
        if raw_text.strip():
            log("WARNING: No diarized segments — falling back to raw STT transcript for summary")
            labelled_lines = [f"speaker1: {raw_text.strip()}"]
        else:
            log("ERROR: No transcribed speech to summarize — transcript is empty!")
            with pipeline_lock:
                pipeline["error"] = "Summarization skipped: transcript is empty."
            return

    labelled = "\n".join(labelled_lines)
    log(f"Summarization: sending {len(labelled.split())} words to Groq...")
    log(f"Transcript preview: {labelled[:200]}...")  # helps debug content issues

    prompt = PROMPT_TEMPLATES.get(summary_type, PROMPT_TEMPLATES["full"]).format(
        transcript=labelled
    )

    client = Groq(api_key=api_key, max_retries=0)

    # Use lightning fast 8B model first for rapid generation
    models_to_try = [
        ("llama-3.1-8b-instant",     "LLaMA-3.1 8B (Fast)"),
        ("llama-3.3-70b-versatile",  "LLaMA-3.3 70B (Fallback)"),
    ]

    for model_id, model_name in models_to_try:
        try:
            log(f"Summarization: calling Groq {model_name}...")
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            summary = resp.choices[0].message.content.strip()
            if not summary:
                log(f"WARNING: Groq returned an empty summary from {model_name}")
                continue

            with pipeline_lock:
                pipeline["summary"] = summary
            log(f"Summary generated successfully via {model_name}!")
            return  # success — exit

        except Exception as e:
            err_str = str(e)
            log(f"Summarization error with {model_name}: {err_str}")
            # If rate-limited or quota exceeded, try next model
            if any(code in err_str for code in ["429", "rate_limit", "quota", "overloaded"]):
                log("Rate limit hit — trying fallback model...")
                continue
            else:
                # Non-retryable error (bad API key, network, etc.)
                with pipeline_lock:
                    pipeline["error"] = f"Groq error: {err_str}"
                return

    # All models failed
    with pipeline_lock:
        pipeline["error"] = "All Groq models failed. Check your API key and rate limits."
    log("ERROR: Summarization failed — all models exhausted.")


def run_pipeline(audio_bytes, suffix, model_size, num_speakers, summary_type, user_email):
    """
    FIX: stt_done_event and diarize_done_event are now created fresh each run
         as local variables and passed into the workers. This prevents a stale
         .set() from a previous run causing workers to be skipped instantly.
    """
    stt_done_event     = threading.Event()
    diarize_done_event = threading.Event()

    try:
        # Step 1 — Preprocess
        set_status("preprocessing")
        log("Preprocessing audio...")
        path = preprocess_audio(audio_bytes, suffix)
        if not path:
            set_status("error")
            with pipeline_lock:
                pipeline["error"]   = "Audio preprocessing failed"
                pipeline["running"] = False
            return

        # Step 2 — STT + Diarization in parallel
        set_status("transcribing")
        log("Launching STT + Diarization in parallel...")
        t_stt = threading.Thread(
            target=stt_worker,
            args=(path, model_size, stt_done_event),
            daemon=True,
        )
        t_dia = threading.Thread(
            target=diarization_worker,
            args=(path, num_speakers, diarize_done_event),
            daemon=True,
        )
        t_stt.start()
        t_dia.start()

        while not stt_done_event.is_set() or not diarize_done_event.is_set():
            time.sleep(0.3)
            if stt_done_event.is_set() and not diarize_done_event.is_set():
                set_status("diarizing")
            elif not stt_done_event.is_set() and diarize_done_event.is_set():
                set_status("transcribing")

        t_stt.join()
        t_dia.join()

        # Step 3 — Merge
        set_status("merging")
        log("Merging STT + speaker labels...")
        merge_stt_diarization()

        # Sanity check after merge
        with pipeline_lock:
            merged_count = len(pipeline["merged_transcript"])
            non_inaudible = sum(
                1 for s in pipeline["merged_transcript"]
                if s["text"] != "[inaudible]"
            )
        log(f"Merge result: {merged_count} total segments, {non_inaudible} with speech")

        # Step 4 — Summarize
        set_status("summarizing")
        users = load_users()
        user_data = users.get(user_email, {})
        api_key = user_data.get("api_key", "").strip()

        key_preview = (api_key[:8] + "...") if api_key else "(empty)"
        log(f"API key check: {key_preview}")
        
        if api_key:
            log("Calling summarization_worker...")
            summarization_worker(api_key, summary_type)
            log(f"summarization_worker returned. summary length={len(pipeline.get('summary',''))}")
        else:
            log("No Groq API key set — skipping summarization")
            log("Tip: Set your Groq API key in Settings!")

        set_status("done")
        log("Pipeline complete!")
        
        # --- Save to History ---
        with pipeline_lock:
            if pipeline["transcript_text"] or pipeline["merged_transcript"]:
                meeting_id = secrets.token_hex(8)
                now_str = datetime.now().isoformat()
                
                meeting_data = {
                    "id": meeting_id,
                    "timestamp": now_str,
                    "duration": pipeline["duration"],
                    "transcript_text": pipeline["transcript_text"],
                    "merged_transcript": list(pipeline["merged_transcript"]),
                    "summary": pipeline["summary"]
                }
                
                filepath = os.path.join(USER_DATA_DIR, f"{meeting_id}.json")
                with open(filepath, "w") as f:
                    json.dump(meeting_data, f)
                
                history = load_history()
                if user_email not in history:
                    history[user_email] = []
                
                sum_preview = (pipeline["summary"][:150] + "...") if pipeline["summary"] else "No summary available."
                
                history[user_email].append({
                    "id": meeting_id,
                    "timestamp": now_str,
                    "duration": pipeline["duration"],
                    "summary_preview": sum_preview
                })
                save_history(history)
                
        # Moved completely outside lock to prevent fatal deadlock
        if pipeline["transcript_text"] or pipeline["merged_transcript"]:
            log(f"Saved to history: {meeting_id}")

    except Exception as e:
        log(f"Pipeline error: {e}")
        set_status("error")
        with pipeline_lock:
            pipeline["error"] = str(e)
    finally:
        with pipeline_lock:
            pipeline["running"] = False

# ─── Flask Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def landing():
    """Public landing / intro page."""
    return send_from_directory("static", "landing.html")


@app.route("/login")
def login_page():
    """Login / register page."""
    return send_from_directory("static", "login.html")


@app.route("/app")
def app_page():
    """Main transcriber app — requires authentication."""
    if "user_email" not in session:
        return redirect("/login")
    return send_from_directory("static", "index.html")


@app.route("/api/start", methods=["POST"])
@require_auth
def start():
    if pipeline["running"]:
        return jsonify({"error": "Pipeline already running"}), 400

    user_email   = session.get("user_email")
    audio_file   = request.files.get("audio")
    model_size   = request.form.get("model_size",   "base")
    num_speakers = int(request.form.get("num_speakers", 2))
    summary_type = request.form.get("summary_type", "full")

    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    reset_pipeline()
    with pipeline_lock:
        pipeline["running"] = True

    suffix      = os.path.splitext(audio_file.filename)[1] or ".wav"
    audio_bytes = audio_file.read()

    t = threading.Thread(
        target=run_pipeline,
        args=(audio_bytes, suffix, model_size, num_speakers, summary_type, user_email),
        daemon=True,
    )
    t.start()
    return jsonify({"message": "Pipeline started"})


@app.route("/api/status")
def status():
    with pipeline_lock:
        return jsonify({
            "status":            pipeline["status"],
            "log":               pipeline["log"][-80:],
            "transcript_text":   pipeline["transcript_text"],
            "merged_transcript": pipeline["merged_transcript"],
            "summary":           pipeline["summary"],
            "error":             pipeline["error"],
            "duration":          pipeline["duration"],
            "running":           pipeline["running"],
        })


@app.route("/api/stop", methods=["POST"])
def stop():
    reset_pipeline()
    return jsonify({"message": "Pipeline reset"})


@app.route("/api/download/<file_type>")
def download(file_type):
    from flask import Response

    with pipeline_lock:
        merged  = list(pipeline["merged_transcript"])
        summary = pipeline["summary"]
        transcript_text = pipeline["transcript_text"]
        hyp_segments    = list(pipeline["hyp_segments"])

    if file_type == "transcript":
        lines = []
        for s in merged:
            if s["text"] == "[inaudible]":
                continue
            m_s   = int(s["start"] // 60);  sec_s = s["start"] % 60
            m_e   = int(s["end"]   // 60);  sec_e = s["end"]   % 60
            lines.append(f"{s['speaker']} [{m_s:02d}:{sec_s:05.2f} --> {m_e:02d}:{sec_e:05.2f}]")
            lines.append(f"  {s['text']}\n")
        content = "\n".join(lines)
        return Response(
            content, mimetype="text/plain",
            headers={"Content-Disposition": "attachment;filename=transcript.txt"},
        )

    elif file_type == "summary":
        if not summary:
            return jsonify({"error": "No summary available yet"}), 404
        return Response(
            summary, mimetype="text/markdown",
            headers={"Content-Disposition": "attachment;filename=summary.md"},
        )

    elif file_type == "json":
        data = json.dumps({
            "timestamp":         datetime.now().isoformat(),
            "transcript_text":   transcript_text,
            "merged_transcript": merged,
            "hyp_segments":      hyp_segments,
            "summary":           summary,
        }, indent=2)
        return Response(
            data, mimetype="application/json",
            headers={"Content-Disposition": "attachment;filename=meeting_output.json"},
        )

    return jsonify({"error": "Unknown file type"}), 400


# ─── Feature Routes ───────────────────────────────────────────────────────────

@app.route("/api/settings", methods=["GET", "POST"])
@require_auth
def settings():
    user_email = session.get("user_email")
    users = load_users()
    user = users.get(user_email, {})
    
    if request.method == "GET":
        has_key = bool(user.get("api_key"))
        return jsonify({"has_api_key": has_key})
        
    if request.method == "POST":
        data = request.get_json() or {}
        api_key = data.get("api_key", "").strip()
        user["api_key"] = api_key
        save_users(users)
        return jsonify({"message": "Settings saved"})

@app.route("/api/settings/verify", methods=["POST"])
@require_auth
def verify_api_key():
    from groq import Groq
    data = request.get_json() or {}
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"valid": False, "error": "No API key provided"})
        
    try:
        client = Groq(api_key=api_key)
        client.models.list()
        return jsonify({"valid": True})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})

@app.route("/api/history", methods=["GET"])
@require_auth
def get_history():
    user_email = session.get("user_email")
    history = load_history()
    return jsonify({"history": history.get(user_email, [])})

@app.route("/api/history/<meeting_id>", methods=["GET"])
@require_auth
def get_meeting(meeting_id):
    user_email = session.get("user_email")
    history = load_history()
    user_meetings = history.get(user_email, [])
    if not any(m["id"] == meeting_id for m in user_meetings):
        return jsonify({"error": "Not found or unauthorized"}), 403
        
    filepath = os.path.join(USER_DATA_DIR, f"{meeting_id}.json")
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
        
    with open(filepath) as f:
        return jsonify(json.load(f))

@app.route("/api/history/<meeting_id>", methods=["DELETE"])
@require_auth
def delete_meeting(meeting_id):
    user_email = session.get("user_email")
    history = load_history()
    user_meetings = history.get(user_email, [])
    
    idx = -1
    for i, m in enumerate(user_meetings):
        if m["id"] == meeting_id:
            idx = i
            break
            
    if idx == -1:
        return jsonify({"error": "Not found"}), 404
        
    del user_meetings[idx]
    save_history(history)
    
    filepath = os.path.join(USER_DATA_DIR, f"{meeting_id}.json")
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception:
            pass
            
    return jsonify({"message": "Meeting deleted"})

@app.route("/api/chat", methods=["POST"])
@require_auth
def chat_with_meeting():
    from groq import Groq
    user_email = session.get("user_email")
    data = request.get_json() or {}
    meeting_id = data.get("meeting_id")
    question = data.get("question")
    
    if not meeting_id or not question:
        return jsonify({"error": "Missing parameters"}), 400
        
    history = load_history()
    user_meetings = history.get(user_email, [])
    if not any(m["id"] == meeting_id for m in user_meetings):
        return jsonify({"error": "Unauthorized"}), 403
        
    users = load_users()
    api_key = users.get(user_email, {}).get("api_key")
    if not api_key:
        return jsonify({"error": "Groq API key not configured in Settings"}), 400
        
    filepath = os.path.join(USER_DATA_DIR, f"{meeting_id}.json")
    try:
        with open(filepath) as f:
            meeting_data = json.load(f)
            
        transcript = ""
        for seg in meeting_data.get("merged_transcript", []):
            if seg["text"] != "[inaudible]":
                transcript += f"{seg['speaker']}: {seg['text']}\n"
                
        if not transcript:
            transcript = meeting_data.get("transcript_text", "")
            
        summary = meeting_data.get("summary", "No summary available.")
        
        # Build an advanced prompt for better AI personality and accuracy
        prompt = f"""You are 'MeetScribe AI', a highly intelligent and proactive assistant. 
Your goal is to help the user understand this specific meeting.

CONTEXT:
---------------------
SUMMARY: 
{summary}
---------------------
TRANSCRIPT:
{transcript[:20000]} 

INSTRUCTIONS:
1. Use both the SUMMARY and TRANSCRIPT to provide a detailed, accurate, and helpful answer.
2. Use Markdown (bolding, bullet points, and tables) to make your response extremely readable.
3. If you can't find a direct answer, summarize the most relevant parts of the meeting and offer to help with a follow-up.
4. Always be professional, efficient, and friendly.

USER QUESTION: {question}
"""

        client = Groq(api_key=api_key)
        # Using a balanced model and higher token count for richer answers
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": "You are a helpful meeting assistant that provides clear, structured answers based on provided meeting data."},
                      {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        return jsonify({"answer": resp.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── Auth Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/auth/me")
def auth_me():
    email = session.get("user_email")
    if email:
        return jsonify({"authenticated": True, "email": email})
    return jsonify({"authenticated": False})


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data     = request.get_json() or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or "@" not in email or "." not in email:
        return jsonify({"error": "Enter a valid email address"}), 400

    # Password rules: letters + numbers, min 6 chars
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if not re.search(r"[A-Za-z]", password):
        return jsonify({"error": "Password must contain at least one letter"}), 400
    if not re.search(r"\d", password):
        return jsonify({"error": "Password must contain at least one number"}), 400

    users = load_users()
    if email in users:
        return jsonify({"error": "Email already registered. Please log in."}), 400

    users[email] = {
        "password_hash": hash_password(password),
        "created_at":    datetime.now().isoformat(),
    }
    save_users(users)

    session["user_email"] = email
    return jsonify({"message": "Account created", "email": email})


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data     = request.get_json() or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    users = load_users()
    user  = users.get(email)
    if not user or user["password_hash"] != hash_password(password):
        return jsonify({"error": "Invalid email or password"}), 401

    session["user_email"] = email
    return jsonify({"message": "Login successful", "email": email})


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.pop("user_email", None)
    return jsonify({"message": "Logged out"})


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    print("\n🎙️  Meeting Transcriber Backend")
    print("=" * 40)
    print("Open your browser at:  http://localhost:5000")
    print("=" * 40 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)