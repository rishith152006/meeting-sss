"""
Quick Groq API test — run this to check if summarization works independently.
Usage:  python test_groq.py YOUR_GROQ_API_KEY
"""
import sys
import os

# Fix Windows console encoding
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

api_key = sys.argv[1] if len(sys.argv) > 1 else ""
if not api_key:
    api_key = input("Paste your Groq API key: ").strip()

print("\n[1] Importing groq library...")
try:
    from groq import Groq
    print("    OK - groq imported")
except ImportError as e:
    print(f"    FAIL - Import failed: {e}")
    print("    Fix: pip install groq")
    sys.exit(1)

print("\n[2] Creating Groq client...")
try:
    client = Groq(api_key=api_key)
    print("    OK - Client created")
except Exception as e:
    print(f"    FAIL - Client error: {e}")
    sys.exit(1)

test_prompt = "Summarize in one sentence: Alice said she will finish the report by Friday. Bob agreed to review it."

for model_id in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
    print(f"\n[3] Testing model: {model_id} ...")
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        summary = resp.choices[0].message.content.strip()
        print(f"    OK - Response: {summary}")
        print(f"\nSUCCESS: Groq API is working with model {model_id}")
        print("The issue is in server.py logic — check the Pipeline Log panel in the browser.\n")
        sys.exit(0)
    except Exception as e:
        print(f"    FAIL - Error: {e}")

print("\nFAIL: All models failed. Check your API key and network connection.\n")
sys.exit(1)
