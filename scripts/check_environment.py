import importlib.util
import sys

REQUIRED = ["flask", "flask_sqlalchemy", "flask_migrate", "flask_cors", "PIL", "cv2", "sklearn", "numpy"]
OPTIONAL = ["torch", "transformers", "whisper", "librosa"]

print(f"Python: {sys.version}")
for name in REQUIRED:
    print(f"{name}: {'OK' if importlib.util.find_spec(name) else 'MISSING'}")
for name in OPTIONAL:
    print(f"{name}: {'OK' if importlib.util.find_spec(name) else 'optional missing'}")
