import sys
import os

try: # ugh whats with c++ man
    import pdfplumber
    with pdfplumber.open(sys.argv[1]) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        print(text)
except ImportError:
    print("pdfplumber not installed", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error extracting PDF: {e}", file=sys.stderr)
    sys.exit(1)