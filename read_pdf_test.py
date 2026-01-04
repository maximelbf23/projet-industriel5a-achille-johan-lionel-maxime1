import pypdf
import os

files = ["resolution_mécanique_5A.pdf", "ProjectEstaca.pdf"]

for file in files:
    print(f"--- CONTENT OF {file} ---")
    try:
        reader = pypdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Print first 500 chars to verify
        print(text[:500] + "...\n[Truncated]\n")
        print(f"Total characters read: {len(text)}")
    except Exception as e:
        print(f"Error reading {file}: {e}")
    print("-" * 30 + "\n")
