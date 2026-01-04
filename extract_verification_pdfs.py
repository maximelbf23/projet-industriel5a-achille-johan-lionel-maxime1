import pypdf
import os
import glob
import sys

sys.stdout.reconfigure(encoding='utf-8')

files = [
    "02_Projet_25-26_5A-IDSA_MAT_AVattré.pdf",
    "ProjectEstaca.pdf"
]

output_file = "extracted_pdf_content.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for filename in files:
        f.write(f"\n\n{'='*50}\nCONTENT OF {filename}\n{'='*50}\n\n")
        
        target = filename
        if not os.path.exists(target):
            print(f"File {target} not found. Trying glob match...")
            base = filename[:10]
            candidates = glob.glob(f"*{base}*.pdf")
            if candidates:
                target = candidates[0]
                print(f"Found candidate: {target}")
            else:
                print(f"ERROR: Could not find {filename}")
                f.write(f"ERROR: Could not find {filename}\n")
                continue

        try:
            reader = pypdf.PdfReader(target)
            for i, page in enumerate(reader.pages):
                f.write(f"--- Page {i+1} ---\n")
                text = page.extract_text()
                if text:
                    text_preview = text[:200].replace('\n', ' ')
                    print(f"Page {i+1} preview: {text_preview}...")
                    f.write(text)
                f.write("\n\n")
            print(f"Extracted {len(reader.pages)} pages from {target}")
        except Exception as e:
            f.write(f"ERROR reading {target}: {e}\n")
            print(f"Error extracting {target}: {e}")

print(f"Extraction complete. Content saved to {output_file}")
