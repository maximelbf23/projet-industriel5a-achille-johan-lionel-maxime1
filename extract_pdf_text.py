import pypdf
import os

pdf_files = [
    "ProjectEstaca Achille.pdf",
    "02_Projet_25-26_5A-IDSA_MAT_AVattré.pdf"
]

output_file = "pdf_content.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            f.write(f"--- START OF {pdf_file} ---\n")
            try:
                reader = pypdf.PdfReader(pdf_file)
                for page_num, page in enumerate(reader.pages):
                    f.write(f"--- Page {page_num + 1} ---\n")
                    f.write(page.extract_text())
                    f.write("\n")
            except Exception as e:
                f.write(f"Error reading {pdf_file}: {e}\n")
            f.write(f"--- END OF {pdf_file} ---\n\n")
        else:
            f.write(f"File not found: {pdf_file}\n")

print(f"Text extracted to {output_file}")
