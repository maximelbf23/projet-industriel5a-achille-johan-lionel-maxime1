"""
Extract key theoretical sections from PDFs for verification.
"""
import pypdf

def extract_pages(filename, start, end):
    reader = pypdf.PdfReader(filename)
    text = ""
    for i in range(start-1, min(end, len(reader.pages))):
        text += f"\n--- Page {i+1} ---\n"
        text += reader.pages[i].extract_text()
    return text

# Extract key sections from resolution_mécanique_5A.pdf
print("=" * 70)
print("DOCUMENT 1: resolution_mécanique_5A.pdf")
print("=" * 70)

# Pages 5-7 contain the M matrix definition and characteristic equation
mech_pdf = extract_pages("resolution_mécanique_5A.pdf", 5, 7)
print(mech_pdf)

print("\n" + "=" * 70)
print("DOCUMENT 2: ProjectEstaca.pdf - Sections pertinentes")
print("=" * 70)

# Extract from ProjectEstaca
estaca_pdf = extract_pages("ProjectEstaca.pdf", 3, 5)
print(estaca_pdf)
