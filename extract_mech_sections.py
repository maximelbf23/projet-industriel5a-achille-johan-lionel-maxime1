import pypdf

filename = "resolution_mécanique_5A.pdf"
try:
    reader = pypdf.PdfReader(filename)
    print(f"--- Analysis of {filename} ---")
    # Read pages 10-14 for steps 8, 9, and Annexe
    for i in range(9, 14):
        if i < len(reader.pages):
            print(f"\n--- Page {i+1} ---")
            text = reader.pages[i].extract_text()
            print(text)
            print("-" * 20)
except Exception as e:
    print(e)
