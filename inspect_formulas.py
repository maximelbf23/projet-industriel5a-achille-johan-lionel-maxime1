import pypdf

filename = "resolution_mécanique_5A.pdf"
try:
    reader = pypdf.PdfReader(filename)
    print(f"--- Analysis of {filename} ---")
    # Read pages 1 to 3 (index 0 to 2)
    for i in range(min(3, len(reader.pages))):
        print(f"\n--- Page {i+1} ---")
        text = reader.pages[i].extract_text()
        print(text)
        print("-" * 20)
except Exception as e:
    print(e)
