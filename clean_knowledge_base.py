import os
import re
import fitz  # PyMuPDF
import unicodedata

def clean_text(text):
    # Normalize unicode characters (e.g. smart quotes to standard quotes)
    text = unicodedata.normalize("NFKD", text)
    
    # Fix hyphenated words broken across lines (e.g. "trans-\naction" -> "transaction")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Remove specific disclaimers or boilerplates
    disclaimer_patterns = [
        r"Disclaimer\s*-\s*These FAQs are issued by the RBI.*",
        r"This document is for information and general guidance purposes only.*"
    ]
    for pattern in disclaimer_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    
    # Split text into lines
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Remove standalone page numbers or very short numeric lines
        if re.fullmatch(r'\d+', stripped) or re.fullmatch(r'Page \d+', stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(stripped)
    
    # Rejoin text
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Remove multiple consecutive blank lines (replace with a single newline)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

def process_file(src_path, dest_path):
    print(f"Processing {src_path}...")
    ext = os.path.splitext(src_path)[1].lower()
    
    try:
        raw_text = ""
        if ext == '.txt':
            with open(src_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
        elif ext == '.pdf':
            # Use PyMuPDF for better extraction
            with fitz.open(src_path) as doc:
                for page in doc:
                    raw_text += page.get_text() + "\n"
        else:
            print(f"Skipping unsupported file type: {src_path}")
            return
            
        cleaned = clean_text(raw_text)
        
        # Determine destination filename (ensure .txt extension for PDF outputs)
        base_name = os.path.basename(src_path)
        name_without_ext = os.path.splitext(base_name)[0]
        # handle multiple extensions like .txt.txt
        name_without_ext = name_without_ext.replace('.txt', '')
        final_dest = os.path.join(os.path.dirname(dest_path), f"{name_without_ext}.txt")
        
        with open(final_dest, 'w', encoding='utf-8') as f:
            f.write(cleaned)
            
        old_size = os.path.getsize(src_path)
        new_size = os.path.getsize(final_dest)
        print(f"Saved {final_dest} (Old size: {old_size} bytes -> New size: {new_size} bytes)")
        
    except Exception as e:
        print(f"Error processing {src_path}: {e}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(base_dir, 'knowledge_base')
    dest_dir = os.path.join(base_dir, 'knowledge_base_cleaned')
    
    if not os.path.exists(src_dir):
        print(f"Source directory not found: {src_dir}")
        return
        
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Reading from: {src_dir}")
    print(f"Writing cleaned text to: {dest_dir}\n")
    
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        if os.path.isfile(src_path):
            process_file(src_path, os.path.join(dest_dir, filename))
            
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
