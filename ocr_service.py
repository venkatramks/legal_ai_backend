import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import os
import tempfile
import numpy as np
import hashlib
import json

class OCRService:
    def __init__(self):
        # Set Tesseract path explicitly - try common installation paths
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', 'User')),
        ]
        
        # Try to find Tesseract executable
        tesseract_found = False
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                tesseract_found = True
                print(f"[*] Found Tesseract at: {path}")
                break
        
        if not tesseract_found:
            print("[!] Warning: Tesseract not found in common paths.")
            print("[!] Please install Tesseract OCR or update the path in ocr_service.py")
            # Still try with default path in case it's in PATH
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Initialize a set to track processed files
        self.processed_files = set()
        self.processed_files_path = "processed_files.json"
        self._load_processed_files()

    def _load_processed_files(self):
        """Load processed files from a JSON file."""
        if os.path.exists(self.processed_files_path):
            try:
                with open(self.processed_files_path, "r") as f:
                    self.processed_files = set(json.load(f))
                print("[*] Loaded processed files from disk.")
            except Exception as e:
                print(f"[!] Failed to load processed files: {e}")

    def _save_processed_files(self):
        """Save processed files to a JSON file."""
        try:
            with open(self.processed_files_path, "w") as f:
                json.dump(list(self.processed_files), f)
            print("[*] Saved processed files to disk.")
        except Exception as e:
            print(f"[!] Failed to save processed files: {e}")

    def preprocess_image(self, image_path):
        """
        Preprocess image to improve OCR accuracy
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply thresholding to get better contrast
            processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            return processed
        
        except Exception as e:
            print(f"Error in image preprocessing: {str(e)}")
            # Fallback to basic processing
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            processed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            return processed
    
    def extract_text_from_image(self, image_path):
        """
        Extract text from a single image using OCR
        """
        try:
            processed_image = self.preprocess_image(image_path)
            
            # Configure OCR settings for better accuracy
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            return text
        
        except Exception as e:
            print(f"Error extracting text from image {image_path}: {str(e)}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF by converting pages to images
        """
        try:
            print(f"[*] Converting PDF pages to images: {pdf_path}")
            
            # Convert PDF to images with high DPI for better quality
            pages = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=None)
            
            extracted_text = ""
            total_pages = len(pages)
            
            for i, page in enumerate(pages):
                print(f"[*] Processing page {i + 1}/{total_pages}...")
                
                # Create temporary image file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_image_path = temp_file.name
                    page.save(temp_image_path, 'PNG')
                
                try:
                    # Extract text from the page
                    page_text = self.extract_text_from_image(temp_image_path)
                    
                    if page_text.strip():
                        extracted_text += f"\n\n--- Page {i + 1} ---\n{page_text}"
                    else:
                        extracted_text += f"\n\n--- Page {i + 1} ---\n[No text detected on this page]"
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
            
            return extracted_text
        
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            return f"Error processing PDF: {str(e)}"
    
    def _generate_file_hash(self, file_path):
        """Generate a unique hash for the file based on its content."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def extract_text(self, file_path):
        """
        Main method to extract text from various file formats
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate a hash for the file
        file_hash = self._generate_file_hash(file_path)

        # Check if the file has already been processed
        if file_hash in self.processed_files:
            print(f"[*] File {file_path} has already been processed. Attempting to load cached content...")
            # Try to return cached extracted text if available
            result_path = f"{file_path}.processed.json"
            if os.path.exists(result_path):
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    # Prefer returning cleaned_text, fall back to raw_text
                    cached_text = cached.get('cleaned_text') or cached.get('raw_text')
                    if cached_text:
                        return cached_text
                    else:
                        # No useful cached text found, continue to re-extract
                        print(f"[!] Cache found but no text available in {result_path}, re-extracting...")
                except Exception as e:
                    print(f"[!] Failed to load cached result {result_path}: {e}. Will re-extract.")
            else:
                # No cache file found; continue to extract normally
                print(f"[!] No cache file {result_path} found for {file_path}. Will re-extract.")

        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                text = self.extract_text_from_image(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Mark the file as processed
            self.processed_files.add(file_hash)
            self._save_processed_files()
            return text

        except Exception as e:
            print(f"Error in text extraction: {str(e)}")
            return f"Error extracting text: {str(e)}"