import sys
import os

def test_imports():
    print("Testing Python imports...")
    
    try:
        import flask
        print("✓ Flask installed")
    except ImportError:
        print("✗ Flask not installed")
        return False
    
    try:
        import flask_cors
        print("✓ Flask-CORS installed")
    except ImportError:
        print("✗ Flask-CORS not installed")
        return False
    
    try:
        import pytesseract
        print("✓ Pytesseract installed")
    except ImportError:
        print("✗ Pytesseract not installed")
        return False
    
    try:
        import pdf2image
        print("✓ PDF2Image installed")
    except ImportError:
        print("✗ PDF2Image not installed")
        return False
    
    try:
        import cv2
        print("✓ OpenCV installed")
    except ImportError:
        print("✗ OpenCV not installed")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow installed")
    except ImportError:
        print("✗ Pillow not installed")
        return False
    
    return True

def test_tesseract():
    print("\nTesting Tesseract OCR...")
    
    # Common Tesseract paths on Windows
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            print(f"✓ Tesseract found at: {path}")
            return True
    
    print("✗ Tesseract OCR not found in common paths")
    print("   Please install Tesseract OCR from:")
    print("   https://github.com/UB-Mannheim/tesseract/wiki")
    return False

def test_directories():
    print("\nTesting directory structure...")
    
    required_dirs = [
        'uploads',
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name} directory exists")
        else:
            print(f"✓ Creating {dir_name} directory...")
            os.makedirs(dir_name, exist_ok=True)
    
    return True

def main():
    print("=== Backend Setup Test ===\n")
    
    success = True
    success &= test_imports()
    success &= test_tesseract()
    success &= test_directories()
    
    print("\n=== Test Results ===")
    if success:
        print("✓ All tests passed! Backend is ready to run.")
        print("\nYou can now start the backend with: python app.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nTip: Make sure you're running this from the activated virtual environment.")
    
    print("\nPress Enter to continue...")
    input()

if __name__ == "__main__":
    main()