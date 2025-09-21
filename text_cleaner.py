import re
import string
from typing import List, Dict

class TextCleaner:
    def __init__(self):
        # Common patterns to remove from legal documents
        self.noise_patterns = [
            r'\bPage\s+\d+\b',  # Page numbers
            r'\bpage\s+\d+\b',  # Page numbers (lowercase)
            r'\d+\s*of\s*\d+',  # Page x of y
            r'^\s*\d+\s*$',     # Lines with only numbers
            r'^\s*[-_=]+\s*$',  # Lines with only separators
            r'\s+',             # Multiple whitespace characters
        ]
        
        # Common header/footer patterns in legal documents
        self.header_footer_patterns = [
            r'^\s*confidential\s*$',
            r'^\s*privileged\s*$',
            r'^\s*attorney[- ]client\s+privilege\s*$',
            r'^\s*draft\s*$',
            r'^\s*copy\s*$',
            r'^\s*original\s*$',
        ]
    
    def remove_page_markers(self, text: str) -> str:
        """Remove page numbers and page markers"""
        for pattern in self.noise_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE | re.MULTILINE)
        return text
    
    def remove_headers_footers(self, text: str) -> str:
        """Remove common headers and footers"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_clean = line.strip().lower()
            
            # Skip common header/footer patterns
            is_header_footer = False
            for pattern in self.header_footer_patterns:
                if re.match(pattern, line_clean):
                    is_header_footer = True
                    break
            
            if not is_header_footer and line.strip():
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and line breaks"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace multiple line breaks with double line break
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Clean up leading/trailing whitespace on each line
        lines = [line.strip() for line in text.split('\n')]
        
        return '\n'.join(lines)
    
    def remove_special_characters(self, text: str) -> str:
        """Remove or replace problematic special characters"""
        # Replace common OCR artifacts
        replacements = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '—': '-',
            '–': '-',
            '…': '...',
            '§': 'Section',
            '¶': 'Paragraph',
        }
        
        for old_char, new_char in replacements.items():
            text = text.replace(old_char, new_char)
        
        # Remove other problematic characters but keep basic punctuation
        # Keep letters, numbers, basic punctuation, and whitespace
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\{\}\"\'\-\+\=\@\#\$\%\&\*\/\\]', ' ', text)
        
        return text
    
    def fix_case_issues(self, text: str) -> str:
        """Fix common case issues from OCR"""
        lines = text.split('\n')
        fixed_lines = []
        
        for line in lines:
            if line.strip():
                # If line is all caps and long, convert to title case
                if len(line.strip()) > 10 and line.strip().isupper():
                    line = line.title()
                
                # Fix common OCR case errors
                line = re.sub(r'\bi\b', 'I', line)  # Fix lowercase 'i' standing alone
                line = re.sub(r'\bllc\b', 'LLC', line, flags=re.IGNORECASE)
                line = re.sub(r'\binc\b', 'Inc', line, flags=re.IGNORECASE)
                line = re.sub(r'\bcorp\b', 'Corp', line, flags=re.IGNORECASE)
                
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def structure_paragraphs(self, text: str) -> str:
        """Improve paragraph structure"""
        lines = text.split('\n')
        structured_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line:
                # Add extra spacing before lines that look like headings or sections
                if (line.isupper() or 
                    re.match(r'^\d+\.', line) or 
                    re.match(r'^[A-Z][^.!?]*:$', line) or
                    any(keyword in line.upper() for keyword in ['WHEREAS', 'THEREFORE', 'ARTICLE', 'SECTION'])):
                    
                    if structured_lines and structured_lines[-1].strip():
                        structured_lines.append('')  # Add blank line before heading
                
                structured_lines.append(line)
        
        return '\n'.join(structured_lines)
    
    def extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract basic metadata from the document"""
        metadata = {}
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}\/\d{1,2}\/\d{4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',
            r'\b\w+\s+\d{1,2},\s+\d{4}\b',
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text))
        
        if dates:
            metadata['dates_found'] = dates[:5]  # Limit to first 5 dates
        
        # Extract potential parties (capitalized names/entities)
        party_pattern = r'\b[A-Z][A-Z\s&,\.]+(?:LLC|INC|CORP|LP|LTD)\b'
        parties = re.findall(party_pattern, text)
        
        if parties:
            metadata['potential_parties'] = list(set(parties[:10]))  # Unique parties, limit to 10
        
        # Extract amounts/numbers
        amount_pattern = r'\$[\d,]+(?:\.\d{2})?'
        amounts = re.findall(amount_pattern, text)
        
        if amounts:
            metadata['amounts_found'] = amounts[:10]  # Limit to first 10 amounts
        
        return metadata
    
    def clean_text(self, raw_text: str) -> str:
        """Main method to clean and normalize text"""
        if not raw_text or not raw_text.strip():
            return ""
        
        # Step 1: Remove page markers and noise
        text = self.remove_page_markers(raw_text)
        
        # Step 2: Remove headers and footers
        text = self.remove_headers_footers(text)
        
        # Step 3: Fix special characters
        text = self.remove_special_characters(text)
        
        # Step 4: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Step 5: Fix case issues
        text = self.fix_case_issues(text)
        
        # Step 6: Structure paragraphs
        text = self.structure_paragraphs(text)
        
        # Final cleanup
        text = text.strip()
        
        return text

    def normalize_for_highlighting(self, text: str) -> str:
        """Perform conservative normalization to improve substring matching for highlights.

        This removes non-printable characters, normalizes smart quotes and dashes,
        fixes common hyphenation across line breaks, and collapses extra whitespace.
        """
        if not text:
            return text

        # Remove non-printable/control characters except common whitespace
        text = ''.join(ch for ch in text if ch.isprintable() or ch in '\n\r\t')

        # Normalize smart quotes and common ligatures
        text = text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2013', '-').replace('\u2014', '-')

        # Fix hyphenation across line breaks: "exam-\nple" -> "example"
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Collapse multiple newlines to a single newline and trim spaces
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)

        return text.strip()
    
    def get_cleaning_report(self, original_text: str, cleaned_text: str) -> Dict:
        """Generate a report on the cleaning process"""
        original_lines = len(original_text.split('\n'))
        cleaned_lines = len(cleaned_text.split('\n'))
        
        original_words = len(original_text.split())
        cleaned_words = len(cleaned_text.split())
        
        metadata = self.extract_metadata(cleaned_text)
        
        return {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'reduction_percentage': round((1 - len(cleaned_text) / len(original_text)) * 100, 2) if len(original_text) > 0 else 0,
            'original_lines': original_lines,
            'cleaned_lines': cleaned_lines,
            'original_words': original_words,
            'cleaned_words': cleaned_words,
            'metadata': metadata
        }