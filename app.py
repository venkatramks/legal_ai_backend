from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import uuid
import logging
from werkzeug.utils import secure_filename
import json
from dotenv import load_dotenv
# Support running as package (python -m backend.app) or as script (python app.py):
# Heavy OCR and image libraries are imported lazily to avoid build-time failures
# on serverless hosts that don't have system packages installed.
def import_services():
    """Return (OCRService, TextCleaner, LLMService) constructors or lightweight fallbacks."""
    LLMService = None
    OCRService = None
    TextCleaner = None

    try:
        # Prefer package-relative imports when available
        try:
            from .llm_service import LLMService as _LLM
            LLMService = _LLM
        except Exception:
            from llm_service import LLMService as _LLM
            LLMService = _LLM
    except Exception:
        # Provide a minimal fallback LLMService that reports unavailability
        class _LLMStub:
            def is_available(self):
                return False
            def analyze_document_type(self, text):
                return None
            def answer_user_query(self, *args, **kwargs):
                return "LLM not available in this environment"
            def get_legal_references(self, *a, **k):
                return []
            def get_what_if_scenarios(self, *a, **k):
                return []
        LLMService = _LLMStub

    try:
        try:
            from .ocr_service import OCRService as _OCR
            from .text_cleaner import TextCleaner as _TC
            OCRService = _OCR
            TextCleaner = _TC
        except Exception:
            from ocr_service import OCRService as _OCR
            from text_cleaner import TextCleaner as _TC
            OCRService = _OCR
            TextCleaner = _TC
    except Exception:
        # Provide lightweight stubs and a PDF/text-extraction fallback.
        # Try to use pdfminer.six for extracting text from PDFs and Pillow+pytesseract
        # for images if available. These imports are done lazily to avoid build-time
        # failures on serverless hosts where system binaries (tesseract) may be missing.
        try:
            from pdfminer.high_level import extract_text as pdf_extract_text
        except Exception:
            pdf_extract_text = None

        try:
            from PIL import Image
        except Exception:
            Image = None

        try:
            import pytesseract
        except Exception:
            pytesseract = None

        class _OCRFallback:
            def extract_text(self, path: str) -> str:
                # PDF extraction
                if pdf_extract_text and path.lower().endswith('.pdf'):
                    try:
                        return pdf_extract_text(path) or ""
                    except Exception:
                        return ""

                # Image extraction using pytesseract (if available and Tesseract binary installed)
                if Image is not None and pytesseract is not None and path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img = Image.open(path)
                        text = pytesseract.image_to_string(img)
                        return text or ""
                    except Exception:
                        return ""

                # Last resort: return empty string
                return ""

        class _TCStub:
            def clean_text(self, text):
                # Minimal cleaning: normalize whitespace
                if not text:
                    return ""
                return ' '.join(text.split())

        OCRService = _OCRFallback
        TextCleaner = _TCStub

    return OCRService, TextCleaner, LLMService
import logging
import threading
import time
import requests

try:
    # Try package-relative import first, then absolute import fallback so both run modes work
    try:
        from .supabase_client import insert_document, update_document, get_document_by_id, append_chat_message, get_chats_for_document
    except Exception:
        from supabase_client import insert_document, update_document, get_document_by_id, append_chat_message, get_chats_for_document
    SUPABASE_AVAILABLE = True
except Exception as e:
    logging.warning(f"Supabase client not available: {e}")
    SUPABASE_AVAILABLE = False

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Restrict CORS to a frontend origin if provided via FRONTEND_URL environment variable.
# If FRONTEND_URL is not set, keep CORS permissive for development convenience.
frontend_origin = os.getenv('https://legal-ai-frontend-two.vercel.app/')
if frontend_origin:
    # Only allow API routes from the configured frontend origin
    CORS(app, resources={r"/api/*": {"origins": frontend_origin}})
    logging.info(f"CORS restricted to frontend origin: {frontend_origin}")
else:
    CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Determine an upload folder that is writable in serverless environments.
# Use an explicit UPLOAD_FOLDER env var if present, otherwise prefer the system
# temporary directory when running on Vercel/serverless (writable) and fall
# back to a project-local 'uploads/' directory for local development.
DEFAULT_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if os.getenv('VERCEL') == '1' or os.getenv('SERVERLESS', '').lower() == 'true':
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or tempfile.gettempdir()
else:
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or DEFAULT_UPLOAD_DIR

# Only create the uploads directory when it's not the system temp dir (which
# already exists and is writable). This avoids attempting to create directories
# in read-only deployments.
try:
    if UPLOAD_FOLDER != tempfile.gettempdir():
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except Exception as e:
    logging.warning(f"Could not create upload directory {UPLOAD_FOLDER}: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Legal Document AI API is running'})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload PDF, PNG, JPG, or JPEG files.'}), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Save the file
        file.save(filepath)

        # If running in a serverless environment (Vercel) or the upload folder is
        # the system temp dir (ephemeral), local filesystem may not persist across
        # requests. To avoid 404s when the client calls /api/process/<file_id>,
        # process synchronously here and return the result.
        serverless_env = (
            os.getenv('VERCEL') == '1' or
            os.getenv('SERVERLESS', '').lower() == 'true' or
            UPLOAD_FOLDER == tempfile.gettempdir()
        )
        if serverless_env:
            try:
                # Lazy-import services and process immediately
                OCRServiceCls, TextCleanerCls, LLMServiceCls = import_services()
                ocr_service = OCRServiceCls()
                text_cleaner = TextCleanerCls()
                llm_service = LLMServiceCls()

                result, from_cache, document_id = process_and_cache_document(filepath, ocr_service, text_cleaner, llm_service)

                return jsonify({
                    'message': 'File uploaded and processed (serverless sync)',
                    'file_id': unique_filename,
                    'filename': filename,
                    'result': result,
                    'cached': from_cache,
                    'document_id': document_id
                }), 200
            except Exception as e:
                # Processing failed, but upload succeeded — return upload result and error
                logging.error(f"Synchronous processing failed on upload: {e}")
                return jsonify({
                    'message': 'File uploaded but processing failed',
                    'file_id': unique_filename,
                    'filename': filename,
                    'error': str(e)
                }), 202

        return jsonify({
            'message': 'File uploaded successfully',
            'file_id': unique_filename,
            'filename': filename
        })
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

def process_and_cache_document(filepath, ocr_service, text_cleaner, llm_service):
    """Process a document with OCR, cleaning, and LLM generation, using Supabase DB cache to avoid repeated LLM calls.

    Returns a tuple (result_dict, from_cache_bool, document_id)
    """
    file_name = os.path.basename(filepath)
    file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    
    # Try to find existing document in Supabase first
    document_row = None
    if SUPABASE_AVAILABLE:
        try:
            document_row = insert_document(
                file_name=file_name,
                file_path=filepath,
                file_size=file_size
            )
            
            # If document has already been processed, return cached result
            if document_row.get('processed_at') and document_row.get('cleaned_text'):
                result = {
                    'file_id': file_name,
                    'raw_text': document_row.get('extracted_text', ''),
                    'cleaned_text': document_row.get('cleaned_text', ''),
                    'document_type': document_row.get('ocr_metadata', {}).get('document_type') if document_row.get('ocr_metadata') else None,
                    'guidance': None,  # Will add chat-based guidance later
                    'llm_available': llm_service.is_available(),
                    'statistics': {
                        'raw_length': len(document_row.get('extracted_text', '')),
                        'cleaned_length': len(document_row.get('cleaned_text', '')),
                        'reduction_percentage': 0
                    }
                }
                
                if result['statistics']['raw_length'] > 0:
                    try:
                        result['statistics']['reduction_percentage'] = round((1 - result['statistics']['cleaned_length'] / result['statistics']['raw_length']) * 100, 2)
                    except Exception:
                        result['statistics']['reduction_percentage'] = 0
                
                return result, True, document_row['id']
        except Exception as e:
            logging.warning(f"Failed to check Supabase for existing document: {e}")

    # Fall back to file-based cache if Supabase unavailable
    result_path = filepath + ".processed.json"
    if not SUPABASE_AVAILABLE and os.path.exists(result_path):
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            return cached, True, None
        except Exception as e:
            logging.warning(f"Failed to read cache {result_path}, will reprocess. Error: {e}")
            try:
                os.remove(result_path)
            except Exception:
                pass

    # Step 1: OCR
    raw_text = ocr_service.extract_text(filepath)

    # Step 2: Clean text
    cleaned_text = text_cleaner.clean_text(raw_text) if raw_text else ""

    # Step 3: Analyze document type
    document_type = None
    if llm_service.is_available() and cleaned_text:
        try:
            document_type = llm_service.analyze_document_type(cleaned_text)
        except Exception as e:
            logging.error(f"Document type analysis failed: {e}")

    # Prepare result structure
    result = {
        'file_id': file_name,
        'raw_text': raw_text,
        'cleaned_text': cleaned_text,
        'document_type': document_type,
        'guidance': None,  # Will be provided via chat interface
        'llm_available': llm_service.is_available(),
        'statistics': {
            'raw_length': len(raw_text) if raw_text else 0,
            'cleaned_length': len(cleaned_text) if cleaned_text else 0,
            'reduction_percentage': 0
        }
    }

    if result['statistics']['raw_length'] > 0:
        try:
            result['statistics']['reduction_percentage'] = round((1 - result['statistics']['cleaned_length'] / result['statistics']['raw_length']) * 100, 2)
        except Exception:
            result['statistics']['reduction_percentage'] = 0

    # Update Supabase with processed results
    document_id = None
    if SUPABASE_AVAILABLE and document_row:
        try:
            ocr_metadata = {
                'document_type': document_type,
                'statistics': result['statistics']
            }
            
            updated_row = update_document(document_row['id'], {
                'extracted_text': raw_text,
                'cleaned_text': cleaned_text,
                'ocr_metadata': ocr_metadata,
                'processed_at': 'now()'
            })
            document_id = updated_row['id']
            
            # Previously we auto-generated an initial guidance message here. Per user request,
            # do not generate or append any assistant guidance automatically. The chat endpoint
            # will answer user questions on demand using the document context.
                    
        except Exception as e:
            logging.warning(f"Failed to update Supabase document: {e}")

    # Save file cache as backup if Supabase failed
    if not SUPABASE_AVAILABLE or not document_id:
        try:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to write cache file {result_path}: {e}")

    return result, False, document_id

@app.route('/api/process/<file_id>', methods=['GET', 'POST'])
def process_document(file_id):
    try:
        # Support GET to allow users/browsers to view process status by visiting
        # `/api/process/<file_id>` in a browser. Delegate to the existing
        # `process_status` logic so GET behaves like a convenience status check.
        if request.method == 'GET':
            return process_status(file_id)
        filepath = os.path.join(UPLOAD_FOLDER, file_id)

        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        result_path = filepath + '.processed.json'
        processing_marker = filepath + '.processing'

        # If cached result exists, return it immediately
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                return jsonify({'result': cached, 'cached': True}), 200
            except Exception as e:
                logging.warning(f"Failed to read cache {result_path}, will reprocess. Error: {e}")
                try:
                    os.remove(result_path)
                except Exception:
                    pass

        # If processing is already in progress for this file, return 202
        if os.path.exists(processing_marker):
            return jsonify({'status': 'processing', 'message': 'Processing already in progress for this file'}), 202

        # Create processing marker and start background processing to avoid duplicate work
        try:
            with open(processing_marker, 'w', encoding='utf-8') as pm:
                json.dump({'started_at': time.time()}, pm)
        except Exception as e:
            logging.warning(f"Failed to create processing marker {processing_marker}: {e}")

        # Initialize services (lazy import to avoid import-time failures)
        OCRServiceCls, TextCleanerCls, LLMServiceCls = import_services()
        ocr_service = OCRServiceCls()
        text_cleaner = TextCleanerCls()
        llm_service = LLMServiceCls()

        def _bg_process():
            try:
                result, from_cache, document_id = process_and_cache_document(filepath, ocr_service, text_cleaner, llm_service)
                logging.info(f"Processing completed for {filepath}, cached: {from_cache}, document_id: {document_id}")
            except Exception as e:
                logging.error(f"Background processing failed for {filepath}: {e}")
            finally:
                # Remove processing marker
                try:
                    if os.path.exists(processing_marker):
                        os.remove(processing_marker)
                except Exception as e:
                    logging.warning(f"Failed to remove processing marker {processing_marker}: {e}")
        
        # In serverless environments (like Vercel), background threads and local disk
        # persistence are unreliable. Detect common serverless env vars and run
        # processing synchronously in that case so the request lifecycle completes
        # only after processing finishes (or you can adapt to enqueue a job).
        serverless_env = os.getenv('VERCEL') == '1' or os.getenv('SERVERLESS', '').lower() == 'true'
        if serverless_env:
            try:
                result, from_cache, document_id = process_and_cache_document(filepath, ocr_service, text_cleaner, llm_service)
                logging.info(f"(sync) Processing completed for {filepath}, cached: {from_cache}, document_id: {document_id}")
            except Exception as e:
                logging.error(f"(sync) Processing failed for {filepath}: {e}")
            finally:
                try:
                    if os.path.exists(processing_marker):
                        os.remove(processing_marker)
                except Exception as e:
                    logging.warning(f"Failed to remove processing marker {processing_marker}: {e}")
        else:
            thread = threading.Thread(target=_bg_process, daemon=True)
            thread.start()

        return jsonify({'status': 'started', 'message': 'Processing has started in background'}), 202

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/api/process/status/<file_id>', methods=['GET'])
def process_status(file_id):
    try:
        filepath = os.path.join(UPLOAD_FOLDER, file_id)
        result_path = filepath + '.processed.json'
        processing_marker = filepath + '.processing'

        # If a processed cached result exists, return it even if the original
        # uploaded file is no longer present (common on serverless hosts).
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                return jsonify({'status': 'done', 'cached': True, 'result': cached}), 200
            except Exception as e:
                logging.warning(f"Failed to read cached result {result_path}: {e}")
                return jsonify({'status': 'error', 'message': 'Failed to read cached result'}), 500

        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        # Check Supabase first if available (use anon key for read if present)
        if SUPABASE_AVAILABLE:
            try:
                # Use the supabase client helper to fetch by file_path via REST
                try:
                    from .supabase_client import _get_headers
                except Exception:
                    from supabase_client import _get_headers
                headers = _get_headers(write=False)
                url = f"{os.getenv('SUPABASE_URL')}/rest/v1/documents?file_path=eq.{filepath}&select=*"
                resp = requests.get(url, headers=headers)
                if resp.status_code == 200 and resp.json():
                    doc = resp.json()[0]
                    if doc.get('processed_at'):
                        # Build result from Supabase data
                        result = {
                            'file_id': file_id,
                            'raw_text': doc.get('extracted_text', ''),
                            'cleaned_text': doc.get('cleaned_text', ''),
                            'document_type': doc.get('ocr_metadata', {}).get('document_type') if doc.get('ocr_metadata') else None,
                            'guidance': None,  # Will be fetched from chats
                            'llm_available': True,
                            'statistics': doc.get('ocr_metadata', {}).get('statistics', {})
                        }
                        return jsonify({'status': 'done', 'cached': True, 'result': result, 'document_id': doc['id']}), 200
            except Exception as e:
                logging.warning(f"Failed to check Supabase status (read): {e}")

        # Fall back to file-based check
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                return jsonify({'status': 'done', 'cached': True, 'result': cached}), 200
            except Exception as e:
                logging.warning(f"Failed to read cache {result_path}: {e}")
                return jsonify({'status': 'error', 'message': 'Failed to read cached result'}), 500

        if os.path.exists(processing_marker):
            return jsonify({'status': 'processing', 'message': 'Processing in progress'}), 200

        return jsonify({'status': 'pending', 'message': 'Processing has not started'}), 200

    except Exception as e:
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500


@app.route('/api/chat/<document_id>', methods=['POST'])
def chat_with_document(document_id):
    """Send a message to chat with a document"""
    try:
        if not SUPABASE_AVAILABLE:
            return jsonify({'error': 'Chat feature requires Supabase configuration'}), 503
            
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
            
        user_message = data['message']
        
        # Get document
        document = get_document_by_id(document_id)
        if not document:
            return jsonify({'error': 'Document not found'}), 404
            
        # Store user message
        append_chat_message(document_id, 'user', user_message)
        
        # Generate AI response if LLM available
        # Ensure LLM is obtained via lazy import
        _, _, LLMServiceCls = import_services()
        llm_service = LLMServiceCls()
        if llm_service.is_available() and document.get('cleaned_text'):
            try:
                # Get recent chat history for context
                chat_history = get_chats_for_document(document_id, limit=10)
                
                # Prepare document text and chat history for a document-aware answer
                document_text = document.get('cleaned_text') or document.get('extracted_text') or ''
                document_type = document.get('ocr_metadata', {}).get('document_type', 'legal document')

                # Build a short chat history string excluding the current user message
                chat_history_str = ''
                if chat_history and len(chat_history) > 1:
                    parts = []
                    for chat in chat_history[-6:-1]:  # Last 5 messages before current
                        role_label = 'User' if chat['role'] == 'user' else 'Assistant'
                        parts.append(f"{role_label}: {chat['message']}")
                    chat_history_str = '\n'.join(parts)

                # Ask the LLM to answer the user's specific question using the document context
                ai_response = llm_service.answer_user_query(user_message, document_text, document_type, chat_history_str)
                
                # Store AI response
                append_chat_message(document_id, 'assistant', ai_response)
                
                return jsonify({
                    'message': ai_response,
                    'document_id': document_id
                })
                
            except Exception as e:
                logging.error(f"Chat AI response failed: {e}")
                fallback_response = "I apologize, but I'm having trouble processing your request right now. Please try again."
                append_chat_message(document_id, 'assistant', fallback_response)
                return jsonify({
                    'message': fallback_response,
                    'document_id': document_id
                })
        else:
            fallback_response = "I'm unable to provide detailed analysis at the moment. Please ensure the document has been processed."
            append_chat_message(document_id, 'assistant', fallback_response)
            return jsonify({
                'message': fallback_response,
                'document_id': document_id
            })
            
    except Exception as e:
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500


@app.route('/api/chat/<document_id>/history', methods=['GET'])
def get_chat_history(document_id):
    """Get chat history for a document"""
    try:
        if not SUPABASE_AVAILABLE:
            return jsonify({'error': 'Chat feature requires Supabase configuration'}), 503
            
        chats = get_chats_for_document(document_id, limit=100)
        return jsonify({'chats': chats, 'document_id': document_id})
        
    except Exception as e:
        return jsonify({'error': f'Failed to get chat history: {str(e)}'}), 500


@app.route('/api/analysis/clauses/<document_id>', methods=['GET'])
def analysis_clauses(document_id):
    try:
        from analysis import extract_and_score_clauses_from_text

        # Fetch document from Supabase if available
        if not SUPABASE_AVAILABLE:
            return jsonify({'error': 'Analysis requires Supabase configuration'}), 503

        document = get_document_by_id(document_id)
        if not document:
            return jsonify({'error': 'Document not found'}), 404

        text = document.get('cleaned_text') or document.get('extracted_text') or ''
        if not text:
            return jsonify({'error': 'Document text unavailable for analysis'}), 400

        try:
            clauses = extract_and_score_clauses_from_text(text)
        except Exception as e:
            logging.warning(f"LLM-driven clause extraction failed, using heuristic fallback: {e}")
            try:
                # Import heuristic extractor (public helper)
                from analysis import heuristic_extract_and_score
                clauses = heuristic_extract_and_score(text)
            except Exception as e2:
                logging.error(f"Heuristic clause extraction also failed: {e2}")
                return jsonify({'error': 'Clause analysis failed and no fallback available'}), 500

        return jsonify({'clauses': clauses, 'document_id': document_id})

    except Exception as e:
        logging.error(f"Clause analysis failed: {e}")
        return jsonify({'error': f'Clause analysis failed: {str(e)}'}), 500


@app.route('/api/analysis/clauses/<document_id>/persisted', methods=['GET'])
def analysis_clauses_persisted(document_id):
    """Return persisted clauses for a document from Supabase (if available)."""
    try:
        if not SUPABASE_AVAILABLE:
            return jsonify({'error': 'Persisted clauses require Supabase configuration'}), 503

        try:
            try:
                from .supabase_client import get_clauses_for_document
            except Exception:
                from supabase_client import get_clauses_for_document

            clauses = get_clauses_for_document(document_id)
            return jsonify({'clauses': clauses, 'document_id': document_id}), 200
        except Exception as e:
            logging.error(f"Failed to fetch persisted clauses: {e}")
            return jsonify({'error': f'Failed to fetch persisted clauses: {str(e)}'}), 500

    except Exception as e:
        logging.error(f"Persisted clauses endpoint failed: {e}")
        return jsonify({'error': f'Persisted clauses endpoint failed: {str(e)}'}), 500


@app.route('/api/analysis/clauses/<document_id>/persist', methods=['POST'])
def analysis_clauses_persist(document_id):
    try:
        if not SUPABASE_AVAILABLE:
            return jsonify({'error': 'Persisting clauses requires Supabase configuration'}), 503

        document = get_document_by_id(document_id)
        if not document:
            return jsonify({'error': 'Document not found'}), 404

        text = document.get('cleaned_text') or document.get('extracted_text') or ''
        if not text:
            return jsonify({'error': 'Document text unavailable for analysis'}), 400


        # If client provided clause payload (including scenarios/legal_references), prefer persisting that
        data = request.get_json() or {}
        client_clauses = data.get('clauses')

        if client_clauses and isinstance(client_clauses, list):
            clauses_to_persist = client_clauses
        else:
            # Extract and score clauses (LLM or heuristic)
            from analysis import extract_and_score_clauses_from_text
            clauses_to_persist = extract_and_score_clauses_from_text(text)

        # Persist clauses (may include scenarios & legal_references keys)
        from supabase_client import insert_clauses_bulk
        inserted = insert_clauses_bulk(document_id, clauses_to_persist)

        return jsonify({'inserted': inserted, 'count': len(inserted), 'document_id': document_id})

    except Exception as e:
        logging.error(f"Persist clause analysis failed: {e}")
        return jsonify({'error': f'Persist clause analysis failed: {str(e)}'}), 500


@app.route('/api/analysis/clauses/<document_id>/undo', methods=['POST'])
def analysis_clauses_undo(document_id):
    """Undo persisted clauses for a document by clause IDs. Expects JSON body: {"clause_ids": ["id1", "id2"]}

    This performs a delete on the `clauses` table for the provided ids. Requires Supabase write key.
    """
    try:
        if not SUPABASE_AVAILABLE:
            return jsonify({'error': 'Undo requires Supabase configuration'}), 503

        data = request.get_json() or {}
        clause_ids = data.get('clause_ids') or []

        if not clause_ids:
            return jsonify({'error': 'clause_ids required'}), 400

        try:
            try:
                from .supabase_client import delete_clauses_by_ids
            except Exception:
                from supabase_client import delete_clauses_by_ids

            deleted = delete_clauses_by_ids(clause_ids)
            return jsonify({'deleted': deleted, 'deleted_ids': clause_ids}), 200
        except Exception as e:
            logging.error(f"Failed to delete clauses via helper: {e}")
            return jsonify({'error': f'Failed to delete clauses: {str(e)}'}), 500

    except Exception as e:
        logging.error(f"Undo endpoint failed: {e}")
        return jsonify({'error': f'Undo failed: {str(e)}'}), 500


@app.route('/api/documents', methods=['GET'])
def list_documents():
    """Get list of all processed documents"""
    try:
        if not SUPABASE_AVAILABLE:
            return jsonify({'error': 'Document list requires Supabase configuration'}), 503
            
    # Use anon or service headers for read
        try:
            try:
                from .supabase_client import _get_headers
            except Exception:
                from supabase_client import _get_headers
            headers = _get_headers(write=False)
        except Exception as e:
            logging.warning(f"Supabase headers unavailable for read: {e}")
            return jsonify({'error': 'Supabase configuration missing for reads'}), 503

        # Get all processed documents, ordered by most recent
        url = f"{os.getenv('SUPABASE_URL')}/rest/v1/documents?select=id,file_name,created_at,processed_at,ocr_metadata&processed_at=not.is.null&order=created_at.desc"
        resp = requests.get(url, headers=headers)

        if resp.status_code != 200:
            return jsonify({'error': 'Failed to fetch documents'}), 500

        documents = resp.json() or []
        return jsonify({'documents': documents})
        
    except Exception as e:
        return jsonify({'error': f'Failed to list documents: {str(e)}'}), 500


@app.route('/api/chats/<document_id>', methods=['DELETE'])
def delete_chats_for_document_endpoint(document_id):
    """Delete all chats for a document and optionally the document record itself."""
    try:
        if not SUPABASE_AVAILABLE:
            return jsonify({'error': 'Delete feature requires Supabase configuration'}), 503

        # Try to use supabase_client helper functions when available
        try:
            try:
                from .supabase_client import delete_chats_for_document, delete_document_by_id
            except Exception:
                from supabase_client import delete_chats_for_document, delete_document_by_id

            deleted = delete_chats_for_document(document_id)
            try:
                delete_document_by_id(document_id)
            except Exception:
                # If document delete fails, we still treat chats deletion as success
                pass
            return jsonify({'deleted': deleted, 'document_id': document_id}), 200

        except Exception as e:
            # Fallback: use REST API to remove chat rows from the `chats` table
            try:
                try:
                    from .supabase_client import _get_headers
                except Exception:
                    from supabase_client import _get_headers
                headers = _get_headers(write=True)
                url = f"{os.getenv('SUPABASE_URL')}/rest/v1/chats?document_id=eq.{document_id}"
                resp = requests.delete(url, headers=headers)
                if resp.status_code in (200, 204):
                    return jsonify({'deleted': True, 'document_id': document_id}), 200
                else:
                    return jsonify({'error': 'Failed to delete chats via REST', 'status_code': resp.status_code, 'body': resp.text}), 500
            except Exception as e2:
                logging.error(f"Delete chats failed: {e} / {e2}")
                return jsonify({'error': f'Delete chats failed: {str(e)}'}), 500

    except Exception as e:
        logging.error(f"Delete chats endpoint failed: {e}")
        return jsonify({'error': f'Delete chats failed: {str(e)}'}), 500

@app.route('/api/legal-knowledge-graph', methods=['POST'])
def get_legal_knowledge_graph():
    """Get relevant legal references for a clause"""
    try:
        data = request.get_json()
        clause_text = data.get('clauseText', '')
        document_type = data.get('documentType', '')
        clause_type = data.get('clauseType', '')
        
        if not clause_text:
            return jsonify({'error': 'Clause text is required'}), 400
            
        # Use LLM service to get relevant legal references
        # Ensure LLM is obtained via lazy import
        _, _, LLMServiceCls = import_services()
        llm_service = LLMServiceCls()
        legal_references = llm_service.get_legal_references(clause_text, document_type, clause_type)
        
        return jsonify({
            'references': legal_references,
            'clause_text': clause_text,
            'document_type': document_type,
            'clause_type': clause_type
        }), 200
        
    except Exception as e:
        logging.error(f"Legal knowledge graph failed: {e}")
        return jsonify({'error': f'Legal knowledge graph failed: {str(e)}'}), 500


@app.route('/api/debug/llm', methods=['GET'])
def debug_llm():
    """Return diagnostic info about LLM availability and environment variables."""
    try:
        # Lazy import the LLMService class
        _, _, LLMServiceCls = import_services()
        llm = LLMServiceCls()

        # Try to introspect llm_service module flags without importing at top-level
        try:
            try:
                from . import llm_service as _llm_mod
            except Exception:
                import llm_service as _llm_mod
            has_genai = bool(getattr(_llm_mod, '_HAS_GENAI', False) and getattr(_llm_mod, 'genai', None) is not None)
        except Exception:
            has_genai = False

        info = {
            'llm_is_available': bool(llm.is_available()),
            'google_api_key_present': bool(os.getenv('GOOGLE_API_KEY')),
            'has_genai_sdk': has_genai,
            'allow_llm_test': os.getenv('ALLOW_LLM_TEST', '0') == '1'
        }
        return jsonify(info), 200
    except Exception as e:
        logging.error(f"LLM debug failed: {e}")
        return jsonify({'error': f'LLM debug failed: {str(e)}'}), 500


@app.route('/api/debug/llm/test', methods=['POST'])
def debug_llm_test():
    """Optionally run a small LLM generation test if explicitly allowed.

    This endpoint will only perform a network/LLM call if ALLOW_LLM_TEST=1 in env to avoid
    accidental external calls from automated systems.
    """
    try:
        if os.getenv('ALLOW_LLM_TEST', '0') != '1':
            return jsonify({'error': 'LLM test not allowed. Set ALLOW_LLM_TEST=1 to enable.'}), 403

        data = request.get_json() or {}
        prompt = data.get('prompt') or 'Say hello in one short sentence.'

        _, _, LLMServiceCls = import_services()
        llm = LLMServiceCls()

        if not llm.is_available():
            return jsonify({'error': 'LLM not available in this environment'}), 503

        # Try a lightweight generation via the existing helper
        try:
            text = llm.generate_legal_guidance(prompt, document_type='test')
        except Exception as e:
            logging.error(f"LLM generation test failed: {e}")
            return jsonify({'error': f'LLM generation test failed: {str(e)}'}), 500

        return jsonify({'result': text}), 200
    except Exception as e:
        logging.error(f"LLM test endpoint failed: {e}")
        return jsonify({'error': f'LLM test endpoint failed: {str(e)}'}), 500

@app.route('/api/what-if-scenarios', methods=['POST'])
def get_what_if_scenarios():
    """Get what-if scenarios for a clause"""
    try:
        data = request.get_json()
        clause_text = data.get('clauseText', '')
        document_type = data.get('documentType', '')
        clause_type = data.get('clauseType', '')
        
        if not clause_text:
            return jsonify({'error': 'Clause text is required'}), 400
            
        # Use LLM service to get what-if scenarios
        # Ensure LLM is obtained via lazy import
        _, _, LLMServiceCls = import_services()
        llm_service = LLMServiceCls()
        scenarios = llm_service.get_what_if_scenarios(clause_text, document_type, clause_type)
        
        return jsonify({
            'scenarios': scenarios,
            'clause_text': clause_text,
            'document_type': document_type,
            'clause_type': clause_type
        }), 200
        
    except Exception as e:
        logging.error(f"What-if scenarios failed: {e}")
        return jsonify({'error': f'What-if scenarios failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(405)
def method_not_allowed(e):
    # Return JSON to avoid HTML 405 pages which confuse frontend polling or
    # users opening endpoints in a browser.
    return jsonify({'error': 'Method not allowed. Use POST to start processing or GET to check status.'}), 405

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Legal Document AI Backend...")
    # Running in production — print deployed URLs instead of localhost for clarity
    print("📍 API available at: https://legal-ai-backend-chi.vercel.app")
    print("📋 Health check: https://legal-ai-backend-chi.vercel.app/api/health")
    app.run(debug=True, host='0.0.0.0', port=5000)
