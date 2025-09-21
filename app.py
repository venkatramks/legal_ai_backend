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
try:
    from .ocr_service import OCRService
    from .text_cleaner import TextCleaner
    from .llm_service import LLMService
except Exception:
    # Fallback to absolute imports when running directly from the backend directory
    from ocr_service import OCRService
    from text_cleaner import TextCleaner
    from llm_service import LLMService
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
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

@app.route('/api/process/<file_id>', methods=['POST'])
def process_document(file_id):
    try:
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

        # Initialize services
        ocr_service = OCRService()
        text_cleaner = TextCleaner()
        llm_service = LLMService()

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
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        result_path = filepath + '.processed.json'
        processing_marker = filepath + '.processing'

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
        llm_service = LLMService()
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

        clauses = extract_and_score_clauses_from_text(text)
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
        llm_service = LLMService()
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
        llm_service = LLMService()
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

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Legal Document AI Backend...")
    print("📍 API available at: http://localhost:5000")
    print("📋 Health check: http://localhost:5000/api/health")
    app.run(debug=True, host='0.0.0.0', port=5000)