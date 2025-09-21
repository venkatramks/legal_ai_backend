import os
import requests
import json
from typing import Optional, Dict, Any, List

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL environment variable must be set for Supabase operations.")

# Build two header sets: one for service_role (write) and one for anon (read)
_headers_service = None
_headers_anon = None

if SUPABASE_SERVICE_KEY:
    _headers_service = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

if SUPABASE_ANON_KEY:
    _headers_anon = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _get_headers(write: bool = False):
    """Return appropriate headers for read/write operations.

    If write=True and service key is not available, raises a RuntimeError.
    If write=False, prefer anon headers when present, otherwise fall back to service headers.
    """
    if write:
        if _headers_service:
            return _headers_service
        raise RuntimeError("SUPABASE_SERVICE_KEY is required for write operations (insert/update). Please set it in server env.")

    # read operation
    if _headers_anon:
        return _headers_anon
    if _headers_service:
        return _headers_service
    # If neither key is present, caller will see request failures; surface a clearer error
    raise RuntimeError("No Supabase key available for read operations. Set SUPABASE_ANON_KEY or SUPABASE_SERVICE_KEY in env.")


def insert_document(
    file_name: str,
    file_path: Optional[str] = None,
    file_size: Optional[int] = None,
    extracted_text: Optional[str] = None,
    cleaned_text: Optional[str] = None,
    ocr_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Insert a new document row or return existing by file_path if provided.

    Returns the inserted row dict.
    """
    payload = {
        "file_name": file_name,
        "file_path": file_path,
        "file_size": file_size,
        "extracted_text": extracted_text,
        "cleaned_text": cleaned_text,
        "ocr_metadata": ocr_metadata,
    }
    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}

    if file_path:
        # Try to find existing
        url = f"{SUPABASE_URL}/rest/v1/documents?file_path=eq.{file_path}&select=*"
        resp = requests.get(url, headers=_get_headers(write=False))
        if resp.status_code == 200 and resp.json():
            return resp.json()[0]

    # Insert new document
    url = f"{SUPABASE_URL}/rest/v1/documents"
    resp = requests.post(url, headers=_get_headers(write=True), json=payload)
    if resp.status_code not in [200, 201]:
        raise RuntimeError(f"Supabase insert document error: {resp.status_code} {resp.text}")
    return resp.json()[0]


def update_document(document_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{SUPABASE_URL}/rest/v1/documents?id=eq.{document_id}"
    resp = requests.patch(url, headers=_get_headers(write=True), json=updates)
    if resp.status_code not in [200, 204]:
        raise RuntimeError(f"Supabase update document error: {resp.status_code} {resp.text}")
    if resp.status_code == 204:
        # No content returned, fetch the updated record
        return get_document_by_id(document_id)
    return resp.json()[0] if resp.json() else get_document_by_id(document_id)


def get_document_by_id(document_id: str) -> Optional[Dict[str, Any]]:
    url = f"{SUPABASE_URL}/rest/v1/documents?id=eq.{document_id}&select=*"
    resp = requests.get(url, headers=_get_headers(write=False))
    if resp.status_code != 200:
        raise RuntimeError(f"Supabase get document error: {resp.status_code} {resp.text}")
    data = resp.json()
    return data[0] if data else None


def append_chat_message(document_id: str, role: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "document_id": document_id,
        "role": role,
        "message": message,
        "metadata": metadata,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    
    url = f"{SUPABASE_URL}/rest/v1/chats"
    resp = requests.post(url, headers=_get_headers(write=True), json=payload)
    if resp.status_code not in [200, 201]:
        raise RuntimeError(f"Supabase insert chat error: {resp.status_code} {resp.text}")
    return resp.json()[0]


def get_chats_for_document(document_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    url = f"{SUPABASE_URL}/rest/v1/chats?document_id=eq.{document_id}&select=*&order=created_at.asc&limit={limit}"
    resp = requests.get(url, headers=_get_headers(write=False))
    if resp.status_code != 200:
        raise RuntimeError(f"Supabase fetch chats error: {resp.status_code} {resp.text}")
    return resp.json() or []


def insert_clauses_bulk(document_id: str, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Insert multiple clause rows for a document. Clauses should be a list of dicts with keys:
    clause_text, clause_headline (optional), start_pos (opt), end_pos (opt), risk (opt), highlights (opt)
    Returns the inserted rows as a list.
    """
    if not clauses:
        return []

    payload = []
    for c in clauses:
        item = {
            'document_id': document_id,
            'clause_text': c.get('clause_text') or c.get('text') or c.get('clause') or '',
            'clause_headline': c.get('clause_headline') or (c.get('clause_text') or c.get('text', ''))[:200],
            'start_pos': c.get('start_pos'),
            'end_pos': c.get('end_pos'),
            'risk': c.get('risk'),
            'highlights': c.get('highlights'),
            # Optional fields to persist LLM outputs so we don't re-analyze on view
            'scenarios': c.get('scenarios'),
            'legal_references': c.get('legal_references')
        }
        # Remove None values
        payload.append({k: v for k, v in item.items() if v is not None})

    url = f"{SUPABASE_URL}/rest/v1/clauses"
    resp = requests.post(url, headers=_get_headers(write=True), json=payload)
    if resp.status_code in [200, 201]:
        return resp.json() or []

    # If insert failed, attempt one fallback: remove optional json fields and retry
    fallback_payload = []
    for item in payload:
        trimmed = {k: v for k, v in item.items() if k not in ('scenarios', 'legal_references')}
        fallback_payload.append(trimmed)

    resp2 = requests.post(url, headers=_get_headers(write=True), json=fallback_payload)
    if resp2.status_code in [200, 201]:
        return resp2.json() or []

    # If still failing, surface original error for debugging
    raise RuntimeError(f"Supabase insert clauses error: {resp.status_code} {resp.text} | fallback: {resp2.status_code} {resp2.text}")


def get_clauses_for_document(document_id: str) -> List[Dict[str, Any]]:
    url = f"{SUPABASE_URL}/rest/v1/clauses?document_id=eq.{document_id}&select=*&order=created_at.asc"
    resp = requests.get(url, headers=_get_headers(write=False))
    if resp.status_code != 200:
        raise RuntimeError(f"Supabase fetch clauses error: {resp.status_code} {resp.text}")
    return resp.json() or []


def delete_clauses_by_ids(clause_ids: List[str]) -> bool:
    """Delete clause rows by a list of clause primary keys (ids). Returns True on success."""
    if not clause_ids:
        return True
    # Build comma-separated list for Supabase 'in' filter. Values must be comma-separated and parenthesized.
    # Ensure values are URL-safe; for simplicity join assuming numeric or uuid strings without commas.
    ids_str = ','.join([str(i) for i in clause_ids])
    url = f"{SUPABASE_URL}/rest/v1/clauses?id=in.({ids_str})"
    resp = requests.delete(url, headers=_get_headers(write=True))
    if resp.status_code in (200, 204):
        return True
    raise RuntimeError(f"Supabase delete clauses error: {resp.status_code} {resp.text}")


def delete_chats_for_document(document_id: str) -> bool:
    """Delete all chat rows for a given document_id. Returns True if deletion succeeded."""
    url = f"{SUPABASE_URL}/rest/v1/chats?document_id=eq.{document_id}"
    resp = requests.delete(url, headers=_get_headers(write=True))
    if resp.status_code in (200, 204):
        return True
    raise RuntimeError(f"Supabase delete chats error: {resp.status_code} {resp.text}")


def delete_document_by_id(document_id: str) -> bool:
    """Delete a document row by id. Returns True if deletion succeeded."""
    url = f"{SUPABASE_URL}/rest/v1/documents?id=eq.{document_id}"
    resp = requests.delete(url, headers=_get_headers(write=True))
    if resp.status_code in (200, 204):
        return True
    raise RuntimeError(f"Supabase delete document error: {resp.status_code} {resp.text}")
