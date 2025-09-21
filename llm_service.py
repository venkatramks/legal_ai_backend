import requests
import os
from typing import Optional, Dict, Any
import logging
from cachetools import TTLCache
_HAS_GENAI = False
genai = None  # type: ignore
try:
    # Try the canonical packaged import first
    from google import genai as _genai  # type: ignore
    genai = _genai
    _HAS_GENAI = True
except Exception:
    try:
        # Older / alternate packaging may expose genai at top-level
        import genai as _genai  # type: ignore
        genai = _genai
        _HAS_GENAI = True
    except Exception:
        # GenAI client not installed in this environment; fallbacks will be used
        genai = None  # type: ignore

class LLMService:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM service with Google GenAI configuration."""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            logging.warning("API key for Google GenAI not found in environment variables or constructor parameter")

        # Initialize a cache with a TTL of 300 seconds (5 minutes) and a max size of 100 entries
        self.cache = TTLCache(maxsize=100, ttl=300)

    def is_available(self) -> bool:
        """Check if the LLM service is available and configured."""
        return self.api_key is not None

    def generate_legal_guidance(self, document_text: str, document_type: str = "legal document") -> Optional[str]:
        """
        Generate legal guidance and simplified explanations for a legal document.
        
        Args:
            document_text (str): The extracted text from the legal document
            document_type (str): The type of document (e.g., "contract", "agreement", "legal notice")
        
        Returns:
            Optional[str]: Generated guidance or None if service unavailable
        """
        if not self.is_available():
            return self._get_fallback_guidance(document_type)

        # Check if the result is already cached
        cache_key = f"{document_type}:{hash(document_text)}"
        if cache_key in self.cache:
            logging.info("Returning cached result for the document")
            return self.cache[cache_key]

        # If GenAI client isn't available, return fallback guidance
        if not _HAS_GENAI or genai is None:
            logging.warning("GenAI client not available; returning fallback guidance")
            return self._get_fallback_guidance(document_type)

        try:
            system_prompt = self._get_system_prompt()
            user_prompt = self._get_user_prompt(document_text, document_type)

            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{system_prompt}\n\n{user_prompt}"
            )

            guidance = response.text.strip()
            logging.info("Successfully generated legal guidance")

            # Cache the result
            self.cache[cache_key] = guidance
            return guidance

        except Exception as e:
            logging.error(f"Error generating legal guidance: {e}")
            return self._get_fallback_guidance(document_type)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for legal document analysis."""
        return """You are a legal document analysis AI assistant designed to help non-lawyers understand complex legal documents. Your role is to:

1. Provide clear, plain-language explanations of legal terminology and concepts
2. Identify key clauses, obligations, and rights in the document
3. Highlight important dates, deadlines, and conditions
4. Explain potential implications and consequences
5. Suggest actionable next steps or considerations
6. Ensure your response does not exceed 10 lines of text

Guidelines:
- Use simple, accessible language that anyone can understand
- Avoid complex legal jargon unless necessary, and always explain it
- Be objective and informative, not advisory
- Focus on clarity and practical understanding
- Always include a disclaimer about consulting legal professionals
- Structure your response with clear sections and bullet points when helpful
- Limit your response to the most important and actionable insights
- Make sure that you dont include special characters in the response
Remember: You are providing educational information only, not legal advice."""

    def _get_user_prompt(self, document_text: str, document_type: str) -> str:
        """Create the user prompt with the document text."""
        # Truncate text if too long to fit within token limits
        max_text_length = 3000  # Conservatively estimate to leave room for response
        if len(document_text) > max_text_length:
            document_text = document_text[:max_text_length] + "... [Document truncated for analysis]"
        
        return f"""Please analyze this {document_type} and provide clear, actionable guidance for someone without legal training:

Document Text:
{document_text}

You are a legal assistant please answer the question by referring to the context of the document only.
Give your responses in a precise manner 
Provide source citations
Keep your response concise but comprehensive, focusing on the most important information a non-lawyer should understand."""

    def _get_fallback_guidance(self, document_type: str) -> str:
        """Provide fallback guidance when LLM service is unavailable."""
        return f"""**Document Analysis Complete**

This {document_type} has been processed and cleaned for better readability. While AI-powered guidance is currently unavailable, here are general recommendations:

**Key Actions:**
• Review the cleaned text carefully for important terms and conditions
• Look for dates, deadlines, and specific obligations
• Identify parties involved and their responsibilities
• Note any financial terms, penalties, or consequences

**Important Considerations:**
• Legal documents often contain time-sensitive information
• Pay attention to termination clauses and renewal terms
• Understand your rights and obligations under this document
• Consider how this document relates to your specific situation

**Next Steps:**
• Save this analysis for your records
• Consult with a qualified legal professional for specific advice
• Review any referenced documents or attachments
• Ensure you understand all terms before taking action

*Note: This is general guidance only. Always consult with a legal professional for advice specific to your situation.*"""

    def analyze_document_type(self, document_text: str) -> str:
        """
        Analyze the document text to determine its type.
        
        Args:
            document_text (str): The extracted text from the document
        
        Returns:
            str: Detected document type
        """
        # First, try to use the GenAI model to get a concise label if available
        try:
            if self.is_available() and _HAS_GENAI and genai is not None:
                client = genai.Client(api_key=self.api_key)
                prompt = (
                    "Classify the following legal document into a single concise type label. "
                    "Return only one short label (no explanation). Examples: 'Contract', 'Lease Agreement', "
                    "'Will', 'Privacy Policy', 'Terms of Service', 'Invoice', 'Legal Notice', 'NDA'.\n\n"
                    "Document Text:\n" + (document_text[:3000] if document_text else "")
                )
                try:
                    resp = client.models.generate_content(model="gemini-2.5", contents=prompt)
                except Exception:
                    # Some environments may only have flash model; try fallback
                    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

                label = (resp.text or '').strip()
                # Remove any markdown, quotes, or trailing punctuation
                if label:
                    label = label.split('\n')[0].strip().strip('"').strip("'")
                    # Normalize common variants
                    mapping = {
                        'contract': 'Contract', 'agreement': 'Contract', 'nda': 'NDA', 'non-disclosure agreement': 'NDA',
                        'lease': 'Lease Agreement', 'lease agreement': 'Lease Agreement',
                        'will': 'Will', 'testament': 'Will',
                        'privacy policy': 'Privacy Policy', 'terms of service': 'Terms of Service',
                        'invoice': 'Invoice', 'bill': 'Invoice', 'legal notice': 'Legal Notice',
                    }
                    key = label.lower()
                    mapped = mapping.get(key)
                    if mapped:
                        return mapped
                    # If label looks like a short single-word/phrase, return title-cased
                    if len(label.split()) <= 4:
                        return label.title()
        except Exception as e:
            # Fall back to keyword heuristics below
            logging.debug(f"GenAI document type classification failed: {e}")

        # Keyword-based fallback detection
        text_lower = (document_text or '').lower()
        if any(word in text_lower for word in ['contract', 'agreement', 'party', 'whereas']):
            return "Contract"
        elif any(word in text_lower for word in ['lease', 'tenant', 'landlord', 'rent']):
            return "Lease Agreement"
        elif any(word in text_lower for word in ['will', 'testament', 'heir', 'beneficiary']):
            return "Will"
        elif any(word in text_lower for word in ['privacy policy', 'data', 'cookies', 'personal information']):
            return "Privacy Policy"
        elif any(word in text_lower for word in ['terms of service', 'terms and conditions', 'user agreement']):
            return "Terms of Service"
        elif any(word in text_lower for word in ['invoice', 'bill', 'payment', 'due date']):
            return "Invoice"
        elif any(word in text_lower for word in ['court', 'lawsuit', 'plaintiff', 'defendant']):
            return "Legal Notice"
        else:
            return "Legal Document"

    def answer_user_query(self, user_query: str, document_text: str, document_type: str = "legal document", chat_history: Optional[str] = None) -> str:
        """
        Answer an arbitrary user query using the document context. This method should return a response focused on the user's question and the provided document content.

        - `user_query`: the user's question or instruction
        - `document_text`: the cleaned/extracted document text (may be truncated by this method)
        - `document_type`: optional label for the document
        - `chat_history`: optional recent chat history to include for context

        Returns a string answer (fallback text if LLM unavailable).
        """
        # If no LLM configured, provide a helpful fallback that points to the document
        if not self.is_available():
            return (
                "I'm currently unable to access the LLM service. "
                "I can still point you to relevant parts of the document if you paste them here, "
                "or you can persist the document and view clause highlights."
            )

        # If GenAI client not present, return fallback
        if not _HAS_GENAI or genai is None:
            logging.warning("GenAI client not available; returning fallback answer")
            return (
                "AI model unavailable. Please try again later or enable the GenAI client. "
                "You can also ask about specific clause text by pasting it here."
            )

        try:
            # Attempt to focus the answer on the most relevant clause(s).
            clause_to_use = None

            # Try to extract clauses using the LLM helper (if available). If extract_clauses
            # isn't possible, we'll fall back to a short document snippet.
            try:
                clauses = None
                if self.is_available() and _HAS_GENAI and genai is not None:
                    try:
                        clauses = self.extract_clauses(document_text)
                    except Exception:
                        clauses = None

                # If we got clause candidates, pick the clause with highest keyword overlap
                if clauses:
                    # Normalize
                    uq_words = set([w.lower() for w in user_query.split() if len(w) > 2])
                    best = None
                    best_score = 0
                    for c in clauses:
                        text = (c.get('text') or '')
                        words = set([w.lower().strip('.,()') for w in text.split() if len(w) > 2])
                        score = len(uq_words & words)
                        if score > best_score:
                            best_score = score
                            best = c
                    if best and best_score > 0:
                        clause_to_use = best.get('text')

            except Exception as e:
                logging.debug(f"Clause extraction step failed or skipped: {e}")

            # If no clause selected, fall back to a short document snippet (first 2000 chars)
            if not clause_to_use:
                snippet_len = 2000
                clause_to_use = (document_text or '')[:snippet_len]

            # Strict system prompt: instruct model to answer ONLY from the clause provided.
            system = (
                "You are a helpful assistant that answers questions strictly from the supplied clause text. "
                "Do NOT provide any additional general legal guidance or paragraphs. If the clause does not contain enough information to answer, reply exactly: 'Insufficient information in the clause to answer.' "
                "Do not invent facts or cite laws not present in the clause. Keep the answer short and directly tied to the clause."
            )

            # Compose user content with only the selected clause and the user's question
            user_content = f"Clause Text:\n{clause_to_use}\n\nUser Question:\n{user_query}"

            # Include a tiny chat history only if it might change interpretation
            if chat_history:
                user_content = f"Chat History:\n{chat_history}\n\n" + user_content

            full_prompt = system + "\n\n" + user_content

            # Use helper to create a client and generate content in a resilient way
            try:
                resp_text = self._generate_content_via_genai(full_prompt)
            except Exception as e:
                logging.error(f"GenAI generation failed: {e}")
                resp_text = None

            answer = (resp_text or '').strip()

            # If the model returns an empty or unhelpful answer, return a clear fallback
            if not answer:
                return "Insufficient information in the clause to answer."

            return answer
        except Exception as e:
            logging.error(f"Error answering user query via GenAI: {e}")
            return "I'm sorry — I couldn't process that question right now. Please try again or ask a more specific question."

    # ---- GenAI client helpers ----
    def _create_genai_client(self):
        """Create and return a GenAI client in a resilient way.

        Tries common construction patterns used across SDK versions.
        Returns the client instance or raises if not possible.
        """
        if not _HAS_GENAI or genai is None:
            raise RuntimeError("GenAI client not available")

        # Some SDK versions use genai.Client(api_key=...)
        try:
            return genai.Client(api_key=self.api_key)
        except Exception:
            pass

        # Other SDKs may provide a connect / from_api_key factory
        try:
            if hasattr(genai, 'Client'):
                client_cls = getattr(genai, 'Client')
                return client_cls(api_key=self.api_key)
        except Exception:
            pass

        # Last-ditch: return genai if it exposes a top-level convenience client
        if hasattr(genai, 'client'):
            return getattr(genai, 'client')

        raise RuntimeError("Unable to construct GenAI client with available SDK")

    def _generate_content_via_genai(self, prompt: str, models: Optional[list] = None, max_retries: int = 2) -> Optional[str]:
        """Attempt to generate content using GenAI, trying a few model names and client call styles.

        Returns response text or None on failure.
        """
        if models is None:
            models = ["gemini-2.5", "gemini-2.5-flash", "gemini-1.0"]

        # If SDK is available, prefer to use the client
        last_exc = None
        if _HAS_GENAI and genai is not None:
            client = self._create_genai_client()

            for model in models:
                try:
                    # Preferred style: client.models.generate_content(...)
                    if hasattr(client, 'models') and hasattr(client.models, 'generate_content'):
                        resp = client.models.generate_content(model=model, contents=prompt)
                        if hasattr(resp, 'text'):
                            return resp.text
                        if isinstance(resp, dict) and 'text' in resp:
                            return resp['text']

                    # Alternate style: client.generate(...) or client.generate_content(...)
                    if hasattr(client, 'generate'):
                        resp = client.generate(model=model, prompt=prompt)
                        if isinstance(resp, dict) and 'text' in resp:
                            return resp['text']
                        if hasattr(resp, 'text'):
                            return resp.text

                    if hasattr(client, 'generate_content'):
                        resp = client.generate_content(model=model, contents=prompt)
                        if hasattr(resp, 'text'):
                            return resp.text
                        if isinstance(resp, dict) and 'text' in resp:
                            return resp['text']

                except Exception as e:
                    last_exc = e
                    logging.debug(f"Model {model} failed with: {e}")
                    continue

        # If SDK usage didn't produce a result, and we have an API key, try REST fallback
        if self.api_key:
            try:
                rest_result = self._generate_via_rest(prompt, models=models)
                if rest_result:
                    return rest_result
            except Exception as e:
                logging.debug(f"REST GenAI fallback failed: {e}")

        # If all attempts fail, raise the last exception for callers to handle
        if last_exc:
            raise last_exc
        return None

    def _generate_via_rest(self, prompt: str, models: Optional[list] = None) -> Optional[str]:
        """Try to call a Google Generative Language REST endpoint with the API key.

        This is a pragmatic fallback when the GenAI SDK is not installed. It attempts a
        small set of likely endpoints and response formats. Returns text or None.
        """
        if models is None:
            models = ["text-bison-001", "text-bison", "gemini-2.5", "gemini-2.5-flash"]

        # Allow overriding base URL/model via environment for portability
        base_url_template = os.getenv('GOOGLE_GENAI_REST_URL') or "https://generativelanguage.googleapis.com/v1beta2/models/{model}:generateText"

        headers = {
            'Content-Type': 'application/json'
        }

        for model in models:
            try:
                url = base_url_template.format(model=model)
                # Use API key as query param if that's how user has access configured
                params = {'key': self.api_key}
                body = {
                    'prompt': {'text': prompt},
                    # conservative defaults
                    'temperature': 0.2,
                    'maxOutputTokens': 512
                }

                resp = requests.post(url, headers=headers, params=params, json=body, timeout=30)
                if resp.status_code != 200:
                    logging.debug(f"REST call to {url} failed: {resp.status_code} {resp.text}")
                    continue

                j = resp.json()

                # Try common response shapes
                # 1) {'candidates': [{'content': '...'}]}
                cands = j.get('candidates') or j.get('outputs') or j.get('choices')
                if isinstance(cands, list) and len(cands) > 0:
                    first = cands[0]
                    for key in ('content', 'text', 'output', 'message'):
                        if isinstance(first, dict) and key in first:
                            return first[key]
                    # If the structure is simple string inside first
                    if isinstance(first, str):
                        return first

                # 2) {'output': '...'} or {'text': '...'}
                for k in ('output', 'text', 'content'):
                    if k in j and isinstance(j[k], str):
                        return j[k]

                # 3) nested fields
                # Try to coalesce any string present
                def find_first_string(obj):
                    if isinstance(obj, str):
                        return obj
                    if isinstance(obj, dict):
                        for v in obj.values():
                            res = find_first_string(v)
                            if res:
                                return res
                    if isinstance(obj, list):
                        for item in obj:
                            res = find_first_string(item)
                            if res:
                                return res
                    return None

                s = find_first_string(j)
                if s:
                    return s

            except Exception as e:
                logging.debug(f"REST attempt for model {model} failed: {e}")
                continue

        return None

    # ----- Clause extraction helpers -----
    def extract_clauses(self, document_text: str):
        """Use LLM to extract likely clause chunks from the document.

        Returns a list of {'id': str, 'text': str} or raises on failure.
        """
        # If GenAI client isn't available, raise so callers can fallback to heuristics
        if not _HAS_GENAI or genai is None:
            raise RuntimeError("LLM client not available for clause extraction")

        try:
            prompt = f"Extract the most important clauses from the following document. Return a JSON array of objects with 'id' and 'text' fields. Limit to the top 12 clauses. Document:\n\n{document_text[:8000]}"
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            # Try to parse a JSON array from response.text
            import json
            txt = response.text.strip()
            # Some models may return markdown; attempt to find the first JSON marker
            start = txt.find('[')
            if start != -1:
                txt = txt[start:]
            data = json.loads(txt)
            clauses = []
            for i, item in enumerate(data):
                if isinstance(item, dict) and 'text' in item:
                    clauses.append({'id': item.get('id') or f'c{i+1}', 'text': item['text']})
            return clauses
        except Exception:
            # Re-raise to allow the caller to fall back
            raise

    def score_clauses(self, clauses):
        """Score a list of clause dicts with risk categories using the LLM.

        Expects clauses as [{'id':..., 'text':...}]. Returns [{'id','clause_text','risk','highlights'}]
        """
        if not _HAS_GENAI or genai is None:
            raise RuntimeError("LLM client not available for clause scoring")

        try:
            import json
            items = [{'id': c['id'], 'text': c['text'][:2000]} for c in clauses]
            prompt = f"For each clause in the following JSON array, assign a risk level: low, medium, or high. Return JSON array of objects with id, clause_text, risk, highlights (array of strings).\n\n{json.dumps(items)}"
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            txt = response.text.strip()
            start = txt.find('[')
            if start != -1:
                txt = txt[start:]
            return json.loads(txt)
        except Exception:
            raise

    def get_legal_references(self, clause_text: str, document_type: str, clause_type: str = "") -> list:
        """
        Get relevant legal references for a specific clause using Gemini AI.
        
        Args:
            clause_text (str): The text of the clause to analyze
            document_type (str): Type of document (e.g., 'Contract', 'Lease Agreement')
            clause_type (str): Specific type of clause (optional)
        
        Returns:
            list: List of relevant legal references with metadata
        """
        if not self.is_available():
            return self._get_mock_legal_references(document_type)

        if not _HAS_GENAI or genai is None:
            logging.warning("GenAI client not available; returning mock legal references")
            return self._get_mock_legal_references(document_type)

        try:
            client = genai.Client(api_key=self.api_key)
            
            prompt = f"""
            Analyze the following clause from a {document_type} document and identify relevant Indian laws, regulations, and legal references.

            Clause Text: "{clause_text}"
            Document Type: {document_type}
            Clause Type: {clause_type}

            Please provide a JSON array of relevant legal references with the following structure:
            [
                {{
                    "id": "unique_id",
                    "title": "Name of the law/regulation/guideline",
                    "type": "act|regulation|guideline|rule|circular",
                    "authority": "Governing authority (e.g., RBI, MCA, Labour Ministry)",
                    "section": "Specific section/rule number (if applicable)",
                    "description": "Brief description of relevance to this clause",
                    "relevance": "high|medium|low",
                    "url": "Official URL (if available)",
                    "lastUpdated": "Date when law was last updated (if known)"
                }}
            ]

            Focus on Indian laws and regulations. Include acts like:
            - Indian Contract Act 1872
            - Consumer Protection Act 2019
            - Information Technology Act 2000
            - Companies Act 2013
            - Labour Codes
            - RBI Guidelines
            - Model Tenancy Act 2021
            - Data Protection laws

            Return only the JSON array, no additional text.
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            response_text = response.text.strip()
            
            # Extract JSON from response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
            
            if start_idx != -1 and end_idx != -1:
                json_text = response_text[start_idx:end_idx + 1]
                import json
                legal_references = json.loads(json_text)
                return legal_references
            else:
                logging.warning("Could not extract valid JSON from Gemini response")
                return self._get_mock_legal_references(document_type)
                
        except Exception as e:
            logging.error(f"Error getting legal references from Gemini: {e}")
            return self._get_mock_legal_references(document_type)

    def _get_mock_legal_references(self, document_type: str) -> list:
        """
        Provide mock legal references based on document type.
        """
        base_references = []
        
        # Common references for all legal documents
        base_references.extend([
            {
                "id": "ica_1872",
                "title": "Indian Contract Act 1872",
                "type": "act",
                "authority": "Ministry of Law and Justice",
                "section": "Section 10, 23",
                "description": "Fundamental principles of contract formation and validity",
                "relevance": "high",
                "url": "https://legislative.gov.in/sites/default/files/A1872-09.pdf",
                "lastUpdated": "1872"
            },
            {
                "id": "cpa_2019",
                "title": "Consumer Protection Act 2019",
                "type": "act",
                "authority": "Ministry of Consumer Affairs",
                "section": "Section 2, 18",
                "description": "Consumer rights and unfair trade practices",
                "relevance": "medium",
                "url": "https://consumeraffairs.nic.in/sites/default/files/CP%20Act%202019.pdf",
                "lastUpdated": "2019"
            }
        ])
        
        # Document type specific references
        if document_type.lower() in ['contract', 'agreement', 'service agreement']:
            base_references.extend([
                {
                    "id": "companies_act_2013",
                    "title": "Companies Act 2013",
                    "type": "act",
                    "authority": "Ministry of Corporate Affairs",
                    "section": "Section 188, 197",
                    "description": "Corporate contract regulations and related party transactions",
                    "relevance": "medium",
                    "url": "https://www.mca.gov.in/content/mca/global/en/acts-rules/acts/companies-act-2013.html",
                    "lastUpdated": "2013"
                }
            ])
            
        elif document_type.lower() in ['lease agreement', 'rental agreement']:
            base_references.extend([
                {
                    "id": "model_tenancy_act_2021",
                    "title": "Model Tenancy Act 2021",
                    "type": "act",
                    "authority": "Ministry of Housing and Urban Affairs",
                    "section": "Section 4, 7, 12",
                    "description": "Comprehensive framework for rental agreements and tenant rights",
                    "relevance": "high",
                    "url": "https://mohua.gov.in/upload/uploadfiles/files/Model%20Tenancy%20Act.pdf",
                    "lastUpdated": "2021"
                }
            ])
            
        elif document_type.lower() in ['employment agreement', 'employment contract']:
            base_references.extend([
                {
                    "id": "labour_codes_2019",
                    "title": "The Code on Wages 2019",
                    "type": "act",
                    "authority": "Ministry of Labour and Employment",
                    "section": "Section 3, 5, 9",
                    "description": "Wage payment, minimum wages, and employment terms",
                    "relevance": "high",
                    "url": "https://labour.gov.in/sites/default/files/SS_Code_on_Wages%2C2019.pdf",
                    "lastUpdated": "2019"
                }
            ])
            
        elif document_type.lower() in ['privacy policy', 'data processing agreement']:
            base_references.extend([
                {
                    "id": "dpdp_act_2023",
                    "title": "Digital Personal Data Protection Act 2023",
                    "type": "act",
                    "authority": "Ministry of Electronics and IT",
                    "section": "Section 6, 8, 11",
                    "description": "Data protection, consent, and individual rights",
                    "relevance": "high",
                    "url": "https://www.meity.gov.in/writereaddata/files/Digital%20Personal%20Data%20Protection%20Act%202023.pdf",
                    "lastUpdated": "2023"
                },
                {
                    "id": "it_act_2000",
                    "title": "Information Technology Act 2000",
                    "type": "act",
                    "authority": "Ministry of Electronics and IT",
                    "section": "Section 43A, 72A",
                    "description": "Data security and breach notification requirements",
                    "relevance": "medium",
                    "url": "https://www.meity.gov.in/content/information-technology-act-2000",
                    "lastUpdated": "2008"
                }
            ])

        return base_references

    def get_what_if_scenarios(self, clause_text: str, document_type: str, clause_type: str = "") -> list:
        """
        Generate what-if scenarios for a specific clause using Gemini AI.
        
        Args:
            clause_text (str): The text of the clause to analyze
            document_type (str): Type of document (e.g., 'Contract', 'Lease Agreement')
            clause_type (str): Risk level or specific type of clause
        
        Returns:
            list: List of what-if scenarios with outcomes and mitigation strategies
        """
        if not self.is_available():
            return self._get_mock_scenarios(document_type, clause_type)

        if not _HAS_GENAI or genai is None:
            logging.warning("GenAI client not available; returning mock scenarios")
            return self._get_mock_scenarios(document_type, clause_type)

        try:
            client = genai.Client(api_key=self.api_key)
            
            prompt = f"""
            Analyze the following clause from a {document_type} document and generate realistic what-if scenarios.

            Clause Text: "{clause_text}"
            Document Type: {document_type}
            Risk Level: {clause_type}

            Please provide a JSON array of what-if scenarios with the following structure:
            [
                {{
                    "id": "unique_scenario_id",
                    "title": "Scenario Title",
                    "description": "Brief description of the scenario",
                    "likelihood": "high|medium|low",
                    "impact": "high|medium|low", 
                    "category": "breach|compliance|financial|operational|legal",
                    "outcomes": ["outcome1", "outcome2", "outcome3"],
                    "mitigation": ["strategy1", "strategy2", "strategy3"],
                    "precedent": "Relevant legal precedent (optional)"
                }}
            ]

            Focus on:
            1. Breach scenarios (what happens if clause is violated)
            2. Compliance issues and regulatory implications
            3. Financial consequences and cost analysis
            4. Operational disruptions and business impact
            5. Legal ramifications and dispute resolution

            Consider Indian legal context and regulations. Provide practical, actionable scenarios.
            Return only the JSON array, no additional text.
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            response_text = response.text.strip()
            
            # Extract JSON from response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
            
            if start_idx != -1 and end_idx != -1:
                json_text = response_text[start_idx:end_idx + 1]
                import json
                scenarios = json.loads(json_text)
                return scenarios
            else:
                logging.warning("Could not extract valid JSON from Gemini response")
                return self._get_mock_scenarios(document_type, clause_type)
                
        except Exception as e:
            logging.error(f"Error getting what-if scenarios from Gemini: {e}")
            return self._get_mock_scenarios(document_type, clause_type)

    def _get_mock_scenarios(self, document_type: str, clause_type: str = "") -> list:
        """
        Provide mock what-if scenarios based on document type and clause risk level.
        """
        scenarios = []
        
        # Common breach scenario for all documents
        scenarios.append({
            "id": "clause_breach",
            "title": "Clause Violation Scenario",
            "description": "Analysis of consequences if this clause is breached or not fulfilled",
            "likelihood": "medium",
            "impact": "high" if clause_type == "high" else "medium",
            "category": "breach",
            "outcomes": [
                "Legal action may be initiated by the non-breaching party",
                "Contractual penalties or damages may be enforced",
                "Business relationship may be damaged or terminated"
            ],
            "mitigation": [
                "Implement monitoring and compliance systems",
                "Regular communication and progress reviews",
                "Consider adding grace periods for minor breaches"
            ]
        })
        
        # Document-specific scenarios
        if document_type.lower() in ['contract', 'agreement']:
            scenarios.extend([
                {
                    "id": "payment_default",
                    "title": "Payment Default Scenario",
                    "description": "Financial implications of delayed or missed payments",
                    "likelihood": "medium",
                    "impact": "medium",
                    "category": "financial",
                    "outcomes": [
                        "Interest charges and late fees may accumulate",
                        "Credit rating impact for the defaulting party",
                        "Suspension of services until payment is received"
                    ],
                    "mitigation": [
                        "Establish clear payment terms and schedules",
                        "Require security deposits or guarantees",
                        "Implement automated payment reminders"
                    ]
                },
                {
                    "id": "scope_expansion",
                    "title": "Scope Creep Scenario",
                    "description": "Challenges when project requirements expand beyond original agreement",
                    "likelihood": "high",
                    "impact": "medium",
                    "category": "operational",
                    "outcomes": [
                        "Budget overruns and resource strain",
                        "Timeline delays and missed deadlines",
                        "Disputes over additional compensation"
                    ],
                    "mitigation": [
                        "Define clear change management procedures",
                        "Require written approval for scope changes",
                        "Establish rates for additional work upfront"
                    ]
                }
            ])
            
        elif document_type.lower() in ['lease agreement', 'rental agreement']:
            scenarios.extend([
                {
                    "id": "property_damage",
                    "title": "Property Damage Scenario",
                    "description": "Liability and consequences of property damage during tenancy",
                    "likelihood": "low",
                    "impact": "high",
                    "category": "legal",
                    "outcomes": [
                        "Tenant liability for repair and restoration costs",
                        "Potential forfeiture of security deposit",
                        "Insurance claims and premium adjustments"
                    ],
                    "mitigation": [
                        "Require comprehensive tenant insurance coverage",
                        "Conduct regular property inspections",
                        "Define clear distinction between wear and damage"
                    ],
                    "precedent": "Model Tenancy Act 2021, Section 7"
                },
                {
                    "id": "early_termination",
                    "title": "Early Termination Scenario", 
                    "description": "Financial and legal consequences of breaking lease early",
                    "likelihood": "medium",
                    "impact": "medium",
                    "category": "financial",
                    "outcomes": [
                        "Early termination penalties as per lease terms",
                        "Loss of security deposit and advance rent",
                        "Difficulty in obtaining future rental references"
                    ],
                    "mitigation": [
                        "Include reasonable termination clauses",
                        "Allow for subletting with landlord approval",
                        "Consider graduated penalty structures"
                    ]
                }
            ])
            
        elif 'employment' in document_type.lower():
            scenarios.append({
                "id": "termination_dispute",
                "title": "Wrongful Termination Scenario",
                "description": "Legal and financial implications of disputed employment termination",
                "likelihood": "medium",
                "impact": "high",
                "category": "legal",
                "outcomes": [
                    "Labor court proceedings and associated legal costs",
                    "Potential compensation for wrongful dismissal",
                    "Reputational damage and regulatory scrutiny"
                ],
                "mitigation": [
                    "Follow proper disciplinary and termination procedures",
                    "Maintain detailed documentation of performance issues",
                    "Provide adequate notice period or compensation in lieu"
                ],
                "precedent": "Industrial Disputes Act 1947, Section 25F"
            })
            
        elif document_type.lower() in ['privacy policy', 'data processing agreement']:
            scenarios.append({
                "id": "data_breach",
                "title": "Data Breach Scenario",
                "description": "Regulatory and financial consequences of personal data compromise",
                "likelihood": "low",
                "impact": "high",
                "category": "compliance",
                "outcomes": [
                    "Regulatory penalties under DPDP Act 2023",
                    "Mandatory breach notification to authorities and individuals",
                    "Class action lawsuits and compensation claims"
                ],
                "mitigation": [
                    "Implement robust cybersecurity frameworks",
                    "Regular security audits and vulnerability assessments",
                    "Comprehensive incident response and recovery plans"
                ],
                "precedent": "Digital Personal Data Protection Act 2023, Section 33"
            })

        return scenarios