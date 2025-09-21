import requests
import os
from typing import Optional, Dict, Any
import logging
from cachetools import TTLCache
_HAS_GENAI = False
try:
    # Import lazily compatible package name if available
    from google import genai  # type: ignore
    _HAS_GENAI = True
except Exception:
    # GenAI client not installed in this environment; methods will fall back or raise as needed
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

            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f"{system_prompt}\n\n{user_prompt}")

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
                prompt = (
                    "Classify the following legal document into a single concise type label. "
                    "Return only one short label (no explanation). Examples: 'Contract', 'Lease Agreement', "
                    "'Will', 'Privacy Policy', 'Terms of Service', 'Invoice', 'Legal Notice', 'NDA'.\n\n"
                    "Document Text:\n" + (document_text[:3000] if document_text else "")
                )
                resp = None
                try:
                    model = genai.GenerativeModel("gemini-1.5")
                    resp = model.generate_content(prompt)
                except Exception:
                    try:
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        resp = model.generate_content(prompt)
                    except Exception:
                        logging.debug("All SDK generation attempts failed for document classification", exc_info=True)

                label = (resp.text or '').strip() if resp is not None else ''
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

            resp = None
            try:
                model = genai.GenerativeModel("gemini-1.5")
                resp = model.generate_content(full_prompt)
            except Exception:
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    resp = model.generate_content(full_prompt)
                except Exception:
                    logging.debug("SDK generation failed for answer_user_query", exc_info=True)

            answer = (resp.text or '').strip() if resp is not None else ''

            # If the model returns an empty or unhelpful answer, return a clear fallback
            if not answer:
                return "Insufficient information in the clause to answer."

            return answer
        except Exception as e:
            logging.error(f"Error answering user query via GenAI: {e}")
            return "I'm sorry — I couldn't process that question right now. Please try again or ask a more specific question."

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
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
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
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
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
            model = genai.GenerativeModel("gemini-1.5-flash")
            
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

            response = model.generate_content(prompt)
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
        Dynamic discovery of authoritative legal references. Instead of returning
        a large hard-coded list, this helper attempts lightweight verification
        of well-known authoritative URLs for the given document type and
        returns only the resources that are reachable.

        If no authoritative resource can be confirmed, an empty list is returned
        (safer than returning fabricated references).
        """
        candidates = {
            'default': [
                'https://legislative.gov.in',
            ],
            'contract': [
                'https://legislative.gov.in/sites/default/files/A1872-09.pdf',
                'https://www.mca.gov.in/content/mca/global/en/acts-rules/acts/companies-act-2013.html',
            ],
            'lease agreement': [
                'https://mohua.gov.in/upload/uploadfiles/files/Model%20Tenancy%20Act.pdf',
            ],
            'employment agreement': [
                'https://labour.gov.in/sites/default/files/SS_Code_on_Wages%2C2019.pdf',
            ],
            'privacy policy': [
                'https://www.meity.gov.in/writereaddata/files/Digital%20Personal%20Data%20Protection%20Act%202023.pdf',
                'https://www.meity.gov.in/content/information-technology-act-2000',
            ]
        }

        key = (document_type or '').lower()
        urls = []
        # assemble fallback candidates
        urls.extend(candidates.get('default', []))
        # try exact match, then prefix matches
        if key in candidates:
            urls.extend(candidates[key])
        else:
            for k, v in candidates.items():
                if k != 'default' and k in key:
                    urls.extend(v)

        verified = []
        seen = set()
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            try:
                # Use a HEAD request first to avoid downloading large PDFs
                h = requests.head(url, timeout=6, allow_redirects=True)
                if h.status_code >= 200 and h.status_code < 400:
                    title = None
                    last_updated = None
                    # prefer Last-Modified header if present
                    if 'Last-Modified' in h.headers:
                        last_updated = h.headers.get('Last-Modified')
                    # If content-type is HTML, try to fetch small portion to extract title
                    ctype = h.headers.get('Content-Type', '')
                    if 'text/html' in ctype:
                        try:
                            r = requests.get(url, timeout=6)
                            txt = r.text
                            # crude title extraction
                            import re
                            m = re.search(r'<title>(.*?)</title>', txt, re.IGNORECASE | re.DOTALL)
                            if m:
                                title = m.group(1).strip()
                        except Exception:
                            title = None
                    # If still no title, infer from URL path
                    if not title:
                        title = url.rstrip('/').split('/')[-1] or url

                    entry = {
                        'id': url.split('/')[-1].split('?')[0],
                        'title': title,
                        'type': 'act' if 'act' in url.lower() or 'legislative' in url.lower() else 'resource',
                        'authority': None,
                        'section': None,
                        'description': None,
                        'relevance': 'high',
                        'url': url,
                        'lastUpdated': last_updated,
                    }
                    verified.append(entry)
                else:
                    # not accessible
                    continue
            except Exception:
                # network or DNS error; skip
                continue

        # Return verified authoritative resources only; if none, return empty list
        return verified

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
            model = genai.GenerativeModel("gemini-1.5-flash")
            
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

            response = model.generate_content(prompt)
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