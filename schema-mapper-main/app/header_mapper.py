"""
Header Mapper - Intelligent column mapping using LangChain and Groq.
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from fuzzywuzzy import fuzz, process
import logging

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from .schema_loader import SchemaLoader

logger = logging.getLogger(__name__)

class HeaderMapper:
    """Intelligent header mapping using AI and fuzzy matching."""
    
    def __init__(self, schema_loader: SchemaLoader):
        self.schema_loader = schema_loader
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Initialize Groq LLM
        if self.groq_api_key:
            self.llm=ChatGroq(groq_api_key=self.groq_api_key, model_name="llama-3.1-8b-instant",  temperature=0.1)
        else:
            logger.warning("GROQ_API_KEY not found. AI mapping will be disabled.")
            self.llm = None
        
        # Mapping templates
        self.mapping_prompt = PromptTemplate(
            input_variables=["source_headers", "canonical_headers", "examples"],
            template="""
You are an expert data analyst specializing in column mapping. Your task is to map source column headers to canonical schema columns.

Source headers: {source_headers}
Canonical headers: {canonical_headers}
Canonical examples: {examples}

For each source header, suggest the best matching canonical header. Consider:
1. Semantic similarity (meaning)
2. Data type compatibility
3. Common naming patterns
4. Business context

Return your response as a JSON object with this structure:
{{
    "mappings": {{
        "source_header": "canonical_header"
    }},
    "confidence_scores": {{
        "source_header": 0.95
    }},
    "reasoning": {{
        "source_header": "Brief explanation of why this mapping was chosen"
    }}
}}

Only include mappings you are confident about (confidence > 0.3). Be conservative and accurate.
"""
        )
        
        # Common mapping patterns for deterministic matching
        self.common_patterns = {
            # Order information
            'order_id': ['order_id', 'order_no', 'order_number', 'reference', 'ref'],
            'order_date': ['order_date', 'ordered_on', 'date', 'order_time'],
            
            # Customer information
            'customer_id': ['customer_id', 'cust_id', 'client_ref', 'client_id'],
            'customer_name': ['customer_name', 'customer', 'client_name', 'name'],
            'email': ['email', 'e-mail', 'contact', 'email_address'],
            'phone': ['phone', 'phone_number', 'mobile', 'contact_number'],
            
            # Address information
            'billing_address': ['billing_address', 'bill_addr', 'bill_to', 'billing'],
            'shipping_address': ['shipping_address', 'ship_addr', 'ship_to', 'shipping'],
            'city': ['city', 'town'],
            'state': ['state', 'province', 'state/province'],
            'postal_code': ['postal_code', 'zip', 'postal', 'pin', 'zip/postal'],
            'country': ['country', 'region', 'country/region'],
            
            # Product information
            'product_sku': ['product_sku', 'sku', 'stock_code', 'item_code'],
            'product_name': ['product_name', 'item', 'desc', 'description', 'product'],
            'category': ['category', 'cat', 'type'],
            'subcategory': ['subcategory', 'subcat', 'sub_type'],
            
            # Quantity and pricing
            'quantity': ['quantity', 'qty', 'units', 'count'],
            'unit_price': ['unit_price', 'price', 'unit_cost', 'rate'],
            'currency': ['currency', 'curr', 'ccy'],
            'discount_pct': ['discount_pct', 'discount', 'disc%', 'discount_percent'],
            'tax_pct': ['tax_pct', 'tax', 'tax%', 'gst', 'vat', 'tax_percent'],
            'shipping_fee': ['shipping_fee', 'ship_fee', 'shipping', 'logistics_fee'],
            'total_amount': ['total_amount', 'total', 'grand_total', 'amount'],
            
            # Tax information
            'tax_id': ['tax_id', 'gstin', 'vat_id', 'reg_no', 'tax_number']
        }
    
    def suggest_mapping(self, source_headers: List[str]) -> Dict[str, any]:
        """
        Suggest mappings from source headers to canonical schema.
        
        Args:
            source_headers: List of source column headers
            
        Returns:
            Dict with mapping suggestions, confidence scores, and unmapped columns
        """
        try:
            canonical_headers = self.schema_loader.get_all_columns()
            
            # First, try deterministic pattern matching
            deterministic_mappings = self._deterministic_mapping(source_headers, canonical_headers)
            
            # Then, use AI for remaining unmapped headers
            ai_mappings = {}
            if self.llm and len(deterministic_mappings['unmapped']) > 0:
                ai_mappings = self._ai_mapping(
                    deterministic_mappings['unmapped'], 
                    canonical_headers
                )
            
            # Combine results
            final_mapping = {**deterministic_mappings['mapping'], **ai_mappings.get('mapping', {})}
            final_confidence = {**deterministic_mappings['confidence'], **ai_mappings.get('confidence', {})}
            
            # Find still unmapped columns
            mapped_source_headers = set(final_mapping.keys())
            unmapped_columns = [h for h in source_headers if h not in mapped_source_headers]
            
            return {
                'mapping': final_mapping,
                'confidence_scores': final_confidence,
                'unmapped_columns': unmapped_columns,
                'mapping_method': {
                    'deterministic': len(deterministic_mappings['mapping']),
                    'ai_assisted': len(ai_mappings.get('mapping', {}))
                }
            }
            
        except Exception as e:
            logger.error(f"Error in header mapping: {e}")
            return {
                'mapping': {},
                'confidence_scores': {},
                'unmapped_columns': source_headers,
                'error': str(e)
            }
    
    def _deterministic_mapping(self, source_headers: List[str], canonical_headers: List[str]) -> Dict[str, any]:
        """Perform deterministic mapping using patterns and fuzzy matching."""
        mapping = {}
        confidence = {}
        unmapped = []
        
        for source_header in source_headers:
            source_lower = source_header.lower().strip()
            best_match = None
            best_score = 0
            
            # Try pattern matching first
            for canonical, patterns in self.common_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in source_lower or source_lower in pattern.lower():
                        best_match = canonical
                        best_score = 0.9  # High confidence for pattern matches
                        break
                if best_match:
                    break
            
            # If no pattern match, try fuzzy matching
            if not best_match:
                fuzzy_result = process.extractOne(source_lower, canonical_headers, scorer=fuzz.ratio)
                if fuzzy_result and fuzzy_result[1] > 60:  # Threshold for fuzzy matching
                    best_match = fuzzy_result[0]
                    best_score = fuzzy_result[1] / 100.0
            
            # Try semantic similarity for common variations
            if not best_match:
                best_match, best_score = self._semantic_mapping(source_lower, canonical_headers)
            
            if best_match and best_score > 0.3:
                mapping[source_header] = best_match
                confidence[source_header] = best_score
            else:
                unmapped.append(source_header)
        
        return {
            'mapping': mapping,
            'confidence': confidence,
            'unmapped': unmapped
        }
    
    def _semantic_mapping(self, source_header: str, canonical_headers: List[str]) -> Tuple[Optional[str], float]:
        """Perform semantic mapping for common variations."""
        # Common semantic mappings
        semantic_map = {
            'id': ['order_id', 'customer_id', 'product_sku'],
            'name': ['customer_name', 'product_name'],
            'date': ['order_date'],
            'amount': ['total_amount', 'unit_price'],
            'price': ['unit_price', 'total_amount'],
            'qty': ['quantity'],
            'addr': ['billing_address', 'shipping_address'],
            'email': ['email'],
            'phone': ['phone'],
            'city': ['city'],
            'state': ['state'],
            'zip': ['postal_code'],
            'country': ['country'],
            'sku': ['product_sku'],
            'desc': ['product_name'],
            'cat': ['category'],
            'sub': ['subcategory'],
            'disc': ['discount_pct'],
            'tax': ['tax_pct'],
            'ship': ['shipping_fee'],
            'total': ['total_amount'],
            'grand': ['total_amount']
        }
        
        for keyword, candidates in semantic_map.items():
            if keyword in source_header:
                # Find the best candidate
                for candidate in candidates:
                    if candidate in canonical_headers:
                        return candidate, 0.7  # Medium confidence for semantic matches
        
        return None, 0.0
    
    def _ai_mapping(self, unmapped_headers: List[str], canonical_headers: List[str]) -> Dict[str, any]:
        """Use AI to map remaining unmapped headers."""
        try:
            # Prepare examples for the AI
            examples = {}
            for header in canonical_headers[:5]:  # Show first 5 as examples
                info = self.schema_loader.get_column_info(header)
                if info:
                    examples[header] = f"{info['description']} (e.g., {info['example']})"
            
            # Create the prompt
            prompt = self.mapping_prompt.format(
                source_headers=unmapped_headers,
                canonical_headers=canonical_headers,
                examples=json.dumps(examples, indent=2)
            )
            
            # Get AI response
            messages = [
                SystemMessage(content="You are a data mapping expert. Provide accurate, conservative mappings."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse the response
            try:
                result = json.loads(response.content)
                return {
                    'mapping': result.get('mappings', {}),
                    'confidence': result.get('confidence_scores', {}),
                    'reasoning': result.get('reasoning', {})
                }
            except json.JSONDecodeError:
                logger.warning("Failed to parse AI response as JSON")
                return {'mapping': {}, 'confidence': {}}
                
        except Exception as e:
            logger.error(f"Error in AI mapping: {e}")
            return {'mapping': {}, 'confidence': {}}
    
    def get_mapping_confidence(self, source_header: str, canonical_header: str) -> float:
        """
        Get confidence score for a specific mapping.
        
        Args:
            source_header: Source column header
            canonical_header: Canonical column header
            
        Returns:
            Confidence score between 0 and 1
        """
        # Use fuzzy matching to calculate confidence
        score = fuzz.ratio(source_header.lower(), canonical_header.lower()) / 100.0
        
        # Boost confidence for pattern matches
        for patterns in self.common_patterns.values():
            if any(pattern.lower() in source_header.lower() for pattern in patterns):
                if canonical_header in self.common_patterns:
                    score = max(score, 0.9)
        
        return min(score, 1.0)
    
    def validate_mapping(self, source_header: str, canonical_header: str) -> Dict[str, any]:
        """
        Validate a specific mapping and provide feedback.
        
        Args:
            source_header: Source column header
            canonical_header: Canonical column header
            
        Returns:
            Dict with validation results
        """
        confidence = self.get_mapping_confidence(source_header, canonical_header)
        
        # Get canonical column info
        column_info = self.schema_loader.get_column_info(canonical_header)
        
        return {
            'valid': confidence > 0.3,
            'confidence': confidence,
            'canonical_info': column_info,
            'suggestions': self._get_mapping_suggestions(source_header, canonical_header)
        }
    
    def _get_mapping_suggestions(self, source_header: str, canonical_header: str) -> List[str]:
        """Get suggestions for improving a mapping."""
        suggestions = []
        
        if canonical_header in self.common_patterns:
            patterns = self.common_patterns[canonical_header]
            if not any(pattern.lower() in source_header.lower() for pattern in patterns):
                suggestions.append(f"Consider using one of these patterns: {', '.join(patterns)}")
        
        return suggestions