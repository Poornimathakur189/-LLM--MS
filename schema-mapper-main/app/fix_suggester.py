"""
Fix Suggester - Provides targeted fix suggestions using AI for remaining issues.
"""

import os
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
import logging

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import pandas as pd

logger = logging.getLogger(__name__)

class FixSuggester:
    """Provides targeted fix suggestions using AI for data quality issues."""
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Initialize Groq LLM
        if self.groq_api_key:
            # self.groq_api_key=os.getenv("GROQ_API_KEY")
            self.llm=ChatGroq(groq_api_key=self.groq_api_key, model_name="llama-3.1-8b-instant",  temperature=0.1)
        else:
            logger.warning("GROQ_API_KEY not found. AI fix suggestions will be disabled.")
            self.llm = None
        
        # Fix suggestion templates
        self.fix_prompt = PromptTemplate(
            input_variables=["issue_type", "issue_details", "context", "canonical_schema"],
            template="""
You are a data quality expert. Analyze the following data quality issue and provide targeted fix suggestions.

Issue Type: {issue_type}
Issue Details: {issue_details}
Context: {context}
Canonical Schema: {canonical_schema}

Provide your response as a JSON object with this structure:
{{
    "fix_id": "unique_fix_identifier",
    "issue_summary": "Brief description of the issue",
    "suggested_fixes": [
        {{
            "fix_type": "automatic|manual|semi_automatic",
            "description": "What this fix does",
            "confidence": 0.95,
            "implementation": "How to implement this fix",
            "examples": ["example1", "example2"]
        }}
    ],
    "alternative_approaches": [
        {{
            "approach": "Alternative way to handle this",
            "pros": ["advantage1", "advantage2"],
            "cons": ["disadvantage1", "disadvantage2"]
        }}
    ],
    "prevention_tips": [
        "How to prevent this issue in the future"
    ]
}}

Be specific, actionable, and provide clear implementation steps. Focus on practical solutions.
"""
        )
        
        # Common fix patterns for deterministic suggestions
        self.common_fixes = {
            'missing_value': {
                'automatic_fixes': [
                    {
                        'description': 'Fill missing values with default based on column type',
                        'implementation': 'Use column-specific defaults (e.g., 0 for numbers, empty string for text)',
                        'confidence': 0.7
                    },
                    {
                        'description': 'Remove rows with missing critical fields',
                        'implementation': 'Drop rows where required fields are null',
                        'confidence': 0.8
                    }
                ],
                'manual_fixes': [
                    {
                        'description': 'Manual data entry for missing values',
                        'implementation': 'Review each missing value and enter appropriate data',
                        'confidence': 0.95
                    }
                ]
            },
            'pattern_mismatch': {
                'automatic_fixes': [
                    {
                        'description': 'Apply regex-based pattern correction',
                        'implementation': 'Use pattern matching to transform values to expected format',
                        'confidence': 0.8
                    }
                ],
                'manual_fixes': [
                    {
                        'description': 'Review and correct pattern mismatches manually',
                        'implementation': 'Examine each mismatch and correct to match expected pattern',
                        'confidence': 0.9
                    }
                ]
            },
            'type_mismatch': {
                'automatic_fixes': [
                    {
                        'description': 'Convert data types using pandas conversion',
                        'implementation': 'Use pd.to_numeric(), pd.to_datetime(), etc.',
                        'confidence': 0.7
                    }
                ],
                'manual_fixes': [
                    {
                        'description': 'Review and correct type mismatches',
                        'implementation': 'Examine each value and convert to appropriate type',
                        'confidence': 0.9
                    }
                ]
            },
            'pricing_inconsistency': {
                'automatic_fixes': [
                    {
                        'description': 'Recalculate total amount from components',
                        'implementation': 'Calculate: (unit_price * quantity * (1 - discount_pct) * (1 + tax_pct)) + shipping_fee',
                        'confidence': 0.9
                    }
                ],
                'manual_fixes': [
                    {
                        'description': 'Review pricing calculations manually',
                        'implementation': 'Check each pricing component and verify calculations',
                        'confidence': 0.95
                    }
                ]
            },
            'duplicate_order_id': {
                'automatic_fixes': [
                    {
                        'description': 'Generate unique order IDs for duplicates',
                        'implementation': 'Append suffix to duplicate order IDs (e.g., ORD-1001-1)',
                        'confidence': 0.8
                    },
                    {
                        'description': 'Remove duplicate rows',
                        'implementation': 'Keep first occurrence, remove subsequent duplicates',
                        'confidence': 0.7
                    }
                ],
                'manual_fixes': [
                    {
                        'description': 'Review duplicate orders manually',
                        'implementation': 'Examine each duplicate to determine if they are truly duplicates or separate orders',
                        'confidence': 0.95
                    }
                ]
            }
        }
    
    def suggest_fixes(self, validation_results: Dict[str, Any], sample_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Suggest fixes for validation issues.
        
        Args:
            validation_results: Results from data validation
            sample_data: Sample of the data for context
            
        Returns:
            List of fix suggestions
        """
        suggestions = []
        issues = validation_results.get('issues', [])
        
        # Group issues by type
        issues_by_type = self._group_issues_by_type(issues)
        
        for issue_type, type_issues in issues_by_type.items():
            if self.llm:
                # Use AI for complex suggestions
                ai_suggestions = self._get_ai_suggestions(issue_type, type_issues, sample_data)
                suggestions.extend(ai_suggestions)
            else:
                # Use deterministic suggestions
                deterministic_suggestions = self._get_deterministic_suggestions(issue_type, type_issues)
                suggestions.extend(deterministic_suggestions)
        
        return suggestions
    
    def _group_issues_by_type(self, issues: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group issues by their type."""
        issues_by_type = {}
        for issue in issues:
            issue_type = issue.get('type', 'unknown')
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
        return issues_by_type
    
    def _get_deterministic_suggestions(self, issue_type: str, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get deterministic fix suggestions based on issue type."""
        suggestions = []
        
        if issue_type in self.common_fixes:
            fix_patterns = self.common_fixes[issue_type]
            
            for fix_type, fixes in fix_patterns.items():
                for fix in fixes:
                    suggestion = {
                        'fix_id': str(uuid.uuid4()),
                        'issue_type': issue_type,
                        'issue_count': len(issues),
                        'fix_type': fix_type,
                        'description': fix['description'],
                        'implementation': fix['implementation'],
                        'confidence': fix['confidence'],
                        'affected_rows': [issue.get('row') for issue in issues],
                        'examples': self._get_examples_from_issues(issues, 3)
                    }
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _get_ai_suggestions(self, issue_type: str, issues: List[Dict[str, Any]], sample_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get AI-powered fix suggestions."""
        try:
            # Prepare context
            issue_details = {
                'count': len(issues),
                'examples': self._get_examples_from_issues(issues, 5),
                'affected_columns': list(set(issue.get('column', '') for issue in issues)),
                'severity_distribution': self._get_severity_distribution(issues)
            }
            
            context = {
                'sample_data': sample_data or {},
                'issue_statistics': issue_details
            }
            
            canonical_schema = {
                'order_id': 'Format: ORD-XXXX',
                'order_date': 'Format: YYYY-MM-DD',
                'customer_id': 'Format: CUST-XXXX',
                'email': 'Valid email address',
                'phone': 'Format: +91-XXXXXXXXXX',
                'postal_code': '6-digit postal code',
                'currency': 'Standard currency codes (INR, USD, etc.)',
                'amounts': 'Numeric values without currency symbols'
            }
            
            # Create prompt
            prompt = self.fix_prompt.format(
                issue_type=issue_type,
                issue_details=json.dumps(issue_details, indent=2),
                context=json.dumps(context, indent=2),
                canonical_schema=json.dumps(canonical_schema, indent=2)
            )
            
            # Get AI response
            messages = [
                SystemMessage(content="You are a data quality expert. Provide practical, actionable fix suggestions."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse response
            try:
                result = json.loads(response.content)
                
                # Convert AI response to our format
                suggestions = []
                for fix in result.get('suggested_fixes', []):
                    suggestion = {
                        'fix_id': result.get('fix_id', str(uuid.uuid4())),
                        'issue_type': issue_type,
                        'issue_count': len(issues),
                        'fix_type': fix.get('fix_type', 'manual'),
                        'description': fix.get('description', ''),
                        'implementation': fix.get('implementation', ''),
                        'confidence': fix.get('confidence', 0.5),
                        'affected_rows': [issue.get('row') for issue in issues],
                        'examples': fix.get('examples', []),
                        'ai_generated': True,
                        'alternative_approaches': result.get('alternative_approaches', []),
                        'prevention_tips': result.get('prevention_tips', [])
                    }
                    suggestions.append(suggestion)
                
                return suggestions
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse AI response as JSON")
                return self._get_deterministic_suggestions(issue_type, issues)
                
        except Exception as e:
            logger.error(f"Error getting AI suggestions: {e}")
            return self._get_deterministic_suggestions(issue_type, issues)
    
    def _get_examples_from_issues(self, issues: List[Dict[str, Any]], max_examples: int = 5) -> List[Dict[str, Any]]:
        """Extract examples from issues for context."""
        examples = []
        for issue in issues[:max_examples]:
            example = {
                'row': issue.get('row'),
                'column': issue.get('column'),
                'value': issue.get('value'),
                'message': issue.get('message')
            }
            examples.append(example)
        return examples
    
    def _get_severity_distribution(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of issue severities."""
        severity_count = {}
        for issue in issues:
            severity = issue.get('severity', 'unknown')
            severity_count[severity] = severity_count.get(severity, 0) + 1
        return severity_count
    
    def apply_fix(self, fix_id: str, accepted: bool, custom_value: Optional[str] = None) -> Dict[str, Any]:
        """
        Apply a specific fix.
        
        Args:
            fix_id: Unique identifier for the fix
            accepted: Whether the fix was accepted by the user
            custom_value: Custom value provided by the user
            
        Returns:
            Dict with application result
        """
        # This would typically involve applying the fix to the data
        # For now, return a mock response
        return {
            'fix_id': fix_id,
            'applied': accepted,
            'custom_value': custom_value,
            'status': 'success' if accepted else 'rejected',
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def get_fix_effectiveness(self, fix_id: str) -> Dict[str, Any]:
        """
        Get effectiveness metrics for a specific fix.
        
        Args:
            fix_id: Unique identifier for the fix
            
        Returns:
            Dict with effectiveness metrics
        """
        # This would typically track fix effectiveness over time
        return {
            'fix_id': fix_id,
            'times_applied': 0,
            'success_rate': 0.0,
            'user_satisfaction': 0.0,
            'last_applied': None
        }
    
    def suggest_preventive_measures(self, validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest preventive measures to avoid future issues.
        
        Args:
            validation_results: Results from data validation
            
        Returns:
            List of preventive measures
        """
        measures = []
        
        # Analyze common issues
        issues = validation_results.get('issues', [])
        issues_by_type = self._group_issues_by_type(issues)
        
        for issue_type, type_issues in issues_by_type.items():
            if issue_type == 'missing_value':
                measures.append({
                    'type': 'data_validation',
                    'description': 'Implement client-side validation to prevent missing values',
                    'implementation': 'Add required field validation in data entry forms',
                    'priority': 'high'
                })
            elif issue_type == 'pattern_mismatch':
                measures.append({
                    'type': 'format_guidelines',
                    'description': 'Provide clear format guidelines for data entry',
                    'implementation': 'Create data entry templates with format examples',
                    'priority': 'medium'
                })
            elif issue_type == 'pricing_inconsistency':
                measures.append({
                    'type': 'calculation_validation',
                    'description': 'Implement automatic calculation validation',
                    'implementation': 'Add real-time calculation checks in pricing forms',
                    'priority': 'high'
                })
        
        return measures
    
    def get_fix_statistics(self) -> Dict[str, Any]:
        """Get statistics about fix suggestions and applications."""
        # This would typically track statistics over time
        return {
            'total_suggestions_made': 0,
            'total_fixes_applied': 0,
            'most_common_issues': [],
            'fix_success_rate': 0.0,
            'user_satisfaction_score': 0.0
        }