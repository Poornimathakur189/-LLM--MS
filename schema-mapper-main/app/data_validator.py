"""
Data Validator - Deterministic data validation and cleaning pipeline.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Deterministic data validation and cleaning pipeline."""
    
    def __init__(self):
        # Validation rules for each canonical column type
        self.validation_rules = {
            'order_id': {
                'pattern': r'^ORD-\d+$',
                'required': True,
                'type': 'string'
            },
            'order_date': {
                'pattern': r'^\d{4}-\d{2}-\d{2}$',
                'required': True,
                'type': 'date'
            },
            'customer_id': {
                'pattern': r'^CUST-\d+$',
                'required': True,
                'type': 'string'
            },
            'customer_name': {
                'required': True,
                'type': 'string',
                'min_length': 1
            },
            'email': {
                'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                'required': True,
                'type': 'email'
            },
            'phone': {
                'pattern': r'^\+91-\d{10}$',
                'required': True,
                'type': 'phone'
            },
            'billing_address': {
                'required': True,
                'type': 'string',
                'min_length': 1
            },
            'shipping_address': {
                'required': True,
                'type': 'string',
                'min_length': 1
            },
            'city': {
                'required': True,
                'type': 'string',
                'min_length': 1
            },
            'state': {
                'required': True,
                'type': 'string',
                'min_length': 1
            },
            'postal_code': {
                'pattern': r'^\d{6}$',
                'required': True,
                'type': 'postal_code'
            },
            'country': {
                'required': True,
                'type': 'string',
                'allowed_values': ['India']
            },
            'product_sku': {
                'pattern': r'^[A-Z]{2}-\d{4}$',
                'required': True,
                'type': 'string'
            },
            'product_name': {
                'required': True,
                'type': 'string',
                'min_length': 1
            },
            'category': {
                'required': True,
                'type': 'string',
                'allowed_values': ['Software Subscription', 'Training', 'Hardware', 'Professional Services']
            },
            'subcategory': {
                'required': True,
                'type': 'string',
                'allowed_values': ['Enterprise', 'SMB', 'Consumer']
            },
            'quantity': {
                'required': True,
                'type': 'integer',
                'min_value': 1
            },
            'unit_price': {
                'required': True,
                'type': 'float',
                'min_value': 0.0
            },
            'currency': {
                'required': True,
                'type': 'string',
                'allowed_values': ['INR', '₹', 'Rs', 'inr']
            },
            'discount_pct': {
                'required': True,
                'type': 'float',
                'min_value': 0.0,
                'max_value': 1.0
            },
            'tax_pct': {
                'required': True,
                'type': 'float',
                'min_value': 0.0,
                'max_value': 1.0
            },
            'shipping_fee': {
                'required': True,
                'type': 'float',
                'min_value': 0.0
            },
            'total_amount': {
                'required': True,
                'type': 'float',
                'min_value': 0.0
            },
            'tax_id': {
                'pattern': r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}$',
                'required': True,
                'type': 'string'
            }
        }
    
    def validate_data(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate data against canonical schema rules.
        
        Args:
            df: DataFrame with source data
            column_mapping: Mapping from source columns to canonical columns
            
        Returns:
            Dict with validation results
        """
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(column_mapping),
            'issues': [],
            'row_issues': {},
            'column_issues': {},
            'summary': {
                'critical_issues': 0,
                'warnings': 0,
                'data_quality_score': 0.0
            }
        }
        
        # Validate each mapped column
        for source_col, canonical_col in column_mapping.items():
            if source_col in df.columns and canonical_col in self.validation_rules:
                column_issues = self._validate_column(
                    df[source_col], 
                    canonical_col, 
                    source_col
                )
                validation_results['column_issues'][canonical_col] = column_issues
                validation_results['issues'].extend(column_issues)
        
        # Validate row-level consistency
        row_issues = self._validate_row_consistency(df, column_mapping)
        validation_results['row_issues'] = row_issues
        validation_results['issues'].extend(row_issues)
        
        # Calculate summary statistics
        validation_results['summary'] = self._calculate_summary(validation_results)
        
        return validation_results
    
    def _validate_column(self, series: pd.Series, canonical_col: str, source_col: str) -> List[Dict[str, Any]]:
        """Validate a single column against its rules."""
        issues = []
        rules = self.validation_rules.get(canonical_col, {})
        
        for idx, value in series.items():
            if pd.isna(value) or value == '' or str(value).lower() in ['nan', 'null', 'none']:
                if rules.get('required', False):
                    issues.append({
                        'type': 'missing_value',
                        'severity': 'critical',
                        'row': idx,
                        'column': canonical_col,
                        'source_column': source_col,
                        'value': value,
                        'message': f'Required field {canonical_col} is missing'
                    })
                continue
            
            # Type validation
            type_issue = self._validate_type(value, rules.get('type'), canonical_col)
            if type_issue:
                issues.append({
                    'type': 'type_mismatch',
                    'severity': 'critical',
                    'row': idx,
                    'column': canonical_col,
                    'source_column': source_col,
                    'value': value,
                    'message': type_issue
                })
                continue
            
            # Pattern validation
            if 'pattern' in rules:
                if not re.match(rules['pattern'], str(value)):
                    issues.append({
                        'type': 'pattern_mismatch',
                        'severity': 'critical',
                        'row': idx,
                        'column': canonical_col,
                        'source_column': source_col,
                        'value': value,
                        'message': f'{canonical_col} does not match expected pattern'
                    })
            
            # Value range validation
            if rules.get('type') in ['integer', 'float']:
                try:
                    num_value = float(value)
                    if 'min_value' in rules and num_value < rules['min_value']:
                        issues.append({
                            'type': 'value_range',
                            'severity': 'warning',
                            'row': idx,
                            'column': canonical_col,
                            'source_column': source_col,
                            'value': value,
                            'message': f'{canonical_col} value {num_value} is below minimum {rules["min_value"]}'
                        })
                    if 'max_value' in rules and num_value > rules['max_value']:
                        issues.append({
                            'type': 'value_range',
                            'severity': 'warning',
                            'row': idx,
                            'column': canonical_col,
                            'source_column': source_col,
                            'value': value,
                            'message': f'{canonical_col} value {num_value} is above maximum {rules["max_value"]}'
                        })
                except (ValueError, TypeError):
                    pass
            
            # Allowed values validation
            if 'allowed_values' in rules:
                if str(value) not in rules['allowed_values']:
                    issues.append({
                        'type': 'invalid_value',
                        'severity': 'warning',
                        'row': idx,
                        'column': canonical_col,
                        'source_column': source_col,
                        'value': value,
                        'message': f'{canonical_col} value "{value}" is not in allowed values: {rules["allowed_values"]}'
                    })
            
            # Length validation
            if 'min_length' in rules:
                if len(str(value)) < rules['min_length']:
                    issues.append({
                        'type': 'length_validation',
                        'severity': 'warning',
                        'row': idx,
                        'column': canonical_col,
                        'source_column': source_col,
                        'value': value,
                        'message': f'{canonical_col} value is too short (minimum {rules["min_length"]} characters)'
                    })
        
        return issues
    
    def _validate_type(self, value: Any, expected_type: str, column_name: str) -> Optional[str]:
        """Validate data type of a value."""
        if expected_type == 'string':
            return None  # Everything can be a string
        elif expected_type == 'integer':
            try:
                int(value)
                return None
            except (ValueError, TypeError):
                return f'{column_name} should be an integer, got {type(value).__name__}'
        elif expected_type == 'float':
            try:
                float(value)
                return None
            except (ValueError, TypeError):
                return f'{column_name} should be a number, got {type(value).__name__}'
        elif expected_type == 'date':
            try:
                datetime.strptime(str(value), '%Y-%m-%d')
                return None
            except ValueError:
                return f'{column_name} should be in YYYY-MM-DD format'
        elif expected_type == 'email':
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if re.match(email_pattern, str(value)):
                return None
            else:
                return f'{column_name} should be a valid email address'
        elif expected_type == 'phone':
            phone_pattern = r'^\+91-\d{10}$'
            if re.match(phone_pattern, str(value)):
                return None
            else:
                return f'{column_name} should be in format +91-XXXXXXXXXX'
        elif expected_type == 'postal_code':
            postal_pattern = r'^\d{6}$'
            if re.match(postal_pattern, str(value)):
                return None
            else:
                return f'{column_name} should be a 6-digit postal code'
        
        return None
    
    def _validate_row_consistency(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """Validate row-level consistency rules."""
        issues = []
        
        # Check for duplicate order IDs
        if 'order_id' in column_mapping.values():
            order_id_col = None
            for source_col, canonical_col in column_mapping.items():
                if canonical_col == 'order_id' and source_col in df.columns:
                    order_id_col = source_col
                    break
            
            if order_id_col:
                duplicates = df[df.duplicated(subset=[order_id_col], keep=False)]
                if not duplicates.empty:
                    for idx in duplicates.index:
                        issues.append({
                            'type': 'duplicate_order_id',
                            'severity': 'critical',
                            'row': idx,
                            'column': 'order_id',
                            'value': df.loc[idx, order_id_col],
                            'message': f'Duplicate order ID: {df.loc[idx, order_id_col]}'
                        })
        
        # Validate pricing consistency
        pricing_columns = ['unit_price', 'quantity', 'discount_pct', 'tax_pct', 'shipping_fee', 'total_amount']
        available_pricing_cols = []
        
        for source_col, canonical_col in column_mapping.items():
            if canonical_col in pricing_columns and source_col in df.columns:
                available_pricing_cols.append((source_col, canonical_col))
        
        if len(available_pricing_cols) >= 3:  # Need at least unit_price, quantity, and total_amount
            for idx, row in df.iterrows():
                try:
                    # Try to calculate expected total
                    unit_price = None
                    quantity = None
                    discount_pct = 0.0
                    tax_pct = 0.0
                    shipping_fee = 0.0
                    total_amount = None
                    
                    for source_col, canonical_col in available_pricing_cols:
                        value = row[source_col]
                        if pd.notna(value) and str(value).lower() not in ['nan', 'null', 'none']:
                            if canonical_col == 'unit_price':
                                unit_price = float(str(value).replace(',', '').replace('₹', '').replace('Rs', '').replace('INR', '').strip())
                            elif canonical_col == 'quantity':
                                quantity = int(float(value))
                            elif canonical_col == 'discount_pct':
                                discount_pct = float(str(value).replace('%', '')) / 100.0 if '%' in str(value) else float(value)
                            elif canonical_col == 'tax_pct':
                                tax_pct = float(str(value).replace('%', '')) / 100.0 if '%' in str(value) else float(value)
                            elif canonical_col == 'shipping_fee':
                                shipping_fee = float(str(value).replace(',', '').replace('₹', '').replace('Rs', '').replace('INR', '').strip())
                            elif canonical_col == 'total_amount':
                                total_amount = float(str(value).replace(',', '').replace('₹', '').replace('Rs', '').replace('INR', '').strip())
                    
                    if unit_price is not None and quantity is not None and total_amount is not None:
                        # Calculate expected total: (unit_price * quantity * (1 - discount_pct) * (1 + tax_pct)) + shipping_fee
                        expected_total = (unit_price * quantity * (1 - discount_pct) * (1 + tax_pct)) + shipping_fee
                        
                        # Allow 1% tolerance for rounding differences
                        if abs(expected_total - total_amount) > (total_amount * 0.01):
                            issues.append({
                                'type': 'pricing_inconsistency',
                                'severity': 'warning',
                                'row': idx,
                                'column': 'total_amount',
                                'value': total_amount,
                                'expected_value': expected_total,
                                'message': f'Total amount {total_amount} does not match calculated value {expected_total:.2f}'
                            })
                
                except (ValueError, TypeError, ZeroDivisionError):
                    # Skip rows with invalid numeric data
                    pass
        
        return issues
    
    def _calculate_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for validation results."""
        issues = validation_results['issues']
        
        critical_issues = len([i for i in issues if i.get('severity') == 'critical'])
        warnings = len([i for i in issues if i.get('severity') == 'warning'])
        
        total_checks = validation_results['total_rows'] * validation_results['total_columns']
        total_issues = len(issues)
        
        # Calculate data quality score (0-100)
        if total_checks > 0:
            data_quality_score = max(0, 100 - (total_issues / total_checks) * 100)
        else:
            data_quality_score = 100.0
        
        return {
            'critical_issues': critical_issues,
            'warnings': warnings,
            'total_issues': total_issues,
            'data_quality_score': round(data_quality_score, 2),
            'issues_by_type': self._group_issues_by_type(issues)
        }
    
    def _group_issues_by_type(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group issues by type for summary."""
        issue_types = {}
        for issue in issues:
            issue_type = issue.get('type', 'unknown')
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        return issue_types
    
    def get_validation_rules(self, column_name: str) -> Dict[str, Any]:
        """Get validation rules for a specific column."""
        return self.validation_rules.get(column_name, {})
    
    def add_custom_rule(self, column_name: str, rule: Dict[str, Any]) -> None:
        """Add a custom validation rule for a column."""
        self.validation_rules[column_name] = rule
        logger.info(f"Added custom validation rule for column: {column_name}")
    
    def remove_rule(self, column_name: str) -> None:
        """Remove validation rule for a column."""
        if column_name in self.validation_rules:
            del self.validation_rules[column_name]
            logger.info(f"Removed validation rule for column: {column_name}")