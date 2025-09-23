"""
Value Normalizer - Normalizes data values to canonical format.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ValueNormalizer:
    """Normalizes data values to canonical format."""
    
    def __init__(self):
        # Date format patterns
        self.date_patterns = [
            (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),  # 2025-01-15
            (r'^\d{2}/\d{2}/\d{4}$', '%d/%m/%Y'),  # 15/01/2025
            (r'^\d{2}-\d{2}-\d{4}$', '%d-%m-%Y'),  # 15-01-2025
            (r'^\d{2}/\d{2}/\d{4}$', '%m/%d/%Y'),  # 01/15/2025 (US format)
            (r'^\d{1,2}-\w{3}-\d{4}$', '%d-%b-%Y'),  # 15-Jan-2025
            (r'^\d{1,2}\s\w{3}\s\d{4}$', '%d %b %Y'),  # 15 Jan 2025
            (r'^\w{3}-\d{2}-\d{4}$', '%b-%d-%Y'),  # Jan-15-2025
        ]
        
        # Currency symbols and their ISO codes
        self.currency_mapping = {
            '₹': 'INR',
            'Rs': 'INR',
            'rs': 'INR',
            'INR': 'INR',
            'inr': 'INR',
            '$': 'USD',
            'USD': 'USD',
            'usd': 'USD',
            '€': 'EUR',
            'EUR': 'EUR',
            'eur': 'EUR',
            '£': 'GBP',
            'GBP': 'GBP',
            'gbp': 'GBP'
        }
        
        # Common text normalizations
        self.text_replacements = {
            # Whitespace normalization
            r'\s+': ' ',  # Multiple spaces to single space
            r'^\s+|\s+$': '',  # Trim leading/trailing spaces
            
            # Common abbreviations
            r'\bSt\b': 'Street',
            r'\bAve\b': 'Avenue',
            r'\bRd\b': 'Road',
            r'\bBlvd\b': 'Boulevard',
            r'\bDr\b': 'Drive',
            r'\bLn\b': 'Lane',
            r'\bCt\b': 'Court',
            r'\bPl\b': 'Place',
            
            # Name formatting
            r'\bMc([A-Z])': r'Mc\1',  # Preserve Mc/Mac prefixes
            r'\bO\'([A-Z])': r'O\'\1',  # Preserve O' prefixes
        }
        
        # Phone number patterns
        self.phone_patterns = [
            r'^\+91-(\d{10})$',  # +91-XXXXXXXXXX
            r'^(\d{10})$',  # XXXXXXXXXX
            r'^\+91\s*(\d{10})$',  # +91 XXXXXXXXXX
            r'^\+91\s*(\d{5})\s*(\d{5})$',  # +91 XXXXX XXXXX
        ]
    
    def normalize_data(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Normalize all data in the DataFrame according to canonical schema.
        
        Args:
            df: Source DataFrame
            column_mapping: Mapping from source columns to canonical columns
            
        Returns:
            Normalized DataFrame
        """
        normalized_df = df.copy()
        
        for source_col, canonical_col in column_mapping.items():
            if source_col in normalized_df.columns:
                normalized_df[source_col] = self.normalize_column(
                    normalized_df[source_col], 
                    canonical_col
                )
        
        return normalized_df
    
    def normalize_column(self, series: pd.Series, canonical_col: str) -> pd.Series:
        """
        Normalize a single column based on its canonical type.
        
        Args:
            series: Pandas Series to normalize
            canonical_col: Canonical column name
            
        Returns:
            Normalized Series
        """
        if canonical_col in ['order_date']:
            return self._normalize_dates(series)
        elif canonical_col in ['unit_price', 'shipping_fee', 'total_amount']:
            return self._normalize_currency(series)
        elif canonical_col in ['discount_pct', 'tax_pct']:
            return self._normalize_percentages(series)
        elif canonical_col in ['quantity']:
            return self._normalize_integers(series)
        elif canonical_col in ['currency']:
            return self._normalize_currency_codes(series)
        elif canonical_col in ['phone']:
            return self._normalize_phone_numbers(series)
        elif canonical_col in ['postal_code']:
            return self._normalize_postal_codes(series)
        elif canonical_col in ['email']:
            return self._normalize_emails(series)
        elif canonical_col in ['customer_name', 'product_name', 'billing_address', 'shipping_address', 'city', 'state']:
            return self._normalize_text(series)
        elif canonical_col in ['order_id', 'customer_id', 'product_sku', 'tax_id']:
            return self._normalize_ids(series)
        else:
            return self._normalize_text(series)
    
    def _normalize_dates(self, series: pd.Series) -> pd.Series:
        """Normalize date values to YYYY-MM-DD format."""
        def normalize_date(value):
            if pd.isna(value) or str(value).lower() in ['nan', 'null', 'none', '']:
                return np.nan
            
            value_str = str(value).strip()
            
            # Try each pattern
            for pattern, date_format in self.date_patterns:
                if re.match(pattern, value_str):
                    try:
                        parsed_date = datetime.strptime(value_str, date_format)
                        return parsed_date.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
            
            # If no pattern matches, try pandas to_datetime
            try:
                parsed_date = pd.to_datetime(value_str)
                return parsed_date.strftime('%Y-%m-%d')
            except:
                logger.warning(f"Could not parse date: {value_str}")
                return value_str  # Return original if can't parse
        
        return series.apply(normalize_date)
    
    def _normalize_currency(self, series: pd.Series) -> pd.Series:
        """Normalize currency values to float."""
        def normalize_currency(value):
            if pd.isna(value) or str(value).lower() in ['nan', 'null', 'none', '']:
                return np.nan
            
            value_str = str(value).strip()
            
            # Remove currency symbols and commas
            for symbol in ['₹', 'Rs', 'rs', 'INR', 'inr', '$', 'USD', 'usd', '€', 'EUR', 'eur', '£', 'GBP', 'gbp']:
                value_str = value_str.replace(symbol, '')
            
            # Remove commas and extra spaces
            value_str = value_str.replace(',', '').strip()
            
            # Handle quoted values
            if value_str.startswith('"') and value_str.endswith('"'):
                value_str = value_str[1:-1]
            
            try:
                return float(value_str)
            except ValueError:
                logger.warning(f"Could not parse currency value: {value}")
                return np.nan
        
        return series.apply(normalize_currency)
    
    def _normalize_percentages(self, series: pd.Series) -> pd.Series:
        """Normalize percentage values to decimal (0-1 range)."""
        def normalize_percentage(value):
            if pd.isna(value) or str(value).lower() in ['nan', 'null', 'none', '']:
                return np.nan
            
            value_str = str(value).strip()
            
            # Remove percentage sign
            if value_str.endswith('%'):
                value_str = value_str[:-1]
            
            try:
                num_value = float(value_str)
                # If value is > 1, assume it's a percentage (e.g., 10% = 0.1)
                if num_value > 1:
                    return num_value / 100.0
                else:
                    return num_value
            except ValueError:
                logger.warning(f"Could not parse percentage value: {value}")
                return np.nan
        
        return series.apply(normalize_percentage)
    
    def _normalize_integers(self, series: pd.Series) -> pd.Series:
        """Normalize integer values."""
        def normalize_integer(value):
            if pd.isna(value) or str(value).lower() in ['nan', 'null', 'none', '']:
                return np.nan
            
            try:
                return int(float(str(value).replace(',', '')))
            except ValueError:
                logger.warning(f"Could not parse integer value: {value}")
                return np.nan
        
        return series.apply(normalize_integer)
    
    def _normalize_currency_codes(self, series: pd.Series) -> pd.Series:
        """Normalize currency codes to standard format."""
        def normalize_currency_code(value):
            if pd.isna(value) or str(value).lower() in ['nan', 'null', 'none', '']:
                return np.nan
            
            value_str = str(value).strip().upper()
            return self.currency_mapping.get(value_str, value_str)
        
        return series.apply(normalize_currency_code)
    
    def _normalize_phone_numbers(self, series: pd.Series) -> pd.Series:
        """Normalize phone numbers to +91-XXXXXXXXXX format."""
        def normalize_phone(value):
            if pd.isna(value) or str(value).lower() in ['nan', 'null', 'none', '']:
                return np.nan
            
            value_str = str(value).strip()
            
            # Try each pattern
            for pattern in self.phone_patterns:
                match = re.match(pattern, value_str)
                if match:
                    if len(match.groups()) == 1:
                        return f"+91-{match.group(1)}"
                    elif len(match.groups()) == 2:
                        return f"+91-{match.group(1)}{match.group(2)}"
            
            # If no pattern matches, try to extract digits
            digits = re.sub(r'\D', '', value_str)
            if len(digits) == 10:
                return f"+91-{digits}"
            elif len(digits) == 12 and digits.startswith('91'):
                return f"+91-{digits[2:]}"
            
            logger.warning(f"Could not normalize phone number: {value_str}")
            return value_str
        
        return series.apply(normalize_phone)
    
    def _normalize_postal_codes(self, series: pd.Series) -> pd.Series:
        """Normalize postal codes to 6-digit format."""
        def normalize_postal_code(value):
            if pd.isna(value) or str(value).lower() in ['nan', 'null', 'none', '']:
                return np.nan
            
            value_str = str(value).strip()
            
            # Extract digits
            digits = re.sub(r'\D', '', value_str)
            
            # Handle cases like "667XX2" - replace X with 0
            if 'X' in value_str.upper():
                digits = re.sub(r'[^0-9]', '0', value_str)
            
            if len(digits) == 6:
                return digits
            elif len(digits) > 6:
                return digits[:6]  # Take first 6 digits
            else:
                # Pad with zeros if less than 6 digits
                return digits.zfill(6)
        
        return series.apply(normalize_postal_code)
    
    def _normalize_emails(self, series: pd.Series) -> pd.Series:
        """Normalize email addresses."""
        def normalize_email(value):
            if pd.isna(value) or str(value).lower() in ['nan', 'null', 'none', '']:
                return np.nan
            
            value_str = str(value).strip().lower()
            
            # Remove extra spaces
            value_str = re.sub(r'\s+', '', value_str)
            
            # Basic email validation
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if re.match(email_pattern, value_str):
                return value_str
            else:
                logger.warning(f"Invalid email format: {value_str}")
                return value_str  # Return as-is for manual review
        
        return series.apply(normalize_email)
    
    def _normalize_text(self, series: pd.Series) -> pd.Series:
        """Normalize text values."""
        def normalize_text(value):
            if pd.isna(value) or str(value).lower() in ['nan', 'null', 'none', '']:
                return np.nan
            
            value_str = str(value).strip()
            
            # Apply text replacements
            for pattern, replacement in self.text_replacements.items():
                value_str = re.sub(pattern, replacement, value_str)
            
            # Title case for names and addresses
            if any(keyword in value_str.lower() for keyword in ['street', 'avenue', 'road', 'boulevard', 'drive', 'lane', 'court', 'place']):
                # Don't title case addresses
                pass
            else:
                # Title case for names
                value_str = value_str.title()
            
            return value_str
        
        return series.apply(normalize_text)
    
    def _normalize_ids(self, series: pd.Series) -> pd.Series:
        """Normalize ID fields."""
        def normalize_id(value):
            if pd.isna(value) or str(value).lower() in ['nan', 'null', 'none', '']:
                return np.nan
            
            value_str = str(value).strip().upper()
            
            # Remove extra spaces
            value_str = re.sub(r'\s+', '', value_str)
            
            return value_str
        
        return series.apply(normalize_id)
    
    def get_normalization_summary(self, original_df: pd.DataFrame, normalized_df: pd.DataFrame, column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Get a summary of normalization changes.
        
        Args:
            original_df: Original DataFrame
            normalized_df: Normalized DataFrame
            column_mapping: Column mapping
            
        Returns:
            Dict with normalization summary
        """
        summary = {
            'total_columns_normalized': len(column_mapping),
            'changes_by_column': {},
            'total_changes': 0
        }
        
        for source_col, canonical_col in column_mapping.items():
            if source_col in original_df.columns and source_col in normalized_df.columns:
                original_series = original_df[source_col]
                normalized_series = normalized_df[source_col]
                
                # Count changes
                changes = 0
                for idx in range(len(original_series)):
                    if pd.notna(original_series.iloc[idx]) and pd.notna(normalized_series.iloc[idx]):
                        if str(original_series.iloc[idx]) != str(normalized_series.iloc[idx]):
                            changes += 1
                
                summary['changes_by_column'][canonical_col] = {
                    'source_column': source_col,
                    'changes_made': changes,
                    'total_values': len(original_series.dropna())
                }
                summary['total_changes'] += changes
        
        return summary