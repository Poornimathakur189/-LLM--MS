"""
Schema Loader - Loads and manages the canonical schema definition.
"""

import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SchemaLoader:
    """Loads and manages the canonical schema definition."""
    
    def __init__(self):
        self.schema_df: Optional[pd.DataFrame] = None
        self.schema_dict: Dict[str, Dict[str, str]] = {}
        self.column_names: List[str] = []
        
    def load_schema(self, schema_path: str) -> bool:
        """
        Load the canonical schema from CSV file.
        
        Args:
            schema_path: Path to the canonical schema CSV file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            schema_path = Path(schema_path)
            if not schema_path.exists():
                logger.error(f"Schema file not found: {schema_path}")
                return False
                
            # Load the schema CSV
            self.schema_df = pd.read_csv(schema_path)
            
            # Validate required columns
            required_columns = ['canonical_name', 'description', 'example']
            missing_columns = [col for col in required_columns if col not in self.schema_df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in schema: {missing_columns}")
                return False
            
            # Build schema dictionary
            self.schema_dict = {}
            self.column_names = []
            
            for _, row in self.schema_df.iterrows():
                canonical_name = row['canonical_name']
                self.schema_dict[canonical_name] = {
                    'description': row['description'],
                    'example': row['example']
                }
                self.column_names.append(canonical_name)
            
            logger.info(f"Loaded schema with {len(self.column_names)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Error loading schema: {e}")
            return False
    
    def get_column_info(self, column_name: str) -> Optional[Dict[str, str]]:
        """
        Get information about a specific column.
        
        Args:
            column_name: Name of the canonical column
            
        Returns:
            Dict with description and example, or None if not found
        """
        return self.schema_dict.get(column_name)
    
    def get_all_columns(self) -> List[str]:
        """Get list of all canonical column names."""
        return self.column_names.copy()
    
    def get_schema_summary(self) -> Dict[str, any]:
        """
        Get a summary of the loaded schema.
        
        Returns:
            Dict with schema summary information
        """
        if not self.schema_df is not None:
            return {
                'total_columns': len(self.column_names),
                'columns': self.column_names,
                'schema_loaded': True
            }
        else:
            return {
                'total_columns': 0,
                'columns': [],
                'schema_loaded': False
            }
    
    def validate_column_exists(self, column_name: str) -> bool:
        """
        Check if a column exists in the canonical schema.
        
        Args:
            column_name: Name of the column to check
            
        Returns:
            bool: True if column exists, False otherwise
        """
        return column_name in self.schema_dict
    
    def get_columns_by_type(self, data_type: str) -> List[str]:
        """
        Get columns that might be of a specific data type based on their names.
        
        Args:
            data_type: Type to search for (e.g., 'date', 'id', 'amount', 'email')
            
        Returns:
            List of column names that match the type
        """
        matching_columns = []
        data_type_lower = data_type.lower()
        
        for column in self.column_names:
            if data_type_lower in column.lower():
                matching_columns.append(column)
        
        return matching_columns
    
    def get_related_columns(self, column_name: str) -> List[str]:
        """
        Get columns that are semantically related to the given column.
        
        Args:
            column_name: Name of the column to find related columns for
            
        Returns:
            List of related column names
        """
        related = []
        column_lower = column_name.lower()
        
        # Define relationship patterns
        relationships = {
            'customer': ['customer_id', 'customer_name', 'email', 'phone'],
            'order': ['order_id', 'order_date', 'total_amount'],
            'product': ['product_sku', 'product_name', 'category', 'subcategory'],
            'address': ['billing_address', 'shipping_address', 'city', 'state', 'postal_code', 'country'],
            'pricing': ['unit_price', 'quantity', 'discount_pct', 'tax_pct', 'shipping_fee', 'total_amount'],
            'contact': ['email', 'phone'],
            'location': ['city', 'state', 'postal_code', 'country']
        }
        
        for category, columns in relationships.items():
            if category in column_lower or any(col in column_lower for col in columns):
                related.extend([col for col in columns if col != column_name])
        
        return list(set(related))  # Remove duplicates