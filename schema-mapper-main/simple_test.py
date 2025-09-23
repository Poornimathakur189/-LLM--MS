"""
Simple test script for the Schema Mapper & Data Quality Fixer application.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_schema_loader():
    """Test the schema loader."""
    print("Testing Schema Loader...")
    
    try:
        from app.schema_loader import SchemaLoader
        schema_loader = SchemaLoader()
        success = schema_loader.load_schema("canonicalSchema/Project6StdFormat.csv")
        
        if success:
            print("✅ Schema loaded successfully")
            print(f"   Total columns: {len(schema_loader.get_all_columns())}")
            print(f"   Sample columns: {schema_loader.get_all_columns()[:5]}")
            return schema_loader
        else:
            print("❌ Failed to load schema")
            return None
    except Exception as e:
        print(f"❌ Error testing schema loader: {e}")
        return None

def test_header_mapper(schema_loader):
    """Test the header mapper."""
    print("\nTesting Header Mapper...")
    
    try:
        from app.header_mapper import HeaderMapper
        header_mapper = HeaderMapper(schema_loader)
        
        # Test with different header sets
        test_headers = [
            # Clean headers (like Project6InputData1.csv)
            ['order_id', 'order_date', 'customer_id', 'customer_name', 'email', 'phone'],
            
            # Messy headers (like Project6InputData2.csv)
            ['Order No', 'OrderDate', 'Cust ID', 'Customer', 'E-mail', 'Phone #'],
            
            # Different headers (like Project6InputData3.csv)
            ['reference', 'ordered_on', 'client_ref', 'client_name', 'contact', 'mobile']
        ]
        
        for i, headers in enumerate(test_headers, 1):
            print(f"\n   Test {i}: {len(headers)} headers")
            result = header_mapper.suggest_mapping(headers)
            
            print(f"   Mapped: {len(result['mapping'])} columns")
            print(f"   Unmapped: {len(result['unmapped_columns'])} columns")
            
            if result['mapping']:
                print("   Sample mappings:")
                for source, target in list(result['mapping'].items())[:3]:
                    confidence = result['confidence_scores'].get(source, 0)
                    print(f"     {source} → {target} ({confidence:.2f})")
        
        return header_mapper
    except Exception as e:
        print(f"❌ Error testing header mapper: {e}")
        return None

def test_data_validator():
    """Test the data validator."""
    print("\nTesting Data Validator...")
    
    try:
        from app.data_validator import DataValidator
        validator = DataValidator()
        
        # Create test data
        test_data = pd.DataFrame({
            'order_id': ['ORD-1001', 'ORD-1002', 'INVALID', 'ORD-1004'],
            'order_date': ['2025-01-15', '2025-01-16', 'invalid-date', '2025-01-18'],
            'email': ['test@example.com', 'invalid-email', 'test2@example.com', ''],
            'quantity': [1, 2, 3, -1],  # Negative quantity should be flagged
            'unit_price': [100.0, 200.0, 300.0, 400.0]
        })
        
        column_mapping = {
            'order_id': 'order_id',
            'order_date': 'order_date', 
            'email': 'email',
            'quantity': 'quantity',
            'unit_price': 'unit_price'
        }
        
        validation_results = validator.validate_data(test_data, column_mapping)
        
        print(f"✅ Validation completed")
        print(f"   Total issues: {len(validation_results['issues'])}")
        print(f"   Critical issues: {validation_results['summary']['critical_issues']}")
        print(f"   Warnings: {validation_results['summary']['warnings']}")
        print(f"   Data quality score: {validation_results['summary']['data_quality_score']}%")
        
        return validator
    except Exception as e:
        print(f"❌ Error testing data validator: {e}")
        return None

def test_value_normalizer():
    """Test the value normalizer."""
    print("\nTesting Value Normalizer...")
    
    try:
        from app.value_normalizer import ValueNormalizer
        normalizer = ValueNormalizer()
        
        # Create test data with various formats
        test_data = pd.DataFrame({
            'dates': ['2025-01-15', '15/01/2025', '15 Jan 2025', 'invalid-date'],
            'currency': ['₹1,000.00', 'Rs 2000', 'INR 3000.50', '4000'],
            'percentages': ['10%', '0.15', '20%', '0.25'],
            'phones': ['+91-9876543210', '9876543210', '+91 9876543210', 'invalid-phone']
        })
        
        column_mapping = {
            'dates': 'order_date',
            'currency': 'unit_price',
            'percentages': 'discount_pct',
            'phones': 'phone'
        }
        
        normalized_data = normalizer.normalize_data(test_data, column_mapping)
        
        print("✅ Normalization completed")
        print("   Sample normalized values:")
        for col in normalized_data.columns:
            print(f"     {col}: {list(normalized_data[col].head(2))}")
        
        return normalizer
    except Exception as e:
        print(f"❌ Error testing value normalizer: {e}")
        return None

def test_learning_system():
    """Test the learning system."""
    print("\nTesting Learning System...")
    
    try:
        from app.learning_system import LearningSystem
        learning_system = LearningSystem("test_learning_data.json")
        
        # Test learning from mappings
        learning_system.learn_from_mapping("Order No", "order_id", 0.9, False)
        learning_system.learn_from_mapping("Order No", "order_id", 0.8, False)
        learning_system.learn_from_mapping("Order No", "order_id", 0.95, True)  # User override
        
        # Test learning from fixes
        learning_system.learn_from_fix("fix_001", True, None)
        learning_system.learn_from_fix("fix_001", True, None)
        learning_system.learn_from_fix("fix_001", False, None)
        
        # Test retrieving learned mappings
        learned_mapping = learning_system.get_learned_mapping("Order No")
        if learned_mapping:
            canonical, confidence = learned_mapping
            print(f"✅ Learned mapping: Order No → {canonical} (confidence: {confidence:.2f})")
        else:
            print("❌ No learned mapping found")
        
        # Get learning summary
        summary = learning_system.get_learning_summary()
        print(f"✅ Learning summary: {summary['total_mapping_patterns']} patterns, {summary['total_fix_patterns']} fixes")
        
        # Clean up test file
        if Path("test_learning_data.json").exists():
            Path("test_learning_data.json").unlink()
        
        return learning_system
    except Exception as e:
        print(f"❌ Error testing learning system: {e}")
        return None

def test_with_sample_files():
    """Test with actual sample files."""
    print("\nTesting with Sample Files...")
    
    try:
        from app.schema_loader import SchemaLoader
        from app.header_mapper import HeaderMapper
        
        schema_loader = SchemaLoader()
        schema_loader.load_schema("canonicalSchema/Project6StdFormat.csv")
        
        header_mapper = HeaderMapper(schema_loader)
        
        sample_files = [
            "sampleDataset/Project6InputData1.csv",
            "sampleDataset/Project6InputData2.csv", 
            "sampleDataset/Project6InputData3.csv"
        ]
        
        for file_path in sample_files:
            if Path(file_path).exists():
                print(f"\n   Testing {file_path}...")
                
                # Read the file
                df = pd.read_csv(file_path)
                headers = df.columns.tolist()
                
                # Test mapping
                mapping_result = header_mapper.suggest_mapping(headers)
                
                print(f"     Rows: {len(df)}, Columns: {len(headers)}")
                print(f"     Mapped: {len(mapping_result['mapping'])} columns")
                print(f"     Unmapped: {len(mapping_result['unmapped_columns'])} columns")
                
                if mapping_result['unmapped_columns']:
                    print(f"     Unmapped columns: {mapping_result['unmapped_columns']}")
            else:
                print(f"   ❌ File not found: {file_path}")
    except Exception as e:
        print(f"❌ Error testing with sample files: {e}")

def main():
    """Run all tests."""
    print("🧪 Testing Schema Mapper & Data Quality Fixer")
    print("=" * 50)
    
    try:
        # Test individual components
        schema_loader = test_schema_loader()
        if schema_loader:
            header_mapper = test_header_mapper(schema_loader)
            validator = test_data_validator()
            normalizer = test_value_normalizer()
            learning_system = test_learning_system()
            
            # Test with sample files
            test_with_sample_files()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("\nTo run the application:")
        print("1. Set your GROQ_API_KEY in .env file")
        print("2. Run: python3 main.py")
        print("3. Open: http://localhost:8000")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()