# Schema Mapper & Data Quality Fixer

A comprehensive FastAPI application that provides intelligent data mapping, cleaning, and quality improvement capabilities. The system suggests mappings to a canonical schema, runs deterministic cleaning/validation, and provides targeted AI-powered fix suggestions for remaining issues.

## Features

### 🎯 Core Capabilities
- **Just drop your file**: Get suggested mapping with confidence scores and easy overrides
- **One click to clean**: Run clean & validate with clear before/after reports
- **Help on leftovers**: Show targeted fix suggestions you can apply
- **Learn as we go**: System remembers accepted fixes for future files
- **No cost surprises**: Predictable behavior with surgical AI usage

### 🔧 Technical Features
- **Intelligent Header Mapping**: Uses LangChain + Groq for fuzzy matching and AI-powered suggestions
- **Deterministic Data Validation**: Comprehensive validation rules for all data types
- **Value Normalization**: Automatic normalization of dates, currencies, percentages, and text
- **Targeted Fix Suggestions**: AI-powered suggestions for remaining data quality issues
- **Learning System**: Remembers and reuses accepted fixes across sessions
- **Modern Web UI**: Beautiful, responsive interface for easy interaction

## Architecture

### Components

1. **Schema Loader** (`src/schema_loader.py`)
   - Loads and manages canonical schema definitions
   - Provides column information and validation rules

2. **Header Mapper** (`src/header_mapper.py`)
   - Intelligent column mapping using fuzzy matching and AI
   - Supports both deterministic patterns and AI-powered suggestions
   - Confidence scoring for all mappings

3. **Data Validator** (`src/data_validator.py`)
   - Comprehensive validation rules for all canonical columns
   - Row-level consistency checks
   - Data quality scoring

4. **Value Normalizer** (`src/value_normalizer.py`)
   - Normalizes data values to canonical format
   - Handles dates, currencies, percentages, phone numbers, etc.
   - Provides normalization summaries

5. **Fix Suggester** (`src/fix_suggester.py`)
   - AI-powered fix suggestions for remaining issues
   - Deterministic fix patterns for common problems
   - Confidence scoring and implementation guidance

6. **Learning System** (`src/learning_system.py`)
   - Remembers user preferences and accepted fixes
   - Improves mapping accuracy over time
   - Tracks success metrics

## Installation

### Prerequisites
- Python 3.8+
- Groq API key (optional, for AI features)

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd /workspace
   ```

2. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt --break-system-packages
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

4. **Run the application:**
   ```bash
   python3 main.py
   ```

5. **Access the web interface:**
   Open http://localhost:8000 in your browser

## Usage

### Web Interface

1. **Upload File**: Drag and drop or select a CSV file
2. **Review Mapping**: Check suggested column mappings with confidence scores
3. **Apply Cleaning**: Click "Apply Cleaning" to run validation and normalization
4. **Review Issues**: See detailed issues found and fix suggestions
5. **Apply Fixes**: Apply suggested fixes with one click

### API Endpoints

- `GET /` - Web interface
- `POST /analyze` - Analyze uploaded file and suggest mappings
- `POST /clean` - Apply cleaning and validation
- `POST /apply-fix` - Apply a specific fix
- `GET /schema` - Get canonical schema information
- `GET /learning-summary` - Get learning system summary
- `GET /health` - Health check

### Sample Data

The application includes three sample datasets to test different scenarios:

1. **Project6InputData1.csv** - Clean, canonical-like headers
2. **Project6InputData2.csv** - Messy headers, mixed formats, currency symbols
3. **Project6InputData3.csv** - Different headers, missing columns, extra columns

## Configuration

### Canonical Schema

The canonical schema is defined in `canonicalSchema/Project6StdFormat.csv` with the following structure:
- `canonical_name`: The standard column name
- `description`: Description of the column
- `example`: Example value

### Validation Rules

Validation rules are defined in `src/data_validator.py` for each canonical column:
- Pattern matching (regex)
- Data type validation
- Value range checks
- Required field validation
- Custom business rules

### Learning Configuration

The learning system can be configured in `src/learning_system.py`:
- Minimum confidence threshold
- Minimum occurrences for learning
- Maximum learning history
- Data retention period

## Testing

Run the test suite to verify functionality:

```bash
python3 simple_test.py
```

This will test:
- Schema loading
- Header mapping (with and without AI)
- Data validation
- Value normalization
- Learning system
- Sample file processing

## API Examples

### Analyze File
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sampleDataset/Project6InputData1.csv"
```

### Apply Cleaning
```bash
curl -X POST "http://localhost:8000/clean" \
  -H "Content-Type: application/json" \
  -d '{
    "suggested_mapping": {
      "Order No": "order_id",
      "OrderDate": "order_date"
    },
    "confidence_scores": {
      "Order No": 0.95,
      "OrderDate": 0.90
    }
  }'
```

### Get Schema
```bash
curl http://localhost:8000/schema
```

## Development

### Project Structure
```
/workspace/
├── main.py                 # FastAPI application
├── requirements.txt        # Dependencies
├── .env.example           # Environment variables template
├── src/                   # Source code
│   ├── __init__.py
│   ├── schema_loader.py
│   ├── header_mapper.py
│   ├── data_validator.py
│   ├── value_normalizer.py
│   ├── fix_suggester.py
│   └── learning_system.py
├── canonicalSchema/       # Canonical schema definition
│   └── Project6StdFormat.csv
├── sampleDataset/         # Sample data files
│   ├── Project6InputData1.csv
│   ├── Project6InputData2.csv
│   └── Project6InputData3.csv
└── README.md             # This file
```

### Adding New Validation Rules

1. Edit `src/data_validator.py`
2. Add rules to the `validation_rules` dictionary
3. Define patterns, types, and constraints
4. Test with sample data

### Adding New Normalization Rules

1. Edit `src/value_normalizer.py`
2. Add new normalization methods
3. Update the `normalize_column` method
4. Test with various data formats

### Extending Learning System

1. Edit `src/learning_system.py`
2. Add new learning patterns
3. Update success metrics
4. Configure retention policies

## Troubleshooting

### Common Issues

1. **Schema not loading**: Check file paths and permissions
2. **AI features not working**: Verify GROQ_API_KEY is set
3. **Import errors**: Ensure all dependencies are installed
4. **Port conflicts**: Change port in main.py if 8000 is occupied

### Debug Mode

Enable debug logging by setting the log level in `main.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test suite
3. Check the API documentation
4. Create an issue with detailed information
