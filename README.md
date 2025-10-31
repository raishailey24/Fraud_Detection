# 🔍 Fraud Analytics Dashboard with Gen-AI Copilot

A production-ready proof-of-concept that integrates Business Intelligence with Generative AI (OpenAI/Claude) for advanced fraud analytics. Built with **Python + Streamlit** for interactive data exploration and AI-powered insights.

## 🎯 Features

### BI Dashboard
- **Interactive KPI Cards**: Real-time fraud metrics, transaction volumes, and detection accuracy
- **Advanced Visualizations**: Time series analysis, distribution plots, category breakdowns, and confusion matrices
- **Dynamic Filters**: Date range, amount, category, merchant, and risk level filtering
- **Feature Engineering**: Automated risk scoring, temporal patterns, merchant statistics, and user behavior analysis

### Gen-AI Copilot
- **Executive Summaries**: AI-generated insights from aggregated metrics (no PII)
- **What-If Analysis**: Scenario planning and impact projections
- **Detection Rules**: Automated fraud rule proposals based on data patterns
- **Custom Queries**: Natural language Q&A about fraud data

### Privacy & Security
- **PII Protection**: Only aggregated metrics sent to AI APIs
- **Configurable Providers**: Support for both OpenAI and Anthropic (Claude)
- **Environment-based Config**: Secure API key management via `.env`

## 📁 Project Structure

```
fraud_genai_bi/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration and environment settings
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── generate_sample_data.py    # Sample data generator
├── README.md                  # This file
│
├── ai/                        # Gen-AI integration layer
│   ├── __init__.py
│   ├── base_provider.py       # Abstract AI provider interface
│   ├── openai_provider.py     # OpenAI implementation
│   ├── anthropic_provider.py  # Anthropic (Claude) implementation
│   └── ai_copilot.py          # AI orchestrator
│
├── utils/                     # Data processing utilities
│   ├── __init__.py
│   ├── data_loader.py         # File loading and validation
│   └── data_processor.py      # Cleaning and feature engineering
│
├── components/                # Dashboard UI components
│   ├── __init__.py
│   ├── kpi_cards.py           # KPI display components
│   ├── visualizations.py      # Plotly charts
│   ├── filters.py             # Filter controls
│   └── ai_panel.py            # AI Copilot interface
│
└── data/                      # Data directory
    └── sample_transactions.csv # Sample dataset
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd fraud_genai_bi

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy the example environment file
copy .env.example .env  # Windows
# cp .env.example .env  # macOS/Linux

# Edit .env and add your API key
# For OpenAI:
OPENAI_API_KEY=sk-your-openai-key-here
AI_PROVIDER=openai

# OR for Anthropic (Claude):
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
AI_PROVIDER=anthropic
```

### 3. Generate Sample Data

```bash
# Generate synthetic transaction data
python generate_sample_data.py
```

This creates `data/sample_transactions.csv` with 10,000 transactions including fraud cases.

### 4. Run the Application

```bash
# Start the Streamlit dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Data Format

Your transaction data should include these columns:

### Required Columns
- `transaction_id`: Unique transaction identifier (string)
- `timestamp`: Transaction date/time (datetime)
- `amount`: Transaction amount (float)
- `merchant`: Merchant name (string)
- `category`: Transaction category (string)
- `is_fraud`: Fraud label - 0 for legitimate, 1 for fraud (int)

### Optional Columns
- `user_id`: User identifier (string)
- `location`: Transaction location (string)

### Example CSV Format

```csv
transaction_id,timestamp,amount,merchant,category,is_fraud,user_id,location
TXN00000001,2024-01-15 14:23:00,45.99,amazon,retail,0,USER0123,new york
TXN00000002,2024-01-15 15:45:00,1250.00,bestbuy,retail,1,USER0456,international
TXN00000003,2024-01-15 16:12:00,8.50,starbucks,restaurant,0,USER0123,new york
```

## 🔧 Configuration Options

Edit `config.py` or `.env` to customize:

### AI Provider Settings
```python
AI_PROVIDER = "openai"  # or "anthropic"
OPENAI_MODEL = "gpt-4-turbo-preview"
ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
```

### Data Settings
```python
MAX_FILE_SIZE_MB = 50
RISK_THRESHOLD_HIGH = 0.7
RISK_THRESHOLD_MEDIUM = 0.4
```

## 🎨 Using the Dashboard

### 1. Load Data
- Use the sidebar to upload your CSV/XLSX file
- Or select "Use Sample Data" to explore with synthetic data

### 2. Apply Filters
- Date range selection
- Amount range slider
- Transaction type (legitimate/fraud)
- Category and merchant filters
- Risk level filtering

### 3. Explore Analytics

**Overview Tab:**
- Key performance indicators
- Fraud trends over time
- Amount distributions
- Category and hourly patterns

**Analytics Tab:**
- Risk score distributions
- Confusion matrix with adjustable threshold
- Merchant analysis
- Detailed data table

**AI Copilot Tab:**
- Generate executive summaries
- Run what-if scenarios
- Get detection rule proposals
- Ask custom questions about your data

## 🤖 AI Copilot Features

### Executive Summary
Generates comprehensive insights including:
- Key findings and trends
- Critical patterns
- Actionable recommendations

### What-If Analysis
Explore scenarios like:
- "What if we implement a $500 transaction limit?"
- "What if we add velocity checks for high-risk merchants?"
- "What if fraud rates increase by 20%?"

### Detection Rules
AI proposes specific rules with:
- Rule conditions and thresholds
- Risk level classification
- Data-driven rationale
- False positive estimates

### Custom Queries
Ask questions like:
- "What are the main characteristics of fraudulent transactions?"
- "Which merchants have the highest fraud rates?"
- "What time of day sees the most fraud?"

## 🔒 Privacy & Security

### PII Protection
- Only **aggregated metrics** are sent to AI APIs
- No individual transaction details or user IDs
- No sensitive personal information in prompts

### API Key Security
- Store keys in `.env` file (never commit to git)
- `.gitignore` prevents accidental commits
- Environment-based configuration

### Data Handling
- All processing happens locally
- No data stored on external servers
- File uploads processed in memory

## 📦 Dependencies

- **streamlit** (1.29.0): Web application framework
- **pandas** (2.1.4): Data manipulation
- **numpy** (1.26.2): Numerical computing
- **plotly** (5.18.0): Interactive visualizations
- **python-dotenv** (1.0.0): Environment variable management
- **scikit-learn** (1.3.2): Machine learning utilities
- **openai** (1.6.1): OpenAI API client
- **anthropic** (0.8.1): Anthropic API client
- **openpyxl** (3.1.2): Excel file support

## 🧪 Testing with Sample Data

The included sample data generator creates realistic fraud scenarios:
- 10,000 transactions over 90 days
- ~2% fraud rate
- Fraudulent transactions have:
  - Higher amounts (2-5x normal)
  - Unusual hours (late night)
  - International/online locations
  - Higher rates in certain categories

## 🛠️ Troubleshooting

### "AI Copilot is not configured"
- Check that `.env` file exists
- Verify API key is correct
- Ensure `AI_PROVIDER` matches your key type

### "Data validation failed"
- Check that all required columns are present
- Verify `is_fraud` contains only 0/1 values
- Ensure `timestamp` is in valid datetime format
- Check that `amount` values are numeric and positive

### "No data matches the current filters"
- Reset filters in the sidebar
- Expand date range
- Select more categories/merchants

### Import Errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python version (3.10+ required)

## 🚀 Production Deployment

For production use, consider:

1. **Database Integration**: Replace CSV loading with database connections
2. **Authentication**: Add user authentication and role-based access
3. **Caching**: Implement Streamlit caching for large datasets
4. **Monitoring**: Add logging and error tracking
5. **Scaling**: Deploy on cloud platforms (AWS, GCP, Azure)
6. **API Rate Limits**: Implement rate limiting for AI API calls
7. **Data Validation**: Add more robust data quality checks

## 📝 License

This is a proof-of-concept project for demonstration purposes.

## 🤝 Contributing

This is a PoC project. For production use, consider:
- Adding unit tests
- Implementing CI/CD pipelines
- Adding more ML models for fraud detection
- Expanding AI capabilities
- Adding more visualization options

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the configuration settings
3. Verify your data format matches requirements

---

**Built with ❤️ using Python, Streamlit, and Generative AI**
