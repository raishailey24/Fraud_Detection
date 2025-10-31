# ⚡ Quick Start Guide

Get the Fraud Analytics Dashboard running in 5 minutes!

## Step 1: Install Dependencies (1 min)

```bash
# Navigate to project directory
cd fraud_genai_bi

# Install required packages
pip install -r requirements.txt
```

## Step 2: Configure API Keys (1 min)

```bash
# Copy the example environment file
copy .env.example .env  # Windows
# OR
cp .env.example .env    # macOS/Linux
```

Edit `.env` and add your API key:

**For OpenAI:**
```env
OPENAI_API_KEY=sk-your-openai-key-here
AI_PROVIDER=openai
```

**For Anthropic (Claude):**
```env
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
AI_PROVIDER=anthropic
```

> **Note:** You can skip this step to explore the dashboard without AI features.

## Step 3: Generate Sample Data (30 seconds)

```bash
python generate_sample_data.py
```

This creates realistic fraud transaction data in `data/sample_transactions.csv`.

## Step 4: Launch Dashboard (30 seconds)

```bash
streamlit run app.py
```

The dashboard opens automatically at `http://localhost:8501` 🎉

## Step 5: Explore Features (2 min)

### In the Sidebar:
1. Select **"Use Sample Data"**
2. Adjust filters (date range, amount, categories)

### In the Dashboard:
1. **Overview Tab**: View KPIs and visualizations
2. **Analytics Tab**: Explore risk scores and confusion matrix
3. **AI Copilot Tab**: Generate insights (if API key configured)

## 🎯 Try These Actions

### Generate Executive Summary
1. Go to **AI Copilot** tab
2. Click **"Executive Summary"** sub-tab
3. Click **"Generate Summary"** button
4. Wait 5-10 seconds for AI analysis

### Run What-If Analysis
1. Go to **"What-If Analysis"** sub-tab
2. Enter scenario: *"What if we implement a $500 transaction limit?"*
3. Click **"Analyze Scenario"**

### Get Detection Rules
1. Go to **"Detection Rules"** sub-tab
2. Click **"Generate Detection Rules"**
3. Review AI-proposed fraud detection rules

### Ask Custom Questions
1. Go to **"Custom Query"** sub-tab
2. Ask: *"What are the main characteristics of fraudulent transactions?"*
3. Click **"Ask AI"**

## 📊 Using Your Own Data

1. Prepare CSV/XLSX with required columns:
   - `transaction_id`, `timestamp`, `amount`, `merchant`, `category`, `is_fraud`

2. In the sidebar, select **"Upload File"**

3. Upload your file and explore!

## 🔧 Troubleshooting

**"AI Copilot is not configured"**
- Add your API key to `.env` file
- Restart the application

**"Module not found" errors**
- Run: `pip install -r requirements.txt`

**"Data validation failed"**
- Check your CSV has all required columns
- Ensure `is_fraud` contains only 0 or 1 values

## 🎓 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize `config.py` for your needs
- Explore different AI models and providers
- Try different detection thresholds in Analytics tab

---

**Need Help?** Check the [README.md](README.md) troubleshooting section.
