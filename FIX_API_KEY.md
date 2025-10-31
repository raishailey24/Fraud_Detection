# 🔑 How to Fix API Key Error

## The Error You're Seeing:
```
OpenAI API error: Error code: 401 - Invalid API key
```

This means the API key in your `.env` file is not valid.

---

## ✅ Solution Steps:

### Step 1: Open Your `.env` File

Navigate to:
```
c:\Users\malvi\OneDrive\Desktop\PythonAutomationProjects\fraud_genai_bi\.env
```

Open it with Notepad or any text editor.

---

### Step 2: Get a Valid API Key

#### For OpenAI:
1. Go to: https://platform.openai.com/api-keys
2. Sign in (or create account)
3. Click "Create new secret key"
4. Name it: "Fraud Dashboard"
5. Copy the key (starts with `sk-proj-...`)

#### For Anthropic (Claude):
1. Go to: https://console.anthropic.com/
2. Sign in (or create account)
3. Go to API Keys section
4. Create new key
5. Copy the key (starts with `sk-ant-...`)

---

### Step 3: Update `.env` File

**For OpenAI:**
```env
# OpenAI Configuration
OPENAI_API_KEY=sk-proj-PASTE-YOUR-ACTUAL-KEY-HERE
OPENAI_MODEL=gpt-4-turbo-preview

# Anthropic Configuration (leave empty)
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# AI Provider Selection
AI_PROVIDER=openai
```

**For Anthropic:**
```env
# OpenAI Configuration (leave empty)
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4-turbo-preview

# Anthropic Configuration
ANTHROPIC_API_KEY=sk-ant-PASTE-YOUR-ACTUAL-KEY-HERE
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# AI Provider Selection
AI_PROVIDER=anthropic
```

---

### Step 4: Save and Restart

1. **Save** the `.env` file
2. **Refresh** your browser (F5)
3. The dashboard will reload with the new API key

---

## 🎯 Testing the Fix

After updating the API key:

1. Go to the **AI Copilot** tab
2. Click **Executive Summary** sub-tab
3. Click **Generate Summary** button
4. You should see AI-generated insights (takes 5-10 seconds)

If it works, you'll see something like:
```
Based on analysis of 10,000 transactions with 2.8% fraud rate:

Key Findings:
• Fraudulent transactions average 3.2x higher amounts
• Peak fraud occurs between 10pm-6am
• International transactions show 5.8x higher fraud rate

Recommendations:
• Implement enhanced verification for high-value night transactions
• Add velocity checks for international merchants
```

---

## 🆓 Free API Options

### OpenAI Free Tier:
- New accounts get $5 free credits
- Enough for ~100-200 AI requests
- Perfect for this demo

### Anthropic Free Tier:
- New accounts get free credits
- Claude 3 Sonnet available

### Alternative: Use Without AI
The dashboard works perfectly without AI keys!
- All BI features work
- Only AI Copilot tab is disabled

---

## 🔒 Security Tips

1. **Never commit `.env` to git** (already in .gitignore)
2. **Don't share your API key** publicly
3. **Rotate keys** after demos/presentations
4. **Set usage limits** in API dashboard

---

## 🆘 Still Having Issues?

### Error: "API key not found"
- Make sure `.env` file exists in project root
- Check there are no extra spaces around the key
- Ensure key starts with `sk-proj-` or `sk-ant-`

### Error: "Rate limit exceeded"
- You've hit API usage limits
- Wait a few minutes or upgrade plan
- Or switch to other provider

### Error: "Module not found"
- Run: `pip install openai anthropic`
- Restart the dashboard

---

## 📞 Quick Help Commands

**Check if .env exists:**
```powershell
dir .env
```

**View .env content (careful - contains secrets!):**
```powershell
type .env
```

**Restart dashboard:**
```powershell
# Stop current server (Ctrl+C in terminal)
streamlit run app.py
```

---

## ✨ Pro Tip

For the competition demo, you can:
1. Use free API credits
2. Pre-generate AI responses and screenshot them
3. Or demo without AI features (BI dashboard is still impressive!)

The judges will understand if you don't have paid API access.
