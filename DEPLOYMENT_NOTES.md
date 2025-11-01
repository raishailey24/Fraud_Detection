# FraudSight Deployment Notes

## Repository Size Optimization

### Issue
- Original parquet files: 374MB total (15 chunks × ~25MB each)
- Streamlit Cloud limit: ~200MB repository size
- Solution: Remove parquet files from git, host externally

### Files Removed from Git
- `data/*.parquet` (all parquet chunks)
- `data/github_chunks/` (duplicate chunks)
- `data/converted/` (converted chunks)

### Alternative Hosting Solutions

#### Option 1: GitHub Releases (Recommended)
1. Create a GitHub Release with parquet files as assets
2. Update app to download from release URLs
3. URLs format: `https://github.com/raishailey24/Fraud_Detection/releases/download/v1.0/chunk_XX.parquet`

#### Option 2: External File Hosting
- Google Drive with public links
- Dropbox with direct download links
- AWS S3 with public access

### Current App Behavior
- App will show download interface
- Downloads chunks dynamically when needed
- Stores locally in user's session
- No large files in repository

### Repository Size After Cleanup
- Before: ~1.47GB (with git history)
- After: ~50MB (code only)
- Deployment: ✅ Under Streamlit limits

### Next Steps
1. ✅ Remove parquet files from git
2. 🔄 Create GitHub Release with parquet files
3. 🔄 Update app download URLs
4. 🔄 Test full deployment
