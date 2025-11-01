"""
Helper script to update Google Drive file IDs in the app configuration.
Run this after uploading your parquet chunks to Google Drive.
"""

def update_google_drive_file_ids():
    """
    Interactive script to update Google Drive file IDs.
    """
    print("Google Drive File ID Updater")
    print("=" * 40)
    print("\nAfter uploading your 4 parquet chunks to Google Drive:")
    print("1. Set sharing to 'Anyone with the link can view'")
    print("2. Copy the file ID from each Google Drive URL")
    print("3. Enter the file IDs below")
    print("\nFile ID is the part after '/d/' in the Google Drive URL:")
    print("Example: https://drive.google.com/file/d/1ABC123xyz/view")
    print("File ID: 1ABC123xyz")
    print()
    
    # Collect file IDs
    file_ids = {}
    chunks = [
        ("chunk_00", "Chunk 1/4"),
        ("chunk_01", "Chunk 2/4"), 
        ("chunk_02", "Chunk 3/4"),
        ("chunk_03", "Chunk 4/4")
    ]
    
    for chunk_key, chunk_desc in chunks:
        while True:
            file_id = input(f"Enter Google Drive file ID for {chunk_desc}: ").strip()
            if file_id and len(file_id) > 10:
                file_ids[chunk_key] = file_id
                break
            else:
                print("Please enter a valid file ID (should be 20+ characters)")
    
    # Read current app.py
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace file IDs
    replacements = {
        'YOUR_CHUNK_0_FILE_ID_HERE': file_ids['chunk_00'],
        'YOUR_CHUNK_1_FILE_ID_HERE': file_ids['chunk_01'],
        'YOUR_CHUNK_2_FILE_ID_HERE': file_ids['chunk_02'],
        'YOUR_CHUNK_3_FILE_ID_HERE': file_ids['chunk_03']
    }
    
    for placeholder, file_id in replacements.items():
        content = content.replace(placeholder, file_id)
    
    # Write updated app.py
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ File IDs updated successfully!")
    print("\nUpdated file IDs:")
    for chunk_key, file_id in file_ids.items():
        print(f"  {chunk_key}: {file_id}")
    
    print("\n🚀 Next steps:")
    print("1. Test the download in your Streamlit app")
    print("2. Commit and push the changes to deploy")
    print("3. Your app will now download from Google Drive!")

if __name__ == "__main__":
    update_google_drive_file_ids()
