# Deployment Guide: Slide Critique Tool (Publisher Model)

This guide will help you push your local application, including the vector database, to Streamlit Community Cloud.

## 1. Prerequisites (Local Terminal)

You have large database files in `slide_memory_db/`. You must use **Git LFS** (Large File Storage) to push them successfully.

### Install Git LFS
If you haven't installed it yet, run:
```bash
git lfs install
```

### Track Large Folders
We need to tell Git to treat your database and image files as large objects. Run these commands **inside your project folder**:

```bash
git lfs track "slide_memory_db/**/*"
git lfs track "slide_images/**/*"
```
*(This creates or updates a `.gitattributes` file. You must commit this file too.)*

## 2. Commit and Push

Run the following commands to stage your changes and push to GitHub:

```bash
# 1. Add all modified files (including the new .gitattributes from LFS)
git add .

# 2. Check status (Verify secrets.json is NOT listed)
git status
# You should NOT see 'secrets.json'. 
# You SHOULD see '.gitattributes', 'requirements.txt', '.gitignore' etc.

# 3. Commit
git commit -m "Prepare for deployment: Update requirements and add DB LFS tracking"

# 4. Push to GitHub
git push origin main
```
*(Note: The first push might take a while as it uploads the database files.)*

## 3. Streamlit Cloud Configuration

1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Deploy your app from your GitHub repository.
3.  **Secrets Management**:
    *   Go to your App's **Settings** -> **Secrets**.
    *   Copy the content of your local `secrets.json` and paste it there.
    *   It should look like this:
        ```toml
        GOOGLE_API_KEY = "your-key-here"
        cloudconvert_api_key = "your-key-here"
        ```
4.  **Reboot**:
    *   If the app fails to start initially, try rebooting it once the secrets are saved.
