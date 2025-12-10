import cloudconvert
import os
import time
import json

def load_api_key():
    """Attempts to load the CloudConvert API key from secrets.json"""
    try:
        secrets_path = os.path.join(os.path.dirname(__file__), "secrets.json")
        if os.path.exists(secrets_path):
            with open(secrets_path, "r") as f:
                secrets = json.load(f)
                return secrets.get("cloudconvert_api_key")
    except Exception as e:
        print(f"Error reading secrets.json: {e}")
    return None

def convert_pptx_to_pdf(input_path, output_path):
    """
    Converts a PPTX file to PDF using CloudConvert API.
    Returns True if successful, False otherwise.
    """
    api_key = load_api_key()
    
    if not api_key:
        print("Error: CloudConvert API key not found in secrets.json")
        return False
        
    # Configure usage
    cloudconvert.configure(api_key=api_key)
    
    filename = os.path.basename(input_path)
    job_tag = str(time.time()) # Unique tag for this job
    
    try:
        print(f"Starting conversion for {filename}...")
        
        # 1. Create a Job
        job = cloudconvert.Job.create(payload={
            "tag": job_tag,
            "tasks": {
                "import-my-file": {
                    "operation": "import/upload"
                },
                "convert-my-file": {
                    "operation": "convert",
                    "input": "import-my-file",
                    "output_format": "pdf",
                    "engine": "office" 
                },
                "export-my-file": {
                    "operation": "export/url",
                    "input": "convert-my-file"
                }
            }
        })
        
        # 2. Upload the file
        upload_task_id = job['tasks'][0]['id']
        upload_task = cloudconvert.Task.find(id=upload_task_id)
        cloudconvert.Task.upload(file_name=input_path, task=upload_task)
        
        # 3. Wait for completion
        print("Waiting for cloud conversion...")
        job = cloudconvert.Job.wait(id=job['id']) # Wait for job to finish
        
        # 4. Download result
        export_task = None
        for task in job['tasks']:
            if task['name'] == 'export-my-file' and task['status'] == 'finished':
                export_task = task
                break
                
        if export_task and export_task.get('result') and export_task['result'].get('files'):
            file_data = export_task['result']['files'][0]
            download_url = file_data['url']
            
            # Use cloudconvert's download helper or requests if needed, 
            # but the SDK usually expects us to handle the download URL or uses a method.
            # Actually, the SDK wraps it. Let's use the file download method if available
            # or simply use the export task 'download' method.
            
            # The SDK documentation suggests:
            cloudconvert.Task.wait(id=export_task['id']) # Double check it's done
            file_name = file_data['filename']
            
            # We need to download this file.
            # CloudConvert Python SDK v2+ doesn't have a direct 'download' one-liner on the task object 
            # except via the 'files' url.
            # We'll use requests to download it since we have the URL.
            
            import requests # Import here to avoid top-level dependency if possible, but it's standard
            
            r = requests.get(download_url)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(r.content)
                print(f"Successfully downloaded converted PDF to {output_path}")
                return True
            else:
                print(f"Failed to download result: {r.status_code}")
                return False
                
        else:
            print("Conversion job failed or no export result found.")
            return False

    except Exception as e:
        print(f"CloudConvert Error: {e}")
        return False
