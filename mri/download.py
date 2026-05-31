from huggingface_hub import hf_hub_download
import os

# Specify the repository ID and the filename you want to download
repo_id = "yasiz/ambient_mri_data"
filename = "mri_data_numpy.zip"  # Replace with the actual filename

# Specify the local directory where you want to save the file
local_dir = "/blob/v-tianyuchen/Projects/Ambient_A/mri/"  # Replace with your desired path

# Ensure the local directory exists
os.makedirs(local_dir, exist_ok=True)

# Download the file
file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=local_dir)

print(f"File downloaded to: {file_path}")
