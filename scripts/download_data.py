import os
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

FILENAME = "diviner_learn_data_c7_processed_Xf_7_2000.npy"

# Get credentials from environment
USERNAME = os.getenv("HUGGINGFACE_USERNAME")
TOKEN = os.getenv("HUGGINGFACE_TOKEN")
DATASET_REPO = f"{USERNAME}/{os.getenv('DATASET_REPO')}"
FILENAME = os.getenv("FILENAME")

if not TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables.")

# Ensure the 'data' directory exists in the project root
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Download the file from the HuggingFace Hub into the 'data' directory
downloaded_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename=FILENAME,
    use_auth_token=TOKEN,
    local_dir=DATA_DIR
)

print(f"Downloaded file to: {downloaded_path}")