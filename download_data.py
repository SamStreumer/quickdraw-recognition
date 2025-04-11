"""
Download script for Quick Draw dataset.
"""

import os
import sys
import argparse
import requests
from tqdm import tqdm

# Default categories to download (expanded list for better model training)
DEFAULT_CATEGORIES = [
    'apple', 'banana', 'car', 'cat', 'dog', 'fish', 
    'house', 'tree', 'bicycle', 'airplane', 'book',
    'clock', 'flower', 'guitar', 'hat', 'moon',
    'pizza', 'star', 'sun', 'umbrella'
]

# Base URL for the dataset
BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

def download_file(url, destination):
    """
    Download a file with progress indication.
    
    Args:
        url (str): URL to download
        destination (str): Local file path to save to
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        print(f"Downloading {os.path.basename(destination)}...")
        
        with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                file.write(data)
                progress_bar.update(len(data))
        
        return True
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # Remove partially downloaded file
        if os.path.exists(destination):
            os.remove(destination)
        return False

def main():
    """
    Main function for downloading the Quick Draw dataset.
    """
    parser = argparse.ArgumentParser(description='Download Quick Draw dataset')
    parser.add_argument('--categories', type=str, nargs='+',
                       help=f'Categories to download (default: {", ".join(DEFAULT_CATEGORIES)})')
    parser.add_argument('--all', action='store_true',
                       help='Download all 345 categories (warning: large download)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Directory to save the downloaded files (default: data)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which categories to download
    if args.all:
        print("Warning: Downloading all 345 categories will require significant disk space.")
        print("The total dataset is over 40GB. Are you sure you want to continue?")
        response = input("Continue? (y/n): ").strip().lower()
        
        if response != 'y':
            print("Download cancelled.")
            return
        
        # Download list of all categories
        print("Fetching list of all categories...")
        try:
            response = requests.get('https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt')
            response.raise_for_status()
            categories = [line.strip() for line in response.text.splitlines() if line.strip()]
        except Exception as e:
            print(f"Error fetching category list: {e}")
            print("Falling back to default categories.")
            categories = DEFAULT_CATEGORIES
    else:
        categories = args.categories if args.categories else DEFAULT_CATEGORIES
    
    print(f"Preparing to download {len(categories)} categories to {args.output_dir}...")
    
    # Download each category
    successful = 0
    failed = 0
    
    for category in categories:
        url = f"{BASE_URL}{category}.npy"
        destination = os.path.join(args.output_dir, f"{category}.npy")
        
        if os.path.exists(destination):
            file_size = os.path.getsize(destination) / (1024 * 1024)  # size in MB
            print(f"File already exists: {destination} ({file_size:.2f} MB)")
            successful += 1
            continue
        
        if download_file(url, destination):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\nDownload Summary:")
    print(f"  Successfully downloaded/found: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total categories: {len(categories)}")
    
    if successful > 0:
        print("\nYou can now train the model with:")
        print(f"  python train_model.py --categories {' '.join(categories[:min(5, successful)])} ...")
        print("\nOr run the application with:")
        print(f"  python main.py")

if __name__ == "__main__":
    main()