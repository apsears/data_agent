#!/usr/bin/env python3
"""Download dataset from Google Drive to data/ directory."""

import os
import sys
import urllib.request
import urllib.parse
from pathlib import Path


def download_from_google_drive(file_id: str, output_path: str) -> bool:
    """
    Download a file from Google Drive using its file ID.

    Args:
        file_id: Google Drive file ID
        output_path: Local path where file should be saved

    Returns:
        True if download successful, False otherwise
    """
    # Google Drive direct download URL
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"

    try:
        print(f"Downloading dataset from Google Drive...")
        print(f"File ID: {file_id}")
        print(f"Output: {output_path}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Download the file
        urllib.request.urlretrieve(download_url, output_path)

        # Verify file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            file_size = os.path.getsize(output_path)
            print(f"✅ Download successful! File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            return True
        else:
            print(f"❌ Download failed - file is empty or doesn't exist")
            return False

    except Exception as e:
        print(f"❌ Download failed with error: {e}")
        return False


def main():
    """Main entry point for dataset download."""
    # Configuration
    GOOGLE_DRIVE_FILE_ID = "109vhmnSLN3oofjFdyb58l_rUZRa0d6C8"
    OUTPUT_PATH = "data/pipeline_data_1.parquet"

    # Get absolute paths
    script_dir = Path(__file__).parent.parent.parent  # Go up from tools/utils/ to project root
    output_file = script_dir / OUTPUT_PATH

    print("=" * 60)
    print("🔄 Claude Data Agent - Dataset Download")
    print("=" * 60)

    # Check if file already exists
    if output_file.exists():
        file_size = output_file.stat().st_size
        print(f"📁 Dataset already exists: {output_file}")
        print(f"   File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")

        # Ask user if they want to redownload
        response = input("\nRedownload anyway? [y/N]: ").strip().lower()
        if response != 'y' and response != 'yes':
            print("✅ Using existing dataset file")
            return 0
        else:
            print("🔄 Redownloading dataset...")

    # Download the dataset
    success = download_from_google_drive(GOOGLE_DRIVE_FILE_ID, str(output_file))

    if success:
        print("\n✅ Dataset download completed successfully!")
        print(f"📍 Location: {output_file}")
        return 0
    else:
        print("\n❌ Dataset download failed!")
        print("🔧 Troubleshooting tips:")
        print("   - Check your internet connection")
        print("   - Verify the Google Drive file is public")
        print("   - Try downloading manually and placing in data/")
        return 1


if __name__ == "__main__":
    sys.exit(main())