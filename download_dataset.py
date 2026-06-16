"""Download football dataset from Kaggle."""

import urllib.request
import zipfile
import os
import pandas as pd

def download_kaggle_dataset():
    """Download the Kaggle football dataset."""
    
    url = 'https://www.kaggle.com/api/v1/datasets/download/martj42/international-football-results-from-1872-to-2017'
    zip_path = 'football_data.zip'
    csv_path = 'football_data.csv'
    
    print("Downloading Kaggle football dataset...")
    print("URL:", url)
    
    try:
        # Create a request with headers to mimic browser
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
        
        print(f"Downloaded {zip_path}")
        
        # Extract the CSV
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find the results CSV file specifically
            file_list = zip_ref.namelist()
            print(f"Files in archive: {file_list}")
            
            # Look for results.csv or the largest CSV
            csv_files = [f for f in file_list if f.endswith('.csv')]
            if 'results.csv' in csv_files:
                csv_file = 'results.csv'
            else:
                # Get the largest CSV (should be the main results file)
                csv_file = max(csv_files, key=lambda f: zip_ref.getinfo(f).file_size)
            
            print(f"Extracting {csv_file}...")
            zip_ref.extract(csv_file)
            
            # Rename to standard name
            if csv_file != csv_path:
                os.rename(csv_file, csv_path)
        
        # Verify the data
        df = pd.read_csv(csv_path)
        print(f"\nSuccess! Loaded {len(df)} matches")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df.iloc[:, 0].min()} to {df.iloc[:, 0].max()}")
        
        # Clean up zip
        os.remove(zip_path)
        print(f"\nDataset saved as: {csv_path}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017")
        print("2. Download the CSV file")
        print("3. Save as: football_data.csv in this directory")
        return False

if __name__ == "__main__":
    download_kaggle_dataset()
