#!/usr/bin/env python3
"""
Script to create a smaller CSV file by sampling rows from a larger CSV file.
"""

import pandas as pd
from pathlib import Path
import logging
import argparse
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def sample_csv(
    input_path: Path | str,
    output_path: Path | str,
    n_rows: int = 1000,
    random_sample: bool = False
) -> None:
    """
    Create a smaller CSV file by taking the first N rows from a larger CSV.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path where to save the output CSV file
        n_rows: Number of rows to include (default: 1000)
        random_sample: If True, take a random sample instead of first N rows
    """
    try:
        logger.info(f"Reading CSV file from {input_path}")
        df = pd.read_csv(input_path)
        
        logger.info(f"Original file has {len(df)} rows")
        if random_sample:
            sample_df = df.sample(n=n_rows)
            logger.info(f"Taking random sample of {n_rows} rows")
        else:
            sample_df = df.head(n_rows)
            logger.info(f"Taking first {n_rows} rows")
            
        logger.info(f"Saving sampled CSV to {output_path}")
        sample_df.to_csv(output_path, index=False)
        logger.info("Done!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

def sample_csv_from_offset(
    input_path: Path | str,
    output_path: Path | str,
    start_row: int = 1000,
    n_rows: int = 500
) -> None:
    """
    Create a smaller CSV file by taking N rows starting from a specific row in a larger CSV.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path where to save the output CSV file
        start_row: Row number to start sampling from (default: 1000)
        n_rows: Number of rows to include (default: 500)
    """
    try:
        logger.info(f"Reading CSV file from {input_path}")
        df = pd.read_csv(input_path)
        
        # Drop rows with any missing values
        df = df.dropna()
        logger.info(f"File has {len(df)} rows after dropping missing values")
        
        if len(df) < start_row + n_rows:
            logger.error("Not enough rows in the file to sample from the specified start row.")
            return
        
        sample_df = df.iloc[start_row:start_row + n_rows]
        logger.info(f"Taking {n_rows} rows starting from row {start_row}")
        
        logger.info(f"Saving sampled CSV to {output_path}")
        sample_df.to_csv(output_path, index=False)
        logger.info("Done!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Sample rows from a CSV file")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("output_file", type=str, help="Path for the output CSV file")
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows to sample (default: 1000)")
    parser.add_argument("--random", action="store_true", help="Take a random sample instead of first N rows")
    parser.add_argument("--offset", action="store_true", help="Sample 500 rows starting from the 1001st row")
    
    args = parser.parse_args()
    
    if args.offset:
        sample_csv_from_offset(
            input_path=args.input_file,
            output_path=args.output_file,
            start_row=1000,
            n_rows=500
        )
    else:
        sample_csv(
            input_path=args.input_file,
            output_path=args.output_file,
            n_rows=args.rows,
            random_sample=args.random
        )

if __name__ == "__main__":
    main()
