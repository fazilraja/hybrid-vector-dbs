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

def main():
    parser = argparse.ArgumentParser(description="Sample rows from a CSV file")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("output_file", type=str, help="Path for the output CSV file")
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows to sample (default: 1000)")
    parser.add_argument("--random", action="store_true", help="Take a random sample instead of first N rows")
    
    args = parser.parse_args()
    
    sample_csv(
        input_path=args.input_file,
        output_path=args.output_file,
        n_rows=args.rows,
        random_sample=args.random
    )

if __name__ == "__main__":
    main()
