import openai
import os
import tiktoken
import time
from typing import List, Dict, Any
import pandas as pd
import asyncio
import chamois
import logging
from tabulate import tabulate
import textwrap
import nest_asyncio

nest_asyncio.apply()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("OPENAI_API_KEY is not set")
    exit(1)

openai.api_key = OPENAI_API_KEY
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_embeddings(
    df: pd.DataFrame,
    num_rows: int = 10, 
    max_tokens: int = 8191, # Max tokens for text-embedding-3-small
    encoding_name: str = "cl100k_base",
    price_per_token: float = 0.02 / 1000000 # Price for text-embedding-3-small per token
) -> List[Dict[str, Any]]:
    """
    Process a DataFrame, adding embeddings to a specified number of rows,
    ensuring the total number of tokens per request does not exceed max_tokens.
    Batches requests to OpenAI API.
    """
    result_list = []
    batch_case_text = []
    batch_ids = []
    current_tokens = 0
    total_tokens = 0
    total_batches = 0
    start_time = time.time()

    print(f"Processing {num_rows} rows...")

    for index, row in df.head(num_rows).iterrows():
        case_text = row['case_text']
        case_id = row['case_id']
        case_title = row['case_title'] # Get case_title

        # Skip row if case_text is not a string or is empty/whitespace
        if not isinstance(case_text, str) or not case_text.strip():
            print(f"Skipping row {index} with invalid case_text: {case_text}")
            continue

        tokens = num_tokens_from_string(case_text, encoding_name)

        # Check if adding this text would exceed the token limit for the current batch
        if current_tokens + tokens > max_tokens:
            # Process the current batch before adding the new text
            if batch_case_text:
                print(f"Processing batch of {len(batch_case_text)} items, tokens: {current_tokens}")
                try:
                    embeddings = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=batch_case_text,
                        dimensions=512
                    ).data[0].embedding
                    print(f"Embeddings: {embeddings}")
                    for i, text in enumerate(batch_case_text):
                        print(f"Batch text: {text}")
                        # Find original title for the corresponding case_id in the batch
                        original_title = df.loc[df['case_id'] == batch_ids[i], 'case_title'].iloc[0]
                        result_list.append({
                            'case_id': batch_ids[i],
                            'case_title': original_title, # Add case_title here
                            'case_text': text,
                            'embeddings': embeddings
                        })
                    total_batches += 1
                except Exception as e:
                    print(f"Error processing batch: {e}")

                # Reset batch
                batch_case_text = []
                batch_ids = []
                current_tokens = 0

        # Add the current text to the batch if it fits (or it's the start of a new batch)
        if tokens <= max_tokens: # Ensure single item is not too large
             batch_case_text.append(case_text)
             batch_ids.append(case_id)
             current_tokens += tokens
             total_tokens += tokens
        else:
            print(f"Skipping row {index} as it exceeds max_tokens: {tokens} tokens")


    # Process the final batch if it's not empty
    if batch_case_text:
        print(f"Processing final batch of {len(batch_case_text)} items, tokens: {current_tokens}")
        try:
            embeddings = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch_case_text,
                dimensions=512
            ).data[0].embedding
            for i, text in enumerate(batch_case_text):
                 # Find original title for the corresponding case_id in the batch
                 original_title = df.loc[df['case_id'] == batch_ids[i], 'case_title'].iloc[0]
                 result_list.append({
                    'case_id': batch_ids[i],
                    'case_title': original_title, # Add case_title here
                    'case_text': text,
                    'embeddings': embeddings
                 })
            total_batches += 1
        except Exception as e:
            print(f"Error processing final batch: {e}")


    end_time = time.time()
    duration = end_time - start_time
    money_burned = total_tokens * price_per_token

    # Print the statistics
    print(f"Completed in {duration:.2f} seconds.")
    print(f"Total number of tokens: {total_tokens:,}")
    print(f"Total number of batches: {total_batches:,}")
    print(f"Money burned: ${money_burned:.6f}")
    return result_list


df = pd.read_csv('../data/legal_text_first_1000.csv')

def wrap_text(text, wide=60):
    if isinstance(text, str):
        return "\n".join(textwrap.wrap(text, wide))
    else:
        return text
    
df_display = df.head(3).copy()

for col in df_display.columns:
    df_display[col] = df_display[col].apply(lambda x: wrap_text(x, 60))
    
print(tabulate(df_display, headers='keys', tablefmt='grid', showindex=False))

all_cases_embeddings = get_embeddings(df)


# Save embeddings to a csv file
df_embeddings = pd.DataFrame(all_cases_embeddings)

for col in df_embeddings.columns:
    df_embeddings[col] = df_embeddings[col].apply(lambda x: wrap_text(x, 60))
    
print(tabulate(df_embeddings.head(3), headers='keys', tablefmt='grid', showindex=False))

df_embeddings.to_csv('../data/all_cases_embeddings.csv', index=False)

print(f"Embeddings saved to all_cases_embeddings.csv")

