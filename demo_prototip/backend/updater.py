import time
import json
import requests
from dotenv import load_dotenv

#local objects
from backend.client import ConfluenceClient
from backend.space import Space


def update_given_spaces(spaces_keys : list = None):
    confluence_client = ConfluenceClient()
    print(spaces_keys)
    results = []
    total_tokens = 0
    total_chunks = 0
    total_pages = 0
    for space_name in spaces_keys:
        space_instance = Space(confluence_client, space_name)
        result = space_instance.update_space()
        results.append(result)
        # total_tokens += result['tokens']
        # total_chunks += result['chunks']
        total_pages += result['page_size']
        print(f"\n{total_pages} up to {space_name}\n")
        with open('space_processing_results.json', 'a+') as result_file:
            json.dump(result, result_file, indent=4)

    with open('space_processing_results_all.json', 'w') as result_file:
        json.dump(results, result_file, indent=4)




# def main():
#     confluence_client = ConfluenceClient()
#     spaces_keys = ['YPZ', 'MYM', 'SPISS', 'YNYGAB']



# if __name__ == "__main__":
#     main()