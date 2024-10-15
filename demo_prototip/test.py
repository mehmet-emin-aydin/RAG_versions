import os
from dotenv import load_dotenv
import sys
import requests
from atlassian import Confluence 
import requests
import json
import time
from pprint import pprint

from backend.updater import update_given_spaces
load_dotenv()

# Confluence oturum bilgileri
confluence_username = os.getenv("CONFLUENCE_USERNAME")
confluence_password = os.getenv("CONFLUENCE_PASSWORD")
confluence_token = os.getenv("CONFLUENCE_TOKEN")

# confluence objesini oluşturma
confluence = Confluence( 
    url="https://wiki.softtech.com.tr",
    token=confluence_token
)


spaces =confluence.get_all_spaces(start=0,limit=500,expand=None)
space_list = [space_item['key'] for space_item in spaces['results']]

update_given_spaces(space_list)