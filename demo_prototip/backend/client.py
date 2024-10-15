import os
from atlassian import Confluence
from dotenv import load_dotenv
load_dotenv()

class ConfluenceClient:
    def __init__(self):
        self.token= os.getenv("CONFLUENCE_TOKEN")
        self.confluence = Confluence(
            url="https://wiki.softtech.com.tr",
            token= self.token
        )
