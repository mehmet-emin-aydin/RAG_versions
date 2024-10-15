
import json
import html2text

from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
# from attachment import Attachment
class Page:
    def __init__(self, confluence_client, space, page_id):
        self.confluence = confluence_client
        self.space = space
        self.page_id = page_id

    def get_version_info(self):
        page_details = self.confluence.get_page_by_id(self.page_id)['version']['when']
        return {'id': self.page_id, 'when': page_details}

    # def update_attachments(self):
    #     attachment_handler = Attachment(self.confluence, self.space, self.page_id)
    #     return attachment_handler.process_attachments()
    
    def get_page_content(self):
        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True
        text_maker.ignore_images = True
        page_content = self.confluence.get_page_by_id(page_id=self.page_id,expand="body.export_view.value")
        text = text_maker.handle(page_content["body"]["export_view"]["value"])
        self.title = page_content['title']
        self.source = 'confluence'
        return text


    def get_space_content(self):
        pass