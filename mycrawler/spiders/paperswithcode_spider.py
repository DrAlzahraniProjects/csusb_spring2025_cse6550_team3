import scrapy
import requests
from PyPDF2 import PdfReader
import io

class PaperSpider(scrapy.Spider):
    name = "paper_spider"
    start_urls = [
        "https://arxiv.org/pdf/2301.04871v2.pdf",
    ]

    def parse(self, response):
        # Fetch the PDF content directly into memory
        pdf_url = response.url
        pdf_response = requests.get(pdf_url)

        # Use BytesIO to handle the PDF content in memory
        pdf_file = io.BytesIO(pdf_response.content)

        # Extract text from the PDF and yield it
        yield from self.extract_text_from_pdf(pdf_file)

    def extract_text_from_pdf(self, pdf_file):
        # Read the PDF file from the BytesIO object
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:  # Check if text extraction was successful
                yield {"text": text}  # Yield each page's text as a dictionary