import scrapy
import os
import json
from PyPDF2 import PdfReader
from io import BytesIO

class PaperSpider(scrapy.Spider):
    name = "paper_spider"
    allowed_domains = ['paperswithcode.com', 'arxiv.org']
    start_urls = ['https://paperswithcode.com/task/breast-cancer-detection']
    processed_files_path = '/data/processed_files.json'
    processed_files = set()

    def start_requests(self):
        print("Start requests --------------------")
        # Load the persisted set of processed URLs when the spider starts
        if os.path.exists(self.processed_files_path):
            try:
                with open(self.processed_files_path, 'r') as f:
                    self.processed_files = set(json.load(f))
                print("Loaded processed files from disk.")
            except Exception as e:
                print(f"Error loading processed files: {e}")
                self.processed_files = set()
        else:
            print("os path doesn't exist")
        print('Paper Spider is starting...')
        for url in self.start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def close(self, reason):
        # Save the updated set to disk when the spider closes
        try:
            with open(self.processed_files_path, 'w') as f:
                json.dump(list(self.processed_files), f)
            print("Saved processed files to disk.")
        except Exception as e:
            print(f"Error saving processed files: {e}")

    def parse(self, response):
        # Select the div with id 'task-papers-list'
        papers_div = response.css('div#task-papers-list')
        # Loop over each paper entry (adjust the selector as needed)
        for paper in papers_div.css('a[href^="/paper/"]'):
            paper_link = response.urljoin(paper.attrib['href'])
            yield scrapy.Request(paper_link, callback=self.parse_paper)

    def parse_paper(self, response):
        # Get the relative PDF URL from the page
        pdf_url_relative = response.css('a[href$=".pdf"]::attr(href)').get()
        if not pdf_url_relative:
            return

        # Convert to an absolute URL for consistency
        pdf_url = response.urljoin(pdf_url_relative)

        # Check if this URL is already processed
        if pdf_url in self.processed_files:
            print(f"Skipping {pdf_url} - already processed.")
            return
        else:
            print(f"Not Skipping {pdf_url}")

        # Extract the filename in a robust way
        pdf_filename = pdf_url.split('/')[-1]
        if pdf_filename in ["2004.03500v2.pdf", "1708.09427v5.pdf"]:
            # Mark the URL as processed now to avoid duplicate scheduling
            self.processed_files.add(pdf_url)
            yield scrapy.Request(pdf_url, callback=self.parse_pdf)
        else:
            print(f"PDF {pdf_filename} did not match the criteria.")

    def parse_pdf(self, response):
        # Read the PDF file
        pdf_file = BytesIO(response.body)
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                yield {"text": text}
