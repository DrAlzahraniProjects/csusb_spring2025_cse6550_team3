import scrapy
import io
from PyPDF2 import PdfReader

class PaperSpider(scrapy.Spider):
    name = 'paper_spider'
    start_urls = ['https://paperswithcode.com/area/medical/cancer']  # Replace with the URL of the website you want to scrape
    max_depth = 3  # Set the maximum depth to follow links

    def parse(self, response, depth=0):
        # Extract PDF links
        pdf_links = response.css('a[href$=".pdf"]::attr(href)').getall()
        for pdf_link in pdf_links:
            # Create absolute URL if the link is relative
            if not pdf_link.startswith('http'):
                pdf_link = response.urljoin(pdf_link)
            yield scrapy.Request(pdf_link, callback=self.extract_text_from_pdf)

        # Follow links if the current depth is less than max_depth
        if depth < self.max_depth:
            next_links = response.css('a::attr(href)').getall()
            for next_link in next_links:
                if not next_link.startswith('http'):
                    next_link = response.urljoin(next_link)
                yield scrapy.Request(next_link, callback=self.parse, cb_kwargs={'depth': depth + 1})

    def extract_text_from_pdf(self, response):
        # Use BytesIO to handle the PDF content in memory
        pdf_file = io.BytesIO(response.body)

        # Extract text from the PDF and yield it
        yield from self.extract_text(pdf_file)

    def extract_text(self, pdf_file):
        # Read the PDF file from the BytesIO object
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:  # Check if text extraction was successful
                yield {"text": text}  # Yield each page's text as a dictionary
