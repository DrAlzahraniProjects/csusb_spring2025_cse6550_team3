import scrapy
from PyPDF2 import PdfReader
from io import BytesIO

class PaperSpider(scrapy.Spider):
    name = "paper_spider"
    start_urls = ['https://paperswithcode.com/task/breast-cancer-detection']  # Replace with the URL you want to scrape

    def parse(self, response):
        # Select the div with id 'task-papers-list'
        papers_div = response.css('div#task-papers-list')

        # Extract all href links that start with '/paper/'
        for paper in papers_div.css('a[href^="/paper/"]'):
            paper_link = response.urljoin(paper.attrib['href'])
            yield scrapy.Request(paper_link, callback=self.parse_paper)

    def parse_paper(self, response):
        # Assuming the PDF link is in the response, you may need to adjust this
        pdf_url = response.css('a[href$=".pdf"]::attr(href)').get()  # Adjust selector as needed
        if pdf_url:
            pdf_id = pdf_url.split('pdf/')[1].split('>')[0]  # Split by 'pdf/' and then by '>'

            if (pdf_id == "2004.03500v2.pdf" or pdf_id == "1708.09427v5.pdf"):
                pdf_url = response.urljoin(pdf_url)
                yield scrapy.Request(pdf_url, callback=self.parse_pdf)

    def parse_pdf(self, response):
        # Read the PDF file
        pdf_file = BytesIO(response.body)
        reader = PdfReader(pdf_file)

        for page in reader.pages:
            text = page.extract_text()
            if text:
                yield {"text": text}
