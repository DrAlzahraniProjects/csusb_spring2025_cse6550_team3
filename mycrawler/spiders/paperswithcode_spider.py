import scrapy

class PaperSpider(scrapy.Spider):
    name = "paper_spider"
    allowed_domains = ['paperswithcode.com']
    start_urls = ['https://paperswithcode.com']

    def parse(self, response):
        # Yield the raw HTML and the URL of the page
        yield {
            "url": response.url,
            "raw_html": response.text
        }

        # Extract all links from the page
        links = response.css('a::attr(href)').getall()
        for link in links:
            absolute_url = response.urljoin(link)
            # Only follow links within the allowed domains
            if any(domain in absolute_url for domain in self.allowed_domains):
                yield scrapy.Request(absolute_url, callback=self.parse)