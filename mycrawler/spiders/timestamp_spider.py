import scrapy

class TimestampSpider(scrapy.Spider):
    name = "timestamp"
    start_urls = ["https://quotes.toscrape.com"]

    def parse(self, response):
        # Example: Extract a timestamp from a meta tag
        timestamp = response.css('meta[name="last-modified"]::attr(content)').get()
        if timestamp:
            yield {"timestamp": timestamp}