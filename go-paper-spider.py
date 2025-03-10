import os
import json
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from mycrawler.spiders.paperswithcode_spider import PaperSpider

# Path (mapped to Docker volume)
papers_output_file_path = "/data/papers_output.csv"

def run_paper_spider():
    """Run the PapersSpider and save the output."""

    settings = get_project_settings()
    settings.update({
        'FEED_URI': papers_output_file_path,
        'FEED_FORMAT': 'csv',
        'FEED_EXPORT_INDENT': 4,
        'FEED_EXPORT_ENCODING': 'utf-8',
        'FEED_OVERWRITE': False,
        'FEED_STORE_EMPTY': False,
        'FEED_EXPORT_FIELDS': ["text"],  # Ensure consistent column order
        'FEED_EXPORTERS': {
        'csv': 'mycrawler.exporters.HeadlessCsvItemExporter',
        },
    })

    process = CrawlerProcess(settings)
    process.crawl(PaperSpider)
    process.start()

def main():
    print("Starting the paper scraping process...")
    run_paper_spider()
    print("Scraping paper complete...")

if __name__ == "__main__":
    main()