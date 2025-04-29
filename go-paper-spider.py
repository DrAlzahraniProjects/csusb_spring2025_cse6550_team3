from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from mycrawler.spiders.paperswithcode_spider import PaperSpider

# Path (mapped to Docker volume)
papers_output_file_path = "/data/papers_output.json"

def run_paper_spider():
    """Run the PapersSpider and save the output."""

    settings = get_project_settings()
    settings.update({
        # Feed/export settings
        'FEED_URI': papers_output_file_path,
        'FEED_FORMAT': 'json',
        'FEED_EXPORT_INDENT': 4,
        'FEED_EXPORT_ENCODING': 'utf-8',
        'FEED_OVERWRITE': False,
        'FEED_STORE_EMPTY': False,
        'FEED_EXPORT_FIELDS': [
            "url", "title", "abstract", "date", "authors",
            "artefact-information", "description", "benchmark_dataset_links",
            "dataset_names", "best_model_names", "dataset_links",
            "paper_list_titles", "paper_list_title_links",
            "paper_list_authors", "paper_list_author_links",
            "paper_list_dates", "paper_list_abstracts"
        ],
        'FEED_EXPORTERS': {
            'json': 'scrapy.exporters.JsonItemExporter',
        },

        # Throttling / performance settings
        'CONCURRENT_REQUESTS': 2,                  # Super low concurrency
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,        # Only 1 request at a time per domain
        'DOWNLOAD_DELAY': 3,                        # Wait 3 seconds between requests
        'RANDOMIZE_DOWNLOAD_DELAY': True,           # Add randomness to delay
        'AUTOTHROTTLE_ENABLED': True,               # AutoThrottle enabled
        'AUTOTHROTTLE_START_DELAY': 5,              # Start with 5 sec delay
        'AUTOTHROTTLE_MAX_DELAY': 20,               # Up to 20 sec if server slow
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 0.5,     # Try to keep 0.5 active requests at a time
        'FEED_EXPORT_BATCH_ITEM_COUNT': 20,         # Write every 20 items to disk
    })

    process = CrawlerProcess(settings)
    process.crawl(PaperSpider)
    process.start()

def main():
    print("Starting the paper scraping process...")
    # run_paper_spider()
    print("Scraping paper complete...")

if __name__ == "__main__":
    main()