import os
import json
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from mycrawler.spiders.timestamp_spider import TimestampSpider
from mycrawler.spiders.paperswithcode_spider import PaperSpider

# Paths (mapped to Docker volume)
timestamp_file = "/data/last_timestamp.json"
output_file_path = "/data/output.csv"

def get_website_timestamp():
    """Fetch the current timestamp from the website."""
    process = CrawlerProcess(get_project_settings())
    timestamp = None

    def store_timestamp(item):
        nonlocal timestamp
        timestamp = item["timestamp"]

    process.crawl(TimestampSpider)
    for _ in process.crawl(TimestampSpider):
        pass
    process.start()

    return timestamp

def get_stored_timestamp():
    """Get the last stored timestamp from the file."""
    if os.path.exists(timestamp_file):
        with open(timestamp_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data.get("timestamp")
    return None

def update_stored_timestamp(timestamp):
    """Update the stored timestamp in the file."""
    with open(timestamp_file, "w", encoding="utf-8") as file:
        json.dump({"timestamp": timestamp}, file)

def run_spider():
    """Run the QuotesSpider and save the output."""
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    settings = get_project_settings()
    settings.update({
        'FEED_URI': output_file_path,
        'FEED_FORMAT': 'csv',
        'FEED_EXPORT_INDENT': 4,
        'FEED_EXPORT_ENCODING': 'utf-8',
        'FEED_OVERWRITE': True,
    })

    process = CrawlerProcess(settings)
    process.crawl(PaperSpider)
    process.start()

def main():
    # Get the current and stored timestamps
    # current_timestamp = get_website_timestamp()
    # stored_timestamp = get_stored_timestamp()

# if current_timestamp != stored_timestamp:
    print("Website has been updated. Running scraper...")
    run_spider()
    # update_stored_timestamp(current_timestamp)
    print("Scraping complete. Timestamp updated.")
# else:
    print("Website has not changed. Skipping scraper.")

if __name__ == "__main__":
    main()