from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from mycrawler.spiders.paperswithcode_spider import PaperSpider
import os

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
        'FEED_OVERWRITE': True, 
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

        # Politeness settings
        'CONCURRENT_REQUESTS': 4,        
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,  
        'DOWNLOAD_DELAY': 1,              
        'RANDOMIZE_DOWNLOAD_DELAY': True,   
        'AUTOTHROTTLE_ENABLED': True,       
        'AUTOTHROTTLE_START_DELAY': 2,     
        'AUTOTHROTTLE_MAX_DELAY': 10,       
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 1.0,
    })

    process = CrawlerProcess(settings)
    process.crawl(PaperSpider)
    process.start()

def main():
    # Ensure the output file is cleared before running
    if os.path.exists(papers_output_file_path):
        os.remove(papers_output_file_path)
    print("Starting the paper scraping process...")
    run_paper_spider()
    print("Scraping paper complete...")

if __name__ == "__main__":
    main()