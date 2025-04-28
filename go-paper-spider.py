from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from mycrawler.spiders.paperswithcode_spider import PaperSpider

# Path (mapped to Docker volume)
papers_output_file_path = "/data/papers_output.json"

def run_paper_spider():
    """Run the PapersSpider and save the output."""

    settings = get_project_settings()
    settings.update({
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
            'json': 'scrapy.exporters.JsonItemExporter',  # Use Scrapy's built-in JSON exporter
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