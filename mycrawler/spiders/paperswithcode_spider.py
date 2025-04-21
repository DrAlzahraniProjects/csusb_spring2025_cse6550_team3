import scrapy

class PaperSpider(scrapy.Spider):
    name = "paper_spider"
    allowed_domains = ['paperswithcode.com']
    start_urls = ['https://paperswithcode.com']


    def parse(self, response):
        # Follow all links, relying on allowed_domains for filtering
        for href in response.css('a::attr(href)').getall():
            yield response.follow(href, callback=self.parse_paper)


    def parse_paper(self, response):
        # Extract data
        title = response.css('h1::text').get()
        abstract = response.css('div.paper-abstract p::text').get()

        # Extract date and authors
        raw_date = response.css('div.authors span.author-span::text').get()
        authors = response.css('div.authors span.author-span a::text').getall()

        date = raw_date.strip() if raw_date else None
        authors = [a.strip() for a in authors if a.strip()]

        artefact_information = response.css('div.artefact-information p::text').getall()
        raw_parts_description = response.css('div.description p ::text').getall()
        description = ' '.join(part.strip() for part in raw_parts_description if part.strip())

        dataset_names = [name.strip() for name in response.css('div.sota-table-preview div.dataset a::text').getall()]
        best_model_names = [model.strip() for model in response.css('div.sota-table-preview tr td:nth-child(3) div.black-links a::text').getall() if model.strip()]
        dataset_links = [response.urljoin(link) for link in response.css('div.task-datasets a::attr(href)').getall()]
        benchmark_dataset_links = [response.urljoin(link) for link in response.css('div.sota-table-preview div.dataset a::attr(href)').getall()]

        paper_list_titles = [paper.strip() for paper in response.css('div#task-papers-list div.col-lg-9.item-content h1 a::text').getall()]
        paper_list_authors = [author.strip() for author in response.css('div#task-papers-list div.authors span.author-span::text').getall() if author.strip()]
        paper_list_title_links = [response.urljoin(link) for link in response.css('div#task-papers-list h1 a::attr(href)').getall()]
        paper_list_author_links = [response.urljoin(link) for link in response.css('div#task-papers-list span.item-github-link a::attr(href)').getall()]
        paper_list_dates = [date.strip() for date in response.css('div#task-papers-list div.col-lg-9.item-content span.item-date-pub::text').getall()]
        paper_list_abstracts = [abstract.strip() for abstract in response.css('div#task-papers-list div.col-lg-9.item-content p.item-strip-abstract::text').getall()]

        # Yield extracted data
        yield {
            "url": response.url,
            "title": title,
            "abstract": abstract,
            "date": date,
            "authors": authors,
            "artefact-information": artefact_information,
            "description": description,
            "benchmark_dataset_links": benchmark_dataset_links,
            "dataset_names": dataset_names,
            "best_model_names": best_model_names,
            "dataset_links": dataset_links,
            "paper_list_titles": paper_list_titles,
            "paper_list_title_links": paper_list_title_links,
            "paper_list_authors": paper_list_authors,
            "paper_list_author_links": paper_list_author_links,
            "paper_list_dates": paper_list_dates,
            "paper_list_abstracts": paper_list_abstracts
        }

        # Follow links from this page to continue crawling
        for href in response.css('a::attr(href)').getall():
            yield response.follow(href, callback=self.parse_paper)
