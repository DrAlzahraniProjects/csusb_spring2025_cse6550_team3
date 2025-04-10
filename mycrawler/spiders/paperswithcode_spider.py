class PaperSpider(scrapy.Spider):
    name = "paper_spider"
    allowed_domains = ['paperswithcode.com']
    start_urls = [
        'https://paperswithcode.com'
    ]

    def parse(self, response):
        # follow every in-domain link
        for href in response.css('a::attr(href)').getall():
            url = response.urljoin(href)
            if any(domain in url for domain in self.allowed_domains):
                yield response.follow(url, callback=self.parse_paper)

    def parse_paper(self, response):
        title = response.css('h1::text').get()
        abstract = response.css('div.paper-abstract p::text').get()

        # grab and clean the spans for date + authors
        raw = response.css('div.authors span.author-span::text').getall()
        clean = [s.strip() for s in raw if s.strip()]
        date = clean[0] if clean else None
        authors = clean[1:] if len(clean) > 1 else []

        artefact_information = response.css('div.artefact-information p::text').getall()

        # join all text nodes in the description (including <strong>, <em>, etc.)
        raw_parts_description = response.css('div.description p ::text').getall()
        description = ' '.join(part.strip() for part in raw_parts_description if part.strip())

        # Extract dataset names (from the second column)
        dataset_names = [
            name.strip()
            for name in response.css('div.sota-table-preview div.dataset a::text').getall()
        ]

        # Extract best model names (from the third column)
        best_model_names = [
            model.strip()
            for model in response.css('div.sota-table-preview tr td:nth-child(3) div.black-links a::text').getall()
            if model.strip()  # Skip empty strings
        ]

        # dataset_links = response.css('div.task-datasets a::attr(href)').getall()
        dataset_links = [
            response.urljoin(link)
            for link in response.css('div.task-datasets a::attr(href)').getall()
        ]

        # Extract paper titles
        paper_list_titles = [
            paper.strip()
            for paper in response.css('div#task-papers-list div.col-lg-9.item-content h1 a::text').getall()
        ]

        paper_list_authors = [
            author.strip()
            for author in response.css('div#task-papers-list div.authors span.author-span::text').getall()
            if author.strip()
        ]

        # In parse_paper() method:
        paper_list_title_links = [
            response.urljoin(link)
            for link in response.css('div#task-papers-list h1 a::attr(href)').getall()
        ]

        paper_list_author_links = [
            response.urljoin(link)
            for link in response.css('div#task-papers-list span.item-github-link a::attr(href)').getall()
        ]

        benchmark_dataset_links = [
            response.urljoin(link)
            for link in response.css('div.sota-table-preview div.dataset a::attr(href)').getall()
        ]

        # Extract publication dates
        paper_list_dates = [
            date.strip()
            for date in response.css('div#task-papers-list div.col-lg-9.item-content span.item-date-pub::text').getall()
        ]

        # Extract abstracts
        paper_list_abstracts = [
            abstract.strip()
            for abstract in response.css('div#task-papers-list div.col-lg-9.item-content p.item-strip-abstract::text').getall()
        ]

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