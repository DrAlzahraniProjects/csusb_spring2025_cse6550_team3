from scrapy.exporters import CsvItemExporter

class HeadlessCsvItemExporter(CsvItemExporter):
    def __init__(self, *args, **kwargs):
        if args[0].tell() > 0:
            kwargs['include_headers_line'] = False  # Skip headers if file has content
        super(HeadlessCsvItemExporter, self).__init__(*args, **kwargs)
