from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage={"root_dir": "./illust_img/ponkan"})
crawler.crawl(keyword="ぽんかん⑧　イラスト", max_num=200)

crawler = BingImageCrawler(storage={"root_dir": "./illust_img/ixy"})
crawler.crawl(keyword="ixy　イラスト", max_num=200)

crawler = BingImageCrawler(storage={"root_dir": "./illust_img/tokunou"})
crawler.crawl(keyword="得能正太郎　イラスト", max_num=200)

crawler = BingImageCrawler(storage={"root_dir": "./illust_img/makihituzi"})
crawler.crawl(keyword="巻羊 イラスト", max_num=200)

crawler = BingImageCrawler(storage={"root_dir": "./illust_img/anmi"})
crawler.crawl(keyword="anmi イラスト", max_num=200)

# crawler = BingImageCrawler(storage={"root_dir": "./illust_img/so-bin"})
# crawler.crawl(keyword="so-bin イラスト", max_num=200)
#
# crawler = BingImageCrawler(storage={"root_dir": "./illust_img/kantoku"})
# crawler.crawl(keyword="カントク イラスト", max_num=200)
