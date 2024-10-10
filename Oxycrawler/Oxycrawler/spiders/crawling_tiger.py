from pathlib import Path
from urllib.parse import urlparse, urlunparse
import scrapy
import os, shutil
from bs4 import BeautifulSoup

class tScrape(scrapy.Spider):
    # Define the name of the spider
    name = 'oxy'

    # Define the allowed domains
    allowed_domains = ["oxy.edu", 'oxyathletics.com']

    # 'theoccidentalnews.com' [To be scraped later]

    def __init__(self):
        super().__init__()
        #Create a file to store scraped urls
        self.scraped_urls_file = 'scraped_urls.txt'
        # Create the file if it doesn't exist and load previously scraped URLs
        if not Path(self.scraped_urls_file).exists():
            Path(self.scraped_urls_file).write_text('')
        with open(self.scraped_urls_file, 'r') as f:
            self.scraped_urls = set(f.read().splitlines())
        # Create a folder to store the scraped files
        self.target_folder = 'web_data'
        if not Path(self.target_folder).exists():
            Path(self.target_folder).mkdir()

    def start_requests(self):
        urls = [
            'https://oxy.edu/'
        ]
        # Other Links to potentially scrape
        ''''https://theoccidentalnews.com/',
            'https://oxyathletics.com/'
        '''
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
    
    def parse(self, response):
        # Extract the title of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get the canonical URL if available
        canonical_link = soup.find("link", rel="canonical")
        if canonical_link:
            canonical_url = canonical_link['href']
        else:
            canonical_url = self.strip_url_parameters(response.url)

        # Skip processing if this canonical URL has already been visited
        if canonical_url in self.scraped_urls:
            self.log(f"Skipping duplicate page: {canonical_url}")
            return


        # Remove script and style tags
        for script in soup(["script", "style", "nav", "noscript", "header", "footer"]):
            script.extract()
        
        # Work to make paragraphs
        paragraphs = []
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'tr', 'td']):
            text = element.get_text(separator=' ', strip=True) 
            if text: 
                paragraphs.append(text)   
        
        plain_text = '\n'.join(paragraphs)


        ''' [Prior join code]
        plain_text = soup.get_text(separator=' ', strip=True) if soup.body else ""
        plain_text = ' '.join(plain_text.split())'''

        '''# Check if scraping an HTML page [Code from Scrapy]
        content_type = response.headers.get('Content-Type', b'').decode('utf-8')
        if 'text/html' not in content_type:
            self.log(f"Content-Type is {content_type}, not scraping.")
            return

        # Extract text from the page
        # plain_text = response.xpath("//body//*[not((contains(@class, "layout-no-sidebars")) or self::script or self::style or self::nav)]//text()").getall()#response.css('::text').getall()
        plain_text = response.xpath(
        "//body//*[not(self::script or self::style or self::nav or self::div or self::noscript)]//text()"
        ).getall()'''

        if plain_text:
            # Join the list of text into a single stringß
            #plain_text = " ".join(plain_text).strip()
            self.log(f"Extracted text: {plain_text[:200]}...")  # Log the first 200 characters for debugging
        else:
            self.log("No text found. Check the XPath expression.")
            plain_text = "No text extracted from the page."

        # Extract the domain from the URL
        domain = urlparse(response.url).netloc

        #plain_text = ' '.join(plain_text).strip()
        page = response.url.split("/")[-2]
        filename = f'{domain}-{page}.txt'
        file_path = os.path.join(self.target_folder, filename)

        try:
            with open(file_path, 'w') as f:
                f.write(plain_text)
            self.log(f'Saved file {filename}')
        except Exception as e:
            self.log(f"Error saving file: {str(e)}")
        
            '''Path(filename).write_text(plain_text)
            self.log(f'Saved file {filename}')
            '''
         # Save the URL to the list of scraped URLs
        with open(self.scraped_urls_file, 'a') as f:
            f.write(response.url + '\n')
        self.scraped_urls.add(response.url)

        # Move the file to the target folder
        try:
            if filename not in [self.scraped_urls_file, self.links_file]:
                shutil.move(filename, self.target_folder + filename)
                self.log(f'Moved {filename} to {self.target_folder}')
        except Exception as e:
            self.log(f"Error moving file: {str(e)}")

        # Extract all links and follow them
        links = response.xpath('//a/@href').getall()
        for link in links:
            # Handle relative links by generating full URLs
            if link and link.startswith('/'):
                link = response.urljoin(link)
            
            # Check if link is in the allowed domains and hasn't been scraped yet
            if any(domain in link for domain in self.allowed_domains) and link not in self.scraped_urls:
                yield scrapy.Request(url=link, callback=self.parse)
        
        def strip_url_parameters(self, url):
            """Strip query parameters from the URL to avoid duplicates."""
            parsed_url = urlparse(url)
            stripped_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, '', '', ''))
            return stripped_url

        '''# Save each link to the links.txt file
        with open('links.txt', 'w') as f:
            f.write(link + '\n')
            # Follow the link
            yield scrapy.Request(url=link, callback=self.parse)'''
    

        '''# Extract Links
        links = response.xpath('//a/@href').getall() # Extracts all links on the page
        for link in links:
            # If the link starts with a '/', join it with the base URL
            if link and link.startswith('/'):
                link = response.urljoin(link)
                self.log(f"Found link: {link}")
            # Writes each link into a file
            with open('links.txt', 'w') as f:
                f.write(f'{link}\n')
            # Follows the link
            yield scrapy.Request(link, callback=self.parse)
            '''