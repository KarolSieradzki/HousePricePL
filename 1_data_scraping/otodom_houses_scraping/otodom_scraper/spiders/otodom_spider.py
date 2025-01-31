import scrapy
import os
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from scrapy.utils.project import get_project_settings
from scrapy.http import HtmlResponse
import time
import re

# Disable warnings in console
import logging
from selenium.webdriver.remote.remote_connection import LOGGER
LOGGER.setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# 
# TU DODAJ KOMENTARZ JAK DZIAŁA CAŁY PROCES SCRAPOWANIA
# 
class OtodomSpider(scrapy.Spider):
    name = 'otodom_spider'
    start_urls = ['https://www.otodom.pl/pl/wyniki/sprzedaz/dom/cala-polska?ownerTypeSingleSelect=ALL&viewType=listing&by=LATEST&direction=DESC&limit=72&page=1']
    
    def __init__(self, *args, **kwargs):
        super(OtodomSpider, self).__init__(*args, **kwargs)
        self.properities = []
        self.already_loaded_links = set()
        self.load_existing_data()
        self.driver = self.init_webdriver()

    def init_webdriver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        return webdriver.Chrome(options=chrome_options)

    def parse(self, response):
        pages_count = self.get_pages_count(response)
        print(f"Liczba stron brana po uwagę: {pages_count}")

        for page_num in range(1, pages_count+1):
            url = f'https://www.otodom.pl/pl/wyniki/sprzedaz/dom/cala-polska?ownerTypeSingleSelect=ALL&viewType=listing&by=LATEST&direction=DESC&limit=72&page={page_num}'
            yield scrapy.Request(url, callback=self.parse_page, meta={'page_num': page_num})

    def parse_page(self, response):
        links = response.css('a[data-cy="listing-item-link"]::attr(href)').getall()
        page_num = response.meta['page_num']

        print(f'Pobrano {len(links)} linków ze strony {page_num}')

        # check if offer is already scraped
        for link in links:
            url = response.urljoin(link)
            if self.is_duplicate(url):
                print(f'Zduplikowany link {url}, pominięto')
                continue
            
            yield scrapy.Request(url, callback=self.parse_property, meta={'page_num': page_num, 'link': url})

    def parse_property(self, response):
        self.driver.get(response.url)

        self.scroll_and_wait()
    
        # Load latitude and longitude
        lat, long = "Brak informacji", "Brak informacji"
        try:
            google_maps_link = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a[title*="Pokaż ten obszar w Mapach Google (otwiera się w nowym oknie)"]'))
            )
            if google_maps_link:
                href = google_maps_link.get_attribute("href")
                match = re.search(r"ll=([0-9.\-]+),([0-9.\-]+)", href)
                if match:
                    lat = match.group(1)
                    long = match.group(2)

        except Exception as e:
            print(f"Error extracting coordinates: {e}")

        #load other property data
        property_data = {
            'link': response.meta['link'],
            'page_number': response.meta['page_num'],
            'Latitude': lat,
            'Longitude': long,
            **self.get_property_details(response)
        }

        print(f"Pobrano dane dla: {property_data['link']}, Latitude: {lat}, Longitude: {long}")

        self.properities.append(property_data)
        self.save_links_to_json()

    def get_property_details(self, response):
        details = self.get_static_details(response)
        details.update(self.get_dynamic_details(response))
        return details

    def get_static_details(self, response):
        static_details = {}

        fields = [
            ("Price", 'strong[data-cy="adPageHeaderPrice"]::text'),
            ("Price per sqm", 'div.css-8pg163.e1k1vyr24 div[aria-label="Cena za metr kwadratowy"]::text'),
            ("Area", 'div.css-58w8b7.eezlw8k0 button:nth-of-type(1) div.css-1ftqasz::text'),
            ("Rooms count", 'div.css-58w8b7.eezlw8k0 button:nth-of-type(2) div.css-1ftqasz::text'),
            ("Address", 'div.css-70qvj9.e42rcgs0 a.css-1jjm9oe.e42rcgs1::text'),
            ("Real estate office name", 'strong[aria-label="Nazwa agencji"].css-15tvki.ee7h84b0::text'),
            ("Date", 'p.e1gioeue5.css-xydenf::text')
        ]

        for field_name, css_selector in fields:
            value = response.css(css_selector).get()
            static_details[field_name] = value.strip() if value else "Brak informacji"

        return static_details

    def get_dynamic_details(self, response):
        dynamic_details = {}

        #get dynamic values from table
        rows = response.css("div.css-t7cajz.e15n0fyo1")
        for row in rows:
            key = row.css("p.e15n0fyo2::text").get()
            if key:
                key = key.replace(":", "").strip()

            value_element = row.css("p.e15n0fyo2")

            if len(value_element) > 1:
                value = value_element[1].css("::text").getall()
                span_values = value_element[1].css("span.css-axw7ok.e15n0fyo4::text").getall()

                value = span_values if span_values else value
                value = ", ".join([v.strip() for v in value]) if value else "Brak informacji"
            else:
                value = "Brak informacji"

            if key:
                dynamic_details[key] = value

        return dynamic_details


    def save_links_to_json(self):
        base_dir = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(base_dir, '../../../results/otodom_houses.json')

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.properities, f, ensure_ascii=False, indent=1)
                print(f'Zapisonao dane do otodom_houses.json, aktualna liczba nieruchomości to: {len(self.properities)}')
        except Exception as e:
            print(f'Błąd podczas zapisywania danych do pliku otodom_houses.json: {e}')


    def load_existing_data(self):
        base_dir = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(base_dir, '../../../results/otodom_houses.json')

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    self.already_loaded_links = {item['link'] for item in existing_data}
                    self.properities = existing_data

                    print(f'Załadowano istniejące dane: {len(self.already_loaded_links)} linków')

            except Exception as e:
                print(f'Błąd podczas ładownaia danych z pliku otodom_houses.json: {e}')

    def get_pages_count(self, response):
        page_numbers = response.css('ul[data-cy="frontend.search.base-pagination.nexus-pagination"] li.css-43nhzf::text').getall()
        print(f'page numbers: {page_numbers}')
        page_numbers = [ page for page in page_numbers if page.isdigit() ]

        if page_numbers:
            max_page = max(page_numbers, key=int)
            print(f"Największy numer strony: {max_page}")
            return int(max_page)
        else: 
            print("Nie znaleziono największego numeru strony")
            return 805
        
    def is_duplicate(self, link):
        return link in self.already_loaded_links
    
    def scroll_and_wait(self, pause_time=1, increment=0.3):
        total_height = self.driver.execute_script("return document.body.scrollHeight")
        current_scroll_position = 0

        while current_scroll_position < total_height:
            current_scroll_position += total_height * increment
            self.driver.execute_script(f"window.scrollTo(0, {current_scroll_position});")
            time.sleep(pause_time)
            total_height = self.driver.execute_script("return document.body.scrollHeight")

