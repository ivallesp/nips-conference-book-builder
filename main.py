from bs4 import BeautifulSoup
import urllib3
import time
import codecs
import json
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
http = urllib3.PoolManager()

url_index = "https://neurips.cc/Conferences/2018/Schedule"

response = http.request('GET', url_index)
soup = BeautifulSoup(response.data)
divs=soup.select("#main > div.col-xs-12.col-sm-9 > div")
events=list(filter(lambda x: "showDetail" in str(x), divs))

def get_event_info(event):
    description_url = "https://neurips.cc/Conferences/2018/Schedule?showEvent="

    title = event.find("div", {"class": "maincardBody"}).text
    url = description_url + event.attrs["onclick"][-6:-1]
    abstract = BeautifulSoup(http.request('GET', url).data).find("div", {"class": "abstractContainer"}).text
    authors = event.find("div", {"class": "maincardFooter"}).text
    type_event = event.find("div", {"class": "maincardHeader"}).text
    date_location = event.findAll("div", {"class": "maincardHeader"})[1].text
    return {"title": title,
           "url": url,
           "abstract": abstract,
           "authors": authors,
           "type": type_event,
           "date_location": date_location}

def generate_markdown_entry(element):
    document = ""
    document += "## " + "[" + element["title"] + "]" + "(" + element["url"] + ")" + "\n"
    document += "**" + element["type"] + " | " + element["date_location"] + "**\n"
    document += "*" + element["authors"] + "*\n"
    document += element["abstract"]
    return document

# Extract the information of each entry
entries = []
for event in tqdm(events):
    count_errors=0
    while count_errors<3:
        try:
            entry = get_event_info(event)
            count_errors = 999
            entries.append(entry)
        except:
            count_errors+=1
            time.sleep(1)
            print("Error found status = {0}".format(count_errors))
            
with codecs.open("entries.json", "wb", "utf-8") as f:
    f.write("\n".join(map(lambda x: json.dumps(x), entries)))        
    
# Build the markdown document
documents = list(map(generate_markdown_entry, entries))
conference_book = "\n\n_________________\n_n".join(documents)
with codecs.open("conference_book.md", "wb", "utf-8") as f:
    f.write(conference_book)

