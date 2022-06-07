import requests
from bs4 import BeautifulSoup

COUNTRY_CODE = 'ru-RU'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': f'{COUNTRY_CODE},en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}


def google(q):
    s = requests.Session()
    q = '+'.join(q.split())
    url = 'https://www.google.com/search?q=' + q + '&ie=utf-8&oe=utf-8'
    r = s.get(url, headers=headers)

    soup = BeautifulSoup(r.text, "html.parser")
    output = []
    filter = soup.find_all("h3")
    for i in range(0, len(filter)):
        print(filter[i].get_text())
        output.append(filter[i].get_text())

    return output

