import requests
from bs4 import BeautifulSoup
headers = {'User-Agent': 'Mozilla/5.0'}
url=f"https://www.w3resource.com/python-exercises/pandas/python-pandas-data-series-exercise-1.php"
response = requests.get(url,headers)
soup = BeautifulSoup(response.content,"html.parser")
question_ = soup.find("p",class_="")
question=list(question_.children)[0]