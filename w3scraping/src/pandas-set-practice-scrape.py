import requests
from bs4 import BeautifulSoup
import nbformat as nbf
headers = {'User-Agent': 'Mozilla/5.0'}
nb = nbf.v4.new_notebook()

for i in range(1,66):
    url=f"https://www.w3resource.com/python-exercises/pandas/practice-set1/pandas-practice-set1-exercise-{i}.php"
    response = requests.get(url,headers)
    soup = BeautifulSoup(response.content,"html.parser")
    question_ = soup.find("p",class_="")
    question=list(question_.children)[0]
    print(i)
    print(question)
    nb['cells'].append(nbf.v4.new_markdown_cell(str(i)+"."+question))
    ans = soup.find(class_='language-python',).text
    nb['cells'].append(nbf.v4.new_code_cell(ans))
    print(ans)
    print()

fname = 'pandas_set.ipynb'
with open(fname, 'w') as f:
    nbf.write(nb, f)