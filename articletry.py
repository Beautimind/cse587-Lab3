from nytimesarticle import articleAPI
import time
api = articleAPI('b44f49f6205b4155b5e42481a797bf40')
articles = api.search( 
     fq = {'news_desk':["sports"]}, 
     begin_date = 20141231,page=str(50))
print articles

import urllib2
from bs4 import BeautifulSoup
def parse_articles(articles):
    '''
    This function takes in a response to the NYT api and parses
    the articles into a list of dictionaries
    '''
    news = []
    for i in articles['response']['docs']:
        dic = {}
        #dic['id'] = i['_id']
        #if i['abstract'] is not None:
        #    dic['abstract'] = i['abstract'].encode("utf8")
        dic['headline'] = i['headline']['main'].encode("utf-8")
        #dic['desk'] = i['news_desk']

        if i['web_url'] is not None:
            print "Download begin: "+i['web_url']
            page = urllib2.urlopen(i['web_url']).read()
            soup = BeautifulSoup(page,"lxml")
            paragraphs = soup.find_all('p', class_='story-body-text story-content')
            # print type(paragraphs)
            text=''
            for i in paragraphs:
                text=text+i.get_text()
            dic['story-content'] = text.encode("utf-8")
            print len(dic['story-content'])
        if len(dic['story-content'])>100:
        	news.append(dic)
        	print "can use"
    return(news)    

def get_articles(date,query):
    '''
    This function accepts a year in string format (e.g.'1980')
    and a query (e.g.'Amnesty International') and it will 
    return a list of parsed articles (in dictionaries)
    for that year.
    '''
    all_articles = []
    for i in range(0,5): #NYT limits pager to first 100 pages. But rarely will you find over 100 pages of results anyway.
        articles = api.search(
               fq = {'news_desk':["science"],'source':['The New York Times']},
               begin_date = '20120501',
               end_date ='20140511',
               sort='oldest',
               page = str(i))
        time.sleep(0.5)
        if "response" in articles:
            articles = parse_articles(articles)
            all_articles = all_articles + articles
    return(all_articles)

'''import time
for i in range(1,100):
    articles = api.search( q = 'China', 
    fq = {'headline':'China', 'source':['Reuters','AP', 'The New York Times']}, 
    begin_date = 20141231,page=str(i))
    print articles.keys()
    time.sleep()
    #parse_articles(articles);'''


Amnesty_all = get_articles('2017','Obama')

import csv
keys = Amnesty_all[0].keys()
with open('Uknownpscience.csv', 'wb') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(Amnesty_all)