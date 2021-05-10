import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

motivational_subjects = ["motivation","self-improvement","inspiration","personal-development","Motivational","Inspirational","Success"]
non_motivational_subjects = ["metoo","terrorism","palestine","racism","politic","sexual-assault","Race","Abuse","war","syria"]
subjects = [motivational_subjects]+[non_motivational_subjects]
month = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

for j in range(len(subjects)):
    print("sub",subjects[j])
    subject = subjects[j]
    i = 0
    totCount = 0
    print("motivational subjects")
    data = []
    for s in subject:
        url = 'https://medium.com/search?q='+s

        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        blogs = soup.find_all('div', class_='postArticle-readMore')
        
        count = 0
        print("\n ....gathering",len(blogs)," blog for subject",s,"...\n")

        for blog in blogs:
            each_blog = []
            # print("3",blog)
            link = blog.find('a')['href']
            print("gathering \"",link,"\" data")

            page = requests.get(link)
            soup = BeautifulSoup(page.text, 'html.parser')
            soup = soup.find('article')

            if(j==0):
                name = '../data/raw/motivational/{}.txt'.format(i)
            else:
                name = '../data/raw/nonMotivational/{}.txt'.format(i)
            paragraphs = soup.find_all('p')
            pragraph_end = True
            if(len(paragraphs) >10):
              
                story_paragraphs=[]
                f = open(name,'w')
               
                for paragraph in paragraphs:
                    t = str((paragraph.text).encode("utf-8"))
                    if paragraph==paragraphs[-1] and (re.search("Thank you for reading",t,re.IGNORECASE) or\
                        re.search("subscribe",t,re.IGNORECASE) or\
                        re.search("Written by",t,re.IGNORECASE) or\
                        re.search("Thanks for reading",t,re.IGNORECASE) or\
                        re.search("published",t,re.IGNORECASE) or\
                        re.search("follow",t,re.IGNORECASE) or\
                        re.search("Recommend button",t,re.IGNORECASE)):
                        continue
                    else:
                        story_paragraphs.append(paragraph.text)
                        text = str((paragraph.text).encode("utf-8"))
                        f.write(text[2:-1])
                        f.write('\n')

                f.close()
                each_blog.append(link)
                each_blog.append(s)
                each_blog.append("{}.txt".format(i))
                strPa = (''.join(str(str(par).encode("utf-8")) for par in story_paragraphs))
                count += len(strPa.split(" "))
                each_blog.append(len(strPa.split(" ")))
                each_blog.append(j)
                each_blog.append(story_paragraphs)

                data.append(each_blog)
                print("i",i)
                i+=1    

        
    columns = ['link','subject','name','count','class','text']
    print("count for subject",s,":",count)

    totCount+=count
    print("total count",totCount)

    df = pd.DataFrame(data, columns=columns)
    if(j==0):
        df.to_csv('../data/raw/motivational.csv', index=False)
    else:
        df.to_csv('../data/raw/nonMotivational.csv', index=False)