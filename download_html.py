import httpx
from urllib.parse import unquote
import re

def download(url):

    res = httpx.get(url)

    filename = unquote(url).split("/")
    filename = filename[-1].replace("_(5e_Race)","")
    filename = filename.replace("(5e_Race)","")
    filename = url
    
    file = open("htmls/" + filename + ".html", "w", encoding="utf-8")
    file.write(res.text)
    
    return filename

if __name__ == "__main__":

    count = 0
    limit = False

    names = {}

    with open("urls.txt") as file:
        for line in file:
            count += 1
            name = download(line[:-1])
            if name in names:
                if line[:-1] not in names[name]:
                    names[name].append(line[:-1])
            else:
                names[name] = [line[:-1]]
            print('%-120s%-60s' % (line[:-1], "\t" + name + ".html"))
            if limit and count >= 10:
                break

    dupes = False
    for name, urls in names.items():
        if len(urls) > 1:
            dupes = True
            print(f'the {name} has {len(urls)} urls:')
            for url in urls:
                print(url)

    if not dupes:
        print("No dupes :)")
    else:
        print("dupes")
