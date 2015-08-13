'''
Created on May 4, 2015

@author: hongning

removing possible duplicates from the crawled content
'''
import json
from os import listdir
from os.path import isfile, join

def saveJSON(filename, jsonObj):
    with open(filename, 'w') as writer:
        writer.write(json.dumps(jsonObj))

def removeAmazonDuplicates(inputfolder, outputfolder, suffix='json'):
    reviewIDlist = set()
    contentList = set()
    
    dupSize = 0
    reviewSize = 0
    for f in listdir(inputfolder):
        filename = join(inputfolder, f) 
        if isfile(filename) and f.endswith(suffix):
            prod = json.load(open(filename))
            reviews = []
            for review in prod['Reviews']:
                content = '' # dedup by content
#                if review['Title'] is not None:
#                    content = review['Title'] + ' '
                if review['Content'] is not None:
                    content += review['Content']
                
                if review['ReviewID'] not in reviewIDlist and content not in contentList:
                    reviewIDlist.add(review['ReviewID'])
                    contentList.add(content)
                    reviews += [review]
                else:
                    dupSize += 1
                reviewSize += 1
            if len(reviews) > 0:
				saveJSON(join(outputfolder, f), {'Reviews':reviews})
#                saveJSON(join(outputfolder, f), {'Reviews':reviews, 'ProductInfo':prod['ProductInfo']})
	print 'Removing %d out of %d duplicates...' % (dupSize, reviewSize)

if __name__ == '__main__':
    removeAmazonDuplicates('./YelpData/RawData/', './YelpData/dedup/')
