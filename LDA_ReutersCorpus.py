# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup,SoupStrainer
import os, sys, codecs, string, time
from gensim import corpora, models, similarities
from itertools import chain
from collections import Counter

start = time.time()

#===================
# CONFIGURATION
raw_files_path = "raw/reuters/"
outputfile = "final/reuterstopics"
ntopics = 90
#===================

print "OPENING SOURCEFILES..."
arrayList = []
for filename in os.listdir(raw_files_path):
	source = open(raw_files_path + filename, "r")
 	data = source.read()
	soup = BeautifulSoup(data)
	contents = soup.findAll('content')
	for content in contents:
		arrayList.append(str(content.text.encode('utf-8').strip('.,+-:;<>').replace('\n', ' ')))
	source.close()

documents = arrayList

print "REMOVING MOST COMMON WORDS..."
# remove common words and tokenize
#ENGLISH STOPLIST from http://www.paulnoll.com/Books/Clear-English/words-01-02-hundred.html
stoplist = set("a &#3; \x03 per - -- --- vs cts said. pct mln mlns dlr dlrs reuter about after again air all along also an and another any are around as at away back be because been before below between both but by came can come could day did different do does don't down each end even every few find first for found from get give go good great had has have he help her here him his home house how I if in into is it its just know large last left like line little long look made make man many may me men might more most Mr. Mr must y name never new next no not now number of off old on one only or other our out over own part people place put read right said same saw say see she sould show small so some soething sound still such take tell than that the them then there these they thing think this those thought three through time to together too tow under up us use very want way we wel went were what when where which while who why will with word work world would write year you your was".split())
#GERMAN
#stoplist = set('der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch es an werden aus er hat dass sie nach wird bei einer der um am sind noch wie einem über einen das so sie zum war haben nur oder aber vor zur bis mehr durch man sein wurde sei in hatte kann gegen vom schon wenn haben seine ihre dann unter wir soll ich eines diese dieser wieder keine seiner worden will zwischen im immer was sagte -'.split())
bigtexts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

print "REMOVING WORDS THAT APPEAR ONLY ONCE..."
# remove words that appear only once
c = Counter(chain.from_iterable(bigtexts))
texts = [[word for word in x if c[word]>1] for x in bigtexts]

print "CREATING DICTIONARY..."
# Create Dictionary.
id2word = corpora.Dictionary(texts)
# Creates the Bag of Word corpus.
mm = [id2word.doc2bow(text) for text in texts]

print "TRAINING LDA MODELS..."
# Trains the LDA models.
lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=ntopics, \
                               update_every=1, chunksize=2000, passes=1)

print "PRINTING FILE..."
# Prints the topics into a textfile.
filename = outputfile + str(ntopics) + ".txt"
file = open(filename, "w")
topiccounter = 0
for top in lda.print_topics(ntopics):
  topiccounter +=1
  file.write("topic #" + str(topiccounter) + "\n" + top + "\n\n")

# write all headlines into the file
#file.write(str(documents))
file.close()

processing_time = time.time() - start

print "Done: " + filename +"/ Processing time: " + str(round(processing_time/60,2)) + " minutes."  