# -*- coding: utf-8 -*-
import os, sys, codecs, csv, string, time
from gensim import corpora, models, similarities
from itertools import chain
import pandas as pd
from collections import Counter

start = time.time()

#===================
outputfile = "final/nyt_lt"
ntopics = 10
#===================

print "FETCHING AND SANITIZING HEADLINES..."
#==== LONDON TIMES
rawlist1 = [line.rstrip().replace('.', '').replace(':', '').replace(';', '') for line in open("raw/londontimes.txt")]
identity1 = string.maketrans("", "")
document1 = [s.translate(identity1, "-?.,;:") for s in rawlist1]

#==== NY TIMES
#reads raw csv
data = pd.read_csv("raw/NYTimes.csv")
#renames columns
data.columns = ["ID", "date", "title", "subject", "topic_code"]
#make a list of all in that specific column
rawlist2 = list(data.title)
#removes all unwanted characters in the list
identity2 = string.maketrans("", "")
document2 = [s.translate(identity2, "-?.,;:") for s in rawlist2]

#==== MERGE HEADLINES
documents = document1 + document2

print "REMOVING MOST COMMON WORDS..."
# remove common words and tokenize
#ENGLISH STOPLIST
stoplist = set("a about above after again against all am an and any are aren't as at be because been before being below between bothbut by can't cannot could didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't havinghhe'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whomwhywhy's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves".split())
#GERMAN
#stoplist = set('der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch es an werden aus er hat dass sie nach wird bei einer der um am sind noch wie einem über einen das so sie zum war haben nur oder aber vor zur bis mehr durch man sein wurde sei in hatte kann gegen vom schon wenn haben seine ihre dann unter wir soll ich eines diese dieser wieder keine seiner worden will zwischen im immer was sagte -'.split())
bigtexts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

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
                               update_every=1, chunksize=500, passes=1)

print "PRINTING FILE..."
filename = outputfile + str(ntopics) + ".txt"
file = open(filename, "w")

# Prints the topics.
topiccounter = 0
for top in lda.show_topics():
  topiccounter +=1
  file.write("topic #" + str(topiccounter) + "\n" + top + "\n\n")

# write all headlines into the file
#file.write(str(documents))
file.close()

processing_time = time.time() - start

print "Done: " + filename +"/ Processing time: " + str(round(processing_time/60,2)) + " minutes."  