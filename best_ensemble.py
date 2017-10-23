import io
import string
import re
import pickle
from sklearn import tree
from sklearn import svm
t = io.open('test_set_x.csv', 'r', encoding='utf-8')
langprobs = [0.0517255231348,0.514063470036,0.250794288427,0.132125771464,0.0512909469379]
chartotals = [601861,5696615,2521698,15024380,460159]
d = {}
latinonly = {}


##construct character count stats for each language
x = io.open('train_set_x.csv', 'r', encoding='utf-8')
y = io.open('train_set_y.csv', 'r', encoding='utf-8')


x.readline()
y.readline()

for line in x:
  splitline = line.split(',')
  yr = y.readline().split(',')

  utt = splitline[1].replace(u" ", "").lower()
  for c in utt:
    if c == u'\n':
      continue
    if c not in d.keys():
      d[c] = 5*[0]
    d[c][int(yr[1])] += 1 #int(yr[1]) is the language, c is the character. Increment by one.

x.close()
y.close()



t.readline()
#several data structures
totprobs = {}
best = {}
percentages = 5*[0]

#construct short utterance statistics
shortd = {}

x = io.open('train_set_x.csv', 'r', encoding='utf-8')
y = io.open('train_set_y.csv', 'r', encoding='utf-8')

x.readline()
y.readline()

for line in x:
  splitline = line.split(',')
  yr = y.readline().split(',')

  utt = splitline[1][:-1].replace(u" ", "").lower()
  if len(utt) < 20:
      utt = ''.join(sorted(utt)) #sorted utterance means character order is irrelevant
      if utt not in shortd.keys():
          shortd[utt] = 5*[0]
      shortd[utt][int(yr[1])] +=1 #int(yr[1]) is the language, c is the character. Increment by one.

x.close()
y.close()


#first and second classifications
for line in t:
  splitline = line.split(u',')
  lineid = splitline[0]

  testprobs = 5*[0]
  #first classification - bruteforce short examples
  if len(splitline[1][:-1]) < 20:
    shortstr = ''.join(sorted(splitline[1][:-1].replace(u' ','')))
    if shortstr in shortd.keys():
      percentages[shortd[shortstr].index(max(shortd[shortstr]))] += 1
      best[int(lineid)] = str(shortd[shortstr].index(max(shortd[shortstr])))
      continue
  #save latin character only examples for the svm
  if re.match('^[a-z0-9]*$', splitline[1][:-1].replace(u' ', '')):
    latinonly[int(lineid)] = splitline[1][:-1].replace(u' ', '')
    continue
  linechars = splitline[1][:-1].split(u' ')
  i = 0

  #calculate naive bayes for all 5 languages
  while i < 5:
    pyx = langprobs[i]

    for char in linechars:
      if char not in d.keys():
        continue

      pxy = (d[char][i]+1)/((sum(d[char])+2)*langprobs[i]) #apply laplace smoothing

    testprobs[i] = pyx
    i+=1
  best[int(lineid)] = str(testprobs.index(max(testprobs))) #store the max value
  percentages[testprobs.index(max(testprobs))] += 1
t.close()

alphanum = string.ascii_lowercase + string.digits
trainingx = []
trainingy = []


from random import randint

#construct our custom training set for the svm
x = io.open('train_set_x.csv', 'r', encoding='utf-8')
y = io.open('train_set_y.csv', 'r', encoding='utf-8')
x.readline()
y.readline()
dd = []

for line in x:
  splitline = line.split(',')
  yr = y.readline().split(',')
  dd.append((list(splitline[1].replace(u" ", '').lower())[:-1], yr[1]))


i = 0
while i < 200000:#rough size
  ind = randint(0,len(d)-1)
  j = 0
  line = dd[ind]#choose random line
  del(dd[ind])#no replacement
  s = ''
  s = s + str(i) + u','
  bad = False
  while j < 20 and len(line[0]) > 0:
    ind2 = randint(0, len(line[0])-1)
    if line[0][ind2] not in (string.ascii_lowercase + string.digits): #non latin character found
        bad = True
        break
    if j == 0:
        s = s + line[0][ind2] #append character to test example
    else:
        s = s + u' ' + line[0][ind2]
    del(line[0][ind2])
    j+=1
  if(bad):#non latin character found: dont save the example
      i+=1
      continue
  s =s +u'\n'
  trainingx.append(s) #append fully constructed example
  trainingy.append(str(i)+','+line[1])
  i+=1
x.close()
y.close()


#transform custom training examples into count vectors
fvecx = []
fvecy = []
alphanum = string.ascii_lowercase + string.digits
for itr in range(1,len(trainingx)):
    line = trainingx[itr]
    sline = line.split(',')
    chars = sline[1][:-1].split(' ')
    vx = 36*[0]
    for c in chars:
        vx[alphanum.index(c)] += 1 #increment character count by one for each appearance
    fvecx.append(vx[:]) #add to training vectors (x)
    fvecy.append(trainingy[itr].split(',')[1][:-1]) #add to training vectors (y)

#fit the vectors on the svm
clf = svm.SVC(cache_size=1000)
clf.fit(fvecx, fvecy)

#transform test examples into count vectors
latincounts = 5*[0]
for k in latinonly.keys():
  chars = latinonly[k]
  vx = 36*[0]
  for c in chars:
    vx[alphanum.index(c)] += 1
  #classify using predict
  a = clf.predict([vx])[0]
  best[k]  = a

#write classifications to output
result = open('testsetresults.csv', 'w')
result.write('Id,Category\n')
for lid in sorted(best.keys()):
  result.write(str(lid) + ',' + best[lid] + '\n')
result.close()
