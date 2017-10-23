import io

x = io.open('train_set_x.csv', 'r', encoding='utf-8')
y = io.open('train_set_y.csv', 'r', encoding='utf-8')
output = io.open('charcounts.csv', 'w', encoding='utf-8')


x.readline()
y.readline()
d = {}

for line in x:
  splitline = line.split(',')
  yr = y.readline().split(',')
  if(yr[0] != splitline[0]):
    print('mismatch')
    x.close()
    y.close()
    exit()

  utt = splitline[1].replace(u" ", "").lower()
  for c in utt:
    if c == u'\n':
      continue
    if c not in d.keys():
      d[c] = 5*[0]
    d[c][int(yr[1])] += 1

x.close()
y.close()

output.write(u'chars,slovak,french,spanish,german,polish\n')

for key in d.keys():
  output.write(key)
  for val in d[key]:
    output.write(u',' + str(val))
  output.write(u'\n')
output.close()
