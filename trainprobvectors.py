import io

x = io.open('train_set_x.csv', 'r', encoding='utf-8')
y = io.open('train_set_y.csv', 'r', encoding='utf-8')
counts = io.open('charcounts.csv', 'r', encoding='utf-8')
output = io.open('KNN_probs.csv', 'w', encoding='utf-8')

counts.readline()
x.readline()
y.readline()
d = {}

for line in counts:
    d[line[0]] = 0

for key in d.keys():
    output.write(u',' + key)

output.write(u',' + 'class\n')
counts.close()

for line in x:
    total = 0;
    splitline = line.split(',')
    yr = y.readline().split(',')
    if(yr[0] != splitline[0]):
        print('mismatch')
        x.close()
        y.close()
        exit()

    utt = splitline[1].replace(u" ", "").lower()
    for c in utt:
        total += 1
        if c == u'\n':
          continue
        d[c] += 1

    output.write(yr[0])
    for key in d.keys():
        output.write(u',' +str(float(d[key])/total))
        d[key] = 0

    output.write(u',' + yr[1] + '\n')


x.close()
y.close()
output.close()
