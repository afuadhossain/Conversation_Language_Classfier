import io

x = io.open('test_set_x.csv', 'r', encoding='utf-8')
counts = io.open('charcounts.csv', 'r', encoding='utf-8')
output = io.open('test_vector_probs.csv', 'w', encoding='utf-8')

counts.readline()
x.readline()
d = {}


for line in counts:
    d[line[0]] = 0

for key in d.keys():
    output.write(u',' + key)

output.write(u'\n')
counts.close()

for line in x:
    total = 0;
    splitline = line.split(',')

    utt = splitline[1].replace(u" ", "").lower()
    for c in utt:
        total += 1
        if c == u'\n':
            continue
        if c not in d.keys():
            continue
        d[c] += 1

    output.write(splitline[0])
    for key in d.keys():
        output.write(u',' +str(float(d[key])/total))
        d[key] = 0
    output.write(u'\n')



x.close()
output.close()
