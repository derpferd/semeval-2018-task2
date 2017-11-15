
fname = "data/slim_train/us_train.labels";

with open(fname) as f:
  content = f.readlines()

#print(content)

'''count=0
for x in content:
  print(x)
  count = count + 1
  if count == 20:
    break'''

for index,y in enumerate(content):
  if y == '3\n' or y == '8\n' or y == '13\n' or y == '1\n' or y == '9\n':
    content[index] = '0\n'
  if y == '6\n':
    content[index] = '5\n'
  if y == '18\n':
    content[index] = '10\n'
  if y == '14\n':
    content[index] = '1\n'
  if y == '15\n':
    content[index] = '3\n'
  if y == '16\n':
    content[index] = '8\n'
  if y == '17\n':
    content[index] = '9\n'
  if y == '19\n':
    content[index] = '6\n'

print('##############################################')

'''count=0
for z in content:
  print(z)
  count = count + 1
  if count == 20:
    break'''


newfname = "data/slim_train/us_slim_train.labels";
thefile = open(newfname,'w')
for item in content:
  thefile.write(item)
