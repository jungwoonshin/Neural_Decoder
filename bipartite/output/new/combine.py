data = ['gpcr','enzyme','ionchannel','malaria','drug','sw','nanet','movie100k']
data2 = ['gpcr2','enzyme2','ionchannel2','malaria2','drug2','sw2','nanet2','movie100k2']

for i in range(len(data)):
	data[i] = data[i] +'.txt'
	data2[i] = data2[i] +'.txt'


for i in range(len(data)):
	file1 = open(data[i])
	file2 = open(data2[i])
	out_file = open('comb/'+data[i].split('.')[0].upper()+'.txt','w')
	for line in file1:
		out_file.write(line)
	for line in file2:
		out_file.write(line)
	out_file.close()


