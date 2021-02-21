
data = ['cora_ml','citeseer','pubmed','cora']

for i in data:
	file = open(i+'.txt')


	writeFile = open('clean_'+i+'.txt','w')
	for index,line in enumerate(file):
		line = line.strip('\n').split('\t')

		if index==0:
			toWrite = '\t'
			for j,word in enumerate(line):
				if j!=0 and j%1==0:
					toWrite += word.upper()+'\t'
			writeFile.write(toWrite+'\n')
		else:

			line = [l.upper() for l in line]
			writeFile.write('\t'.join(line)+'\n')