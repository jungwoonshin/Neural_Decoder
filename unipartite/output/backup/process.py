

data = ['cora_ml','citeseer','pubmed','cora']


for data_name in data:
	file = open(data_name+'.txt')
	write_file = open('clean_'+data_name+'.txt','w')

	# print(int(5/3))
	model_list = ['gae','gae-nd','vgae','vgae-nd','lgae','lgae-nd','sage','sage-nd','arga','arga-nd','arvga','arvga-nd','lrga','lrga-nd']

	for index, line in enumerate(file):
		print(line)
		if line == '\n' or line == '': break

		line = line.strip('\n').split(' ')
		auc = line[3].strip("'").strip(",\'")
		ap = line[4].strip("']")
		hit = line[2].split("'")[2].strip('')


		model_number = int(index/3)
		model_name = model_list[model_number]
		if index % 3 == 0:
			toWrite = model_name.upper() + '\t' + auc +'\t'+ap+'\t'
		toWrite += hit + '\t'
		# print(toWrite)
		# exit()
		if (index-2) % 3 == 0 and index > 1:
			write_file.write(toWrite+'\n')