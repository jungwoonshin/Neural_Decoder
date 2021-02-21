
from collections import defaultdict
# data = ['cora_ml','citeseer','pubmed']

data = ['gpcr','enzyme','ionchannel','malaria','drug','sw','nanet','movie100k']
model_list = ['gae','gae-nd','vgae','vgae-nd','lgae','lgae-nd','sage','sage-nd','arga','arga-nd','arvga','arvga-nd','lrga','lrga-nd']

auc_dict = defaultdict(list)
ap_dict = defaultdict(list)
for data_name in data:
	file = open(data_name.upper()+'.txt')
	for index, line in enumerate(file):
		print(':'+line)
		if line == '\n' or line == '': break
		line = line.strip('\n').split("'")
		auc = line[2]
		ap = line[4]

		model_name = model_list[index]
		auc_dict[model_name].append(auc)
		ap_dict[model_name].append(ap)
print(auc_dict)
print(ap_dict)

data = [i.capitalize() for i in data]
writeFile = open('auc.txt','w')
writeFile.write('\t'+ '\t'.join(data) +'\n')
writeFile2 = open('ap.txt','w')
writeFile2.write('\t'+ '\t'.join(data) +'\n')
for model_name in model_list:
	toWrite = model_name.upper() +'\t'
	for i in auc_dict[model_name]:
		toWrite += i +'\t'
	writeFile.write(toWrite+'\n')

	toWrite = model_name.upper() +'\t'
	for i in ap_dict[model_name]:
		toWrite += i +'\t'
	writeFile2.write(toWrite+'\n')


