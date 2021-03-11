
import numpy as np

data = ['cora_ml','citeseer','pubmed','cora']
model_list = ['gae','gae-nd','vgae','vgae-nd','lgae','lgae-nd','sage','sage-nd','arga','arga-nd','arvga','arvga-nd','lrga','lrga-nd']

def getSign(value):
	if value < 0.0:
		return str(value)
	else:
		return '+'+str(value)

for data_name in data:
	# file = open(data_name+'.txt')
	file = open('clean_'+data_name+'.txt')
	write_file = open('improvement_'+data_name+'.txt','w')
	
	dict_score = {}
	for index, line in enumerate(file):
		lines = line.strip('\n').strip(' ') .split('\t')[1:]
		lines = [x for x in lines if x != '']
		score_list = []
		for line in lines:
			# print('line:',line)
			print(data_name)
			if '±' in line: score_std = line.split('±')
			else: score_std = line.split('+') 
			score = float(score_std[0])
			std = float(score_std[1])
			score_list.append(score)
			dict_score[model_list[index]] = score_list
	file.close()

	print(dict_score)

	improvement_dict = {}
	for i in range(0,len(model_list),2):
		regular = dict_score[model_list[i]]
		nd = dict_score[model_list[i+1]]

		improvement_percentage = []
		for j in range(len(nd)):
			difference = (nd[j]-regular[j])
			percentage = difference/regular[j]
			percentage = round(percentage*100.0, 2)
			difference = round(difference, 2)
			improvement_percentage.append((difference,percentage))

		improvement_dict[model_list[i+1]] = improvement_percentage

	print(improvement_dict)

	# average_diff = []
	# average_2d = np.array([[]])
	# average_2d = np.empty((0,10), float)
	average_2d = []

	file = open('clean_'+data_name+'.txt')
	for index, line in enumerate(file):
		if index % 2 == 1:
			continue
		write_file.write(line)

		diff_percent = improvement_dict[model_list[index+1]]
		lineToWrite = model_list[index+1][-3:].upper() + '\t'
		average = []
		for i in range(len(diff_percent)):
			diff_and_percent = diff_percent[i]
			diff = diff_and_percent[0]
			percent = diff_and_percent[1]
			lineToWrite += getSign(diff) + '(' + getSign(percent) + '%)' +'\t'
			average.append(diff)
			average.append(percent)

		average_2d.append(average)

		write_file.write(lineToWrite+'\n')
		print(lineToWrite)



	average_2d = np.array(average_2d)
	final_average = np.mean(average_2d,axis=0)
	final_average = np.around(final_average, decimals=2)

	print(final_average)

	lineToWrite = 'Average\t'
	for i in range(0,len(final_average),2):

		diff = final_average[i]
		percent = final_average[i+1]
		lineToWrite+= getSign(diff)+'('+getSign(percent)+'%)\t'
	write_file.write(lineToWrite+'\n')
