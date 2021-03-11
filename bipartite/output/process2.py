
from collections import defaultdict
import numpy as np

def getSign(value):
	if value < 0.0:
		return str(value)
	else:
		return '+'+str(value)

file = open('ap.txt')
next(file)
writeFile = open('ap2.txt','w')
writeFile.write('AP	GPCR	Enzyme	Ionchannel	Malaria	Drug	SW	Nanet	Movie100k\n')
    
average_2d = []
while True:
	line1 = orig_line = file.readline()
	line2 = file.readline()
	if not line2:
		break
	
	line1 = line1.strip('\n').strip(' ').split('\t')
	line2 = line2.strip('\n').strip(' ').split('\t')

	toWrite = '-ND\t'
	average = []
	for index in range(1,len(line1)):
		if line1[index] == '': continue
		score1 = float(line1[index].split('+')[0])
		score2 = float(line2[index].split('+')[0])

		diff = score2 - score1
		diff = round(diff,2)
		percentage = diff/score1
		percentage = round(percentage*100.0,2)
		toWrite += getSign(diff) +'('+getSign(percentage)+'%)\t'
		average.append(diff)
		average.append(percentage)
	average_2d.append(average)
	writeFile.write(orig_line)
	writeFile.write(toWrite+'\n')

average_2d = np.mean(average_2d, axis=0)
average_2d = np.around(average_2d, decimals=2)
lineToWrite = 'Average\t'
for i in range(0,len(average_2d),2):

	diff = average_2d[i]
	percent = average_2d[i+1]
	lineToWrite+= getSign(diff)+'('+getSign(percent)+'%)\t'
writeFile.write(lineToWrite+'\n')
