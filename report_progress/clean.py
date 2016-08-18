import os, re
"""
SOURCE:
https://github.com/Chao1155/python-script-utilities/blob/master/MacBook/clean_your_bib_file.py
"""
def clean(fileName):
	remove_keys = ['month','abstract', 'mendeley-groups', 'keywords', 'issn']
	blind_keys = ['url']
	bibf = open(fileName, 'r')
	cleanf = open(fileName + '.tmp', 'w')
	lines = bibf.readlines()
	for line in lines:
		if re.search('^'+'|'.join(blind_keys),line):
			cleanf.write('%%%%' + line)
		elif not re.search('^'+'|'.join(remove_keys),line):
			#print line
			cleanf.write(line)
	cleanf.close()
	os.remove(fileName)
	os.rename(fileName + '.tmp', fileName)

clean('bib/autofaces.bib')
