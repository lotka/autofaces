import os, re,sys, sys

def clean(fileName):
    remove_keys = ['abstract', 'isbn', 'mendeley-groups', 'keywords', 'file', 'issn','month']
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

for f in sys.argv[1:]:
    if os.path.isfile(f) and f[-4:] == '.bib':
        clean(f)
        print 'Cleaned ', f
    else:
        print 'Warning: Can\'t process file ',f,' as it is not a .bib file'
