import os
print os.environ['PYTHONHASHSEED']
print tyoe(os.environ['PYTHONHASHSEED'])

class testClass(object):
    def __init__(self,x):
        self.x = x

    def test(self):
        print hash(self)

    def hash(self):
        print hash(self.x)

def ahh():
    print 'lolz'

x = testClass(4)
print x.hash()
