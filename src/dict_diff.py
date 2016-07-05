A = { 'key1' : { 'subkey2' : 4, 'subkey2' : 5}, 'key2' : -500 }
B = { 'key1' : { 'subkey2' : 4, 'subkey2' : 5}, 'key2' : 400 , 'key3' : 400}



print A == B

def diff(A,B):
    dictType = type({})
    res = {}
    for key in A:
        if A[key] == dictType and key in B:
            res += diff(A[key],B[key])

