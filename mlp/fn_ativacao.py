from numpy import e
def sigmoide(v):
    return 1/(1+e**-v)

def sigmoide_diff(v):
    return e**-v/(1+e**-v)**2