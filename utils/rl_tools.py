import scipy.signal as signal

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def good_discount(x, gamma):
    return discount(x, gamma)
