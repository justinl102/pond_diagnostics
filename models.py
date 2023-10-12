def exponential_fit_3d(x, a,b,c,d):
  return a*x**3 + b*x**2 + c*x + d

def exponential_fit_2d(x, b,c,d):
  return b*x**2 + c*x + d

def exponential_decay(N0, k, t):
    return N0 * math.exp(-k * t)


