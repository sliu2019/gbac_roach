import numpy as np

def moving_distance(unit1, unit2):
  phi = (unit2-unit1) % (2*np.pi)
  sign = -1
  # used to calculate sign
  if not ((phi >= 0 and phi <= np.pi) or (
          phi <= -np.pi and phi >= -2*np.pi)):
      sign = 1
  if phi > np.pi:
      result = 2*np.pi-phi
  else:
      result = phi
  return result*sign