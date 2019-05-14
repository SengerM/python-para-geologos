import numpy as np
from scipy.stats import norm

CONFIDENCE_LEVEL = .95

def freqconfint_gaussian(x, sigma=1, clevel=.68, step=.01):
	"""
	This function calculates the frequentist symmetric interval for the 
	mean of a sample <x> of gaussian variables, assumed all with the same
	mean. Different values for each sigma_i is admited if passed as a numpy
	array in the argument <sigma>.
	"""
	x_obs = (x/sigma**2).sum() / (1/sigma**2).sum()
	just_a_number = (sigma**-2).sum()
	x_max = 0
	while norm.cdf(just_a_number*x_obs, loc=x_max*just_a_number, scale=just_a_number**.5) > (1-clevel)/2:
		x_max += step
	x_min = 0
	while norm.cdf(just_a_number*x_obs, loc=x_min*just_a_number, scale=just_a_number**.5) > 1-(1-clevel)/2:
		x_min += step
	return x_min, x_max

data = np.genfromtxt(input('Tell me the name of the file with the data, please...\n--> '), delimiter=',', skip_header = 0) # Read the data.
data = data.transpose() # Accomodate the data.
q_i = data[0] # Get data.
sigma_i = data[1] # Get data.
q_min, q_max = freqconfint_gaussian(q_i, sigma = sigma_i, clevel = CONFIDENCE_LEVEL) # Calculation of the confidence interval.
q_obs = (q_i/sigma_i**2).sum() / (1/sigma_i**2).sum() # Calculation of the MLE estimator.
MSWD = 1/(len(q_i)-1)*( (q_i - q_i.mean())**2/sigma_i**2 ).sum()

print('q_obs = ' + str(q_obs) + ' is the MLE')
print('[q_min, q_max] = [' + str(q_min) + ', ' + str(q_max) + '] is the ' + str(CONFIDENCE_LEVEL*100) + ' % confidence level interval frequentist interval')
print('The previous frequentist interval can be equivalently written as ' + str((q_max+q_min)/2) + ' +- ' + str((q_max-q_min)/2))
print('MSWD = ' + str(MSWD))
