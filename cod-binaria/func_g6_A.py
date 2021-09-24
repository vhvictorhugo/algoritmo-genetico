# Referência e detalhamento do código: https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/

# genetic algorithm search for continuous function optimization
import numpy as np
import matplotlib.pyplot as plt

from numpy import asarray
from numpy.random import randint
from numpy.random import rand

INFACTIBLE = 9999999

#check if solution is factible
def is_factible(child):
	x1, x2 = child
	g1 = -(x1 - 5) ** 2 - (x2 - 5) ** 2 + 100 <= 0
	g2 = -(x1 - 6) ** 2 + (x2 - 5) ** 2 - 82.81 <= 0
	return g1 and g2

# objective function
def objective(v):
	if(is_factible(v)):
		x1, x2 = v
		return (x1 - 10) ** 3 + (x2 - 20) ** 3
	else:
		return INFACTIBLE
 
# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# store
		decoded.append(value)
	return decoded

# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]
 
# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]
 
# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

#generate child for next generation
def generate_next_gen(selected, r_cross, r_mut):
	children = list()
	for i in range(0, n_pop, 2):
		# get selected parents in pairs
		p1, p2 = selected[i], selected[i+1]
		# crossover and mutation
		for c in crossover(p1, p2, r_cross):
			# mutation
			mutation(c, r_mut)
			# store for next generation
			children.append(c)
	# replace population
	return children

# genetic algorithm
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
	# enumerate generations
	for gen in range(n_iter):
		# decode population
		decoded = [decode(bounds, n_bits, p) for p in pop]
		# evaluate all candidates in the population
		scores = [objective(d) for d in decoded]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				# print(">%d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		pop = generate_next_gen(selected, r_cross, r_mut)
	return [best, best_eval]
 
# define range for input
bounds = asarray([[13, 100], [0, 100]])
# define the total iterations
n_iter = 100
# bits per variable
n_bits = 16
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))


scores = []

for i in range(0, 30):
	# perform the genetic algorithm search
	best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
	print('Done!')
	decoded = decode(bounds, n_bits, best)
	print('f(%s) = %f' % (decoded, score))
	scores.append(score)

print(f"Mínimo: {min(scores)}")
print(f"Máximo: {max(scores)}")
print(f"Média: {np.mean(scores)}")
print(f"Desvio Padrão: {np.std(scores)}")

plt.boxplot(scores)
plt.show()

# Mínimo: -7950.713960289385
# Máximo: -7950.1766629219055
# Média: -7950.194572834154
# Desvio Padrão: 0.09644782914386364