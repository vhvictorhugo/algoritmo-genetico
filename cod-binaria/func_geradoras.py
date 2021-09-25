# Referência e detalhamento do código: https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/

# genetic algorithm search for continuous function optimization
import math
import numpy as np
import matplotlib.pyplot as plt

from numpy import asarray
from numpy.random import randint
from numpy.random import rand

INFACTIBLE = 9999999
PD = 1800

#check if solution is factible
def is_factible(pi_l):
	sum_of_pis = sum(pi_l)
	return sum_of_pis >= PD - 0.5 and sum_of_pis <= PD + 0.5

# objective function
def objective(vars):
	pi_l = []; pimin_l = []; a_l = []; b_l = []; c_l = []; e_l = []; f_l = []

	while(len(vars) > 0):
		pi, pimin, a, b, c, e, f, *others = vars
		pi_l.append(pi)
		pimin_l.append(pimin)
		a_l.append(a)
		b_l.append(b)
		c_l.append(c)
		e_l.append(e)
		f_l.append(f)
		vars = others

	if(is_factible(pi_l)):
		i = 0
		solution = 0
		for i in range(0, len(pi_l)):
			solution += a_l[i]*pi_l[i]**2 + b_l[i]*pi_l[i] + c_l[i] + abs(e_l[i] * math.sin(f_l[i] * (pimin_l[i] - pi_l[i])))
		return solution
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

table = [
  {'pimin': 0, 'pimax': 680, 'a': 0.00028, 'b': 8.10, 'c': 550, 'e': 300, 'f': 0.035},
	{'pimin': 0, 'pimax': 260, 'a': 0.00056, 'b': 8.10, 'c': 309, 'e': 200, 'f': 0.042},
	{'pimin': 0, 'pimax': 360, 'a': 0.00056, 'b': 8.10, 'c': 307, 'e': 150, 'f': 0.042},
	{'pimin': 60, 'pimax': 180, 'a': 0.00324, 'b': 7.74, 'c': 240, 'e': 150, 'f': 0.063},
	{'pimin': 60, 'pimax': 180, 'a': 0.00324, 'b': 7.74, 'c': 240, 'e': 150, 'f': 0.063},
	{'pimin': 60, 'pimax': 180, 'a': 0.00324, 'b': 7.74, 'c': 240, 'e': 150, 'f': 0.063},
	{'pimin': 60, 'pimax': 180, 'a': 0.00324, 'b': 7.74, 'c': 240, 'e': 150, 'f': 0.063},
	{'pimin': 60, 'pimax': 180, 'a': 0.00324, 'b': 7.74, 'c': 240, 'e': 150, 'f': 0.063},
	{'pimin': 60, 'pimax': 180, 'a': 0.00324, 'b': 7.74, 'c': 240, 'e': 150, 'f': 0.063},
	{'pimin': 40, 'pimax': 120, 'a': 0.00284, 'b': 8.60, 'c': 126, 'e': 100, 'f': 0.084},
	{'pimin': 40, 'pimax': 120, 'a': 0.00284, 'b': 8.60, 'c': 126, 'e': 100, 'f': 0.084},
	{'pimin': 55, 'pimax': 120, 'a': 0.00284, 'b': 8.60, 'c': 126, 'e': 100, 'f': 0.084},
	{'pimin': 55, 'pimax': 120, 'a': 0.00284, 'b': 8.60, 'c': 126, 'e': 100, 'f': 0.084},
]

scores = []
bounds = []

for line in table:
  # define range for input
	bounds.append([line['pimin'], line['pimax']]) #pi
	bounds.append([line['pimin'], line['pimin']])
	bounds.append([line['a'], line['a']])
	bounds.append([line['b'], line['b']])
	bounds.append([line['c'], line['c']])
	bounds.append([line['e'], line['e']])
	bounds.append([line['f'], line['f']])

bounds = asarray(bounds)

# define the total iterations
n_iter = 200
# bits per variable
n_bits = 16
# define the population size
n_pop = 100
# crossover rate
r_cross = 1.5
# mutation rate
r_mut = 0.6 / (float(n_bits) * len(bounds))

for i in range(0, 30):
	# perform the genetic algorithm search
	best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
	print('Done!')

	if(best == 0):
		print('não deu pra achar solução... :(')
	else:
		decoded = decode(bounds, n_bits, best)
		print('f(%s) = %f' % (decoded, score))
		scores.append(score)


if len(scores):
	print(f"Mínimo: {min(scores)}")
	print(f"Máximo: {max(scores)}")
	print(f"Média: {np.mean(scores)}")
	print(f"Desvio Padrão: {np.std(scores)}")

	plt.boxplot(scores)
	plt.show()

# Mínimo: 18854.72280032163
# Máximo: 19369.428938369663
# Média: 19099.894751442465
# Desvio Padrão: 155.03225565450362