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
def is_factible(child):
	sum_of_pis = sum(child)
	return sum_of_pis >= PD - 0.5 and sum_of_pis <= PD + 0.5

# objective function
def objective(v, parameters, bounds):

	if(is_factible(v)):
		solution = 0
		for i in range(n_variables):
			solution += parameters[i][0]*v[i]**2 + parameters[i][1]*v[i] + parameters[i][2] + abs(parameters[i][3] * math.sin(parameters[i][4] * (bounds[i][0] - v[i])))
		return solution
	else:
		return INFACTIBLE

# generate one subject - used in inicial solution
def generate_subject(bounds, n_variables):
	subject = []

	for i in range (n_variables):
		value = np.random.uniform(bounds[i][0], bounds[i][1], 1)[0]
		subject.append(value)

	return subject

# generate first generation
def generate_first_gen(bounds, n_pop, n_variables):
	pop = []    

	for i in range (n_pop):
		subject = generate_subject(bounds, n_variables)
		pop.append(subject)

	return pop

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
		# select crossover point
		pt = randint(1, len(p1))
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator - uniform mutation
def mutation(child, bounds, n_variables, r_mut):
	# applying the perturbation
	if rand() < r_mut:
		random_value = np.random.uniform(-1.0, 1.0, n_variables) # random value between -1 and 1
		for i in range (n_variables):
			if (bounds[i][0] <= child[i] + random_value[i] <= bounds[i][1]):	# determines if (new value + solution) is within bounds
				child[i] += random_value[i]
	return

#generate child for next generation
def generate_next_gen(selected, bounds, n_variables, r_cross, r_mut):
	children = list()
	for i in range(0, n_pop, 2):
		# get selected parents in pairs
		p1, p2 = selected[i], selected[i+1]
		# crossover and mutation
		for c in crossover(p1, p2, r_cross):
			# mutation
			mutation(c, bounds, n_variables, r_mut)
			# store for next generation
			children.append(c)
	# replace population
	return children

# genetic algorithm
def genetic_algorithm(bounds, parameters, n_iter, n_pop, n_variables, r_cross, r_mut):
	# initial population - random
	pop = generate_first_gen(bounds, n_pop, n_variables)
	# keep track of best solution
	best, best_eval = 0, objective(pop[0], parameters, bounds)
	# enumerate generations
	for gen in range(n_iter):
		# evaluate all candidates in the population
		scores = [objective(d, parameters, bounds) for d in pop]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		pop = generate_next_gen(selected, bounds, n_variables, r_cross, r_mut)
	return [best, best_eval]
 
# define range for input
bounds = asarray([[0, 680], [0, 360], [0, 360], [60, 180], [60, 180], [60, 180], [60, 180], [60, 180], [60, 180], [40, 120], [40, 120], [55, 120], [55, 120]])
# define parameters
constants = asarray([[0.00028, 8.1, 550, 300, 0.035], [0.00056, 8.1, 309, 200, 0.042], [0.00056, 8.1, 307, 150, 0.042], [0.00324, 7.74, 240, 150, 0.063], [0.00324, 7.74, 240, 150, 0.063], [0.00324, 7.74, 240, 150, 0.063], [0.00324, 7.74, 240, 150, 0.063], [0.00324, 7.74, 240, 150, 0.063], [0.00324, 7.74, 240, 150, 0.063], [0.00284, 8.6, 126, 100, 0.084], [0.00284, 8.6, 126, 100, 0.084], [0.00284, 8.6, 126, 100, 0.084], [0.00284, 8.6, 126, 100, 0.084]])
# define the total iterations
n_iter = 1000
# define the population size
n_pop = 500
# define the number of decision variables
n_variables = 13
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 0.01

scores = []

for i in range(0, 30):
	# perform the genetic algorithm search
	best, score = genetic_algorithm(bounds, constants, n_iter, n_pop, n_variables, r_cross, r_mut)
	print('Done!')
	print('f(%s) = %f' % (best, score))
	scores.append(score)

print(f"Mínimo: {min(scores)}")
print(f"Máximo: {max(scores)}")
print(f"Média: {np.mean(scores)}")
print(f"Desvio Padrão: {np.std(scores)}")

plt.boxplot(scores)
plt.ylim(17000, 20000)
plt.show()

# Mínimo: 18408.760842238204
# Máximo: 18983.640533193633
# Média: 18684.567863340817
# Desvio Padrão: 137.87866251767713

# valores para execução c menor valor (18408.760)
# P1: 362.28774622104675
# P2: 299.1579411012902
# P3: 225.07221491913225
# P4: 158.96394751477243
# P5: 71.10888477737109
# P6: 111.71745919949923
# P7: 96.90443017853309
# P8: 104.20999468658191
# P9: 159.18036802805153
# P10: 60.135618271461716
# P11: 40.62851339015322
# P12: 55.24600986500449
# P13: 55.10603160041962