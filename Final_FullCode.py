#Metaheuristic Optimization, Course Project , Done by Zaid Jaber 20-Jan-2023
#Section[1] Variables and Libraries#################################################################################
#Import Needed Libraries 
import pygad
import random
import numpy as np
#Random Seed Control 
seedvalue =4 #4 is best
seedflag = False #True = Fixed Crossover Operator, False = Random Crossover and Mutarion Operator 
seedflag_PopSel = False
#Fitness Function Variables 
L = 4    #Building C Testing 
N= 10    #Number of floors above the ground floor 
LP = 7   #number of passengers in each elevator 
P =  L*LP # total number of passengers in all elevators (Number of Genes)
#v rated speed, 1.6 m/s, a rated acceleration, 1 m/s2, j rated jerk, 1 m/s3
#df floor height, 4.2m (equal floor heights) tf = df/v + v/a + a/j =5.225
#tdo door opening time, 2 s tdc door closing time, 3 s tpi passenger boarding time, 1.2 s tpo passenger alighting time, 1.2 s
a=1;j=1
df = 4.2; v =1.6; tdo =2; tdc = 3; tpi =1.2; tpo = 1.2;tf = df/v + v/a + a/j

#Testing Initial Solution (Used to Generate Random Population)
testsolution =[7, 3, 1, 4, 4, 10, 10, 6, 4, 8, 10, 8, 8, 5, 2, 4, 6, 9, 3, 1, 3, 3, 3, 5, 7, 7, 4, 8]
#Section[2] Fitness Function #################################################################################
def fitness_func(solution,solution_idx):
    Total_stops = 0
    Total_h =0
    S_list =[]
    H_list=[]
    for i in range(0,len(solution),LP):
        num_stops = len(set(solution[i:i+LP]))
        Total_stops = Total_stops + num_stops
        S_list.append(num_stops)

    for i in range(0,len(solution),LP):
        h = max(solution[i:i+LP])
        Total_h = Total_h+h
        H_list.append(h)
    #print(S_list)    
    #print(Total_stops) 
    average_s = Total_stops/L
    average_h = Total_h/L 
    #print(H_list)
    #print(average_h) 
     
    RTT = 2*average_h*(df/v) +(average_s+1)*(tf-(df/v)+tdo+tdc)+LP*(tpi+tpo)
    fitness = -1*RTT #  1/average_s or 1/average_h or 1/RTT
    return fitness
#Section[3] GA Configrations #################################################################################
fitness_function = fitness_func
num_generations = 100 # Number of generations.
num_parents_mating = 500 #(no change) Number of solutions to be selected as parents in the mating pool.
sol_per_pop = 701 # Number of solutions in the population.
parent_selection_type = "Tournament" #(Tour is better than rank is better than random) Type of parent selection.
keep_parents =20 #(no change) Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.
last_fitness = 0
population_list=[]
gene_space = testsolution
#Section[4] Crossover Operator #################################################################################
def pmx_crossover(parent1, parent2, sequence_start, sequence_end):
  # initialise a child
  child = np.zeros(parent1.shape[0])
  # get the genes for parent one that are passed on to child one
  parent1_to_child1_genes = parent1[sequence_start: sequence_end]
  # get the position of genes for each respective combination
  for gene in range(sequence_start, sequence_end):
    child[gene] = parent1[gene]
  #it will give you a child with new genes choosen from the first parent 
  # gene of parent 2 not in the child
  #print(child)
  genes_not_in_child_list = parent2.tolist()
  for i in parent1[sequence_start:sequence_end]:
    genes_not_in_child_list.remove(i)
  #print(parent2)
  #print(genes_not_in_child_list)
  genes_not_in_child = np.array(genes_not_in_child_list)
  if genes_not_in_child.shape[0] >= 1:
    for gene in genes_not_in_child:
      lookup = gene
      not_in_sequence = True
      while not_in_sequence:
        position_in_parent2 = np.where(parent2 == lookup)[0][0]
        parent2[position_in_parent2] = 0
        if position_in_parent2 in range(sequence_start, sequence_end):
          lookup = parent1[position_in_parent2]
        else:
          child[position_in_parent2] = gene
          not_in_sequence = False

  return child
###################################################################################################################

def crossover_func(parents, offspring_size, ga_instance):
  offspring = []
  idx = 0
  #print(offspring_size[0])
  while len(offspring) != offspring_size[0]:
    #print("stuck")
    #print("here",np.array(offspring))
    # locate the parents
    parent1 = parents[idx % parents.shape[0], :].copy()
    parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

    # find gene sequence in parent 1
    if seedflag == True:  
      random.seed(seedvalue)
    sequence_start = random.randint(1, parent1.shape[0]-1)
    if seedflag == True:  
      random.seed(seedvalue)
    sequence_end = random.randint(sequence_start, parent1.shape[0]-1)

    # perform crossover
    child1 = pmx_crossover(parent1, parent2, sequence_start, sequence_end)
     # locate the parents
    parent1 = parents[idx % parents.shape[0], :].copy()
    parent2 = parents[(idx + 1) % parents.shape[0], :].copy()   
    child2 = pmx_crossover(parent2, parent1, sequence_start, sequence_end)
    

    offspring.append(child1)
    offspring.append(child2)
    #print(offspring_size[0])
    #print(len(offspring))
    idx += 1
  return np.array(offspring)
  
#Section[5] Mutation Operator #################################################################################
def mutation_func(offspring, ga_instance):

  for chromosome_idx in range(offspring.shape[0]):
      # define a sequence of genes to reverse
      if seedflag == True:  
        random.seed(seedvalue)
      sequence_start = random.randint(1, offspring[chromosome_idx].shape[0] - 2) 
      if seedflag == True:  
        random.seed(seedvalue)
      sequence_end = random.randint(sequence_start, offspring[chromosome_idx].shape[0] - 1)
    
      genes = offspring[chromosome_idx, sequence_start:sequence_end]

         # start at the start of the sequence assigning the reverse sequence back to the chromosome
      index = 0
      if len(genes) > 0:
          for gene in range(sequence_start, sequence_end): 
            offspring[chromosome_idx, gene] = genes[index]
            index += 1

      return offspring
#Section[6] to print Generation number when it is done #################################################################################
def on_generation(ga):
    print("Generation", ga.generations_completed)
    #print(ga.population)
#Section[7] Create Initial Population #################################################################################
for i in range(sol_per_pop):
  if seedflag_PopSel == True:
    np.random.seed(i)
  nxm_random_num=list(np.random.permutation(gene_space)) 
  population_list.append(nxm_random_num) # add to the population_list
#print (population_list)
#Section[8] Apply Mutation #################################################################################
def on_crossover(ga_instance, offspring_crossover):
    # apply mutation to ensure uniqueness 
    offspring_mutation  = mutation_func(offspring_crossover, ga_instance)

    # save the new offspring set as the parents of the next generation
    ga_instance.last_generation_offspring_mutation = offspring_mutation

#Section[9] Run GA #################################################################################
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       initial_population=population_list,
                       gene_space=gene_space,
                       gene_type=int,
                       parent_selection_type=parent_selection_type,
                       on_generation=on_generation,
                       mutation_type= mutation_func,
                       crossover_type=crossover_func, 
                       keep_parents=keep_parents,
                       mutation_probability=0.1,
                       #save_solutions=True,
                       )
ga_instance.run()

#Section[10] Show Results ################################################################################
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f'Generation of best solution: {ga_instance.best_solution_generation}')
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
#ga_instance.plot_fitness()
