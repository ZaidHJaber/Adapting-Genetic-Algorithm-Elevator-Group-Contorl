import pygad
import random
import numpy as np
parent1=np.array([1, 3, 4, 10, 5, 4, 12, 3, 8, 9, 5, 5, 5, 5, 7, 12])
parent2=np.array([3, 4, 10, 1, 5, 5, 5, 5, 7, 12, 5, 4, 12, 3, 8, 9])
sequence_start = random.randint(1, parent1.shape[0]-4)
sequence_end = random.randint(sequence_start, parent1.shape[0]-1)
print(sequence_start,"and",sequence_end)
print(parent1)
def pmx_crossover(parent1, parent2, sequence_start, sequence_end):
  # initialise a child
  child = np.zeros(parent1.shape[0])
  # get the genes for parent one that are passed on to child one
  parent1_to_child1_genes = parent1[sequence_start: sequence_end]
  print(parent1_to_child1_genes)
  # get the position of genes for each respective combination
  #parent1_to_child1 =  np.isin(parent1,parent1_to_child1_genes).nonzero()[0] #it give the index of non zero values 
  #print(parent1_to_child1) #we need to change this!
  for gene in range(sequence_start, sequence_end):
    child[gene] = parent1[gene]
  #it will give you a child with new genes choosen from the first parent 
  # gene of parent 2 not in the child
  print(child)
  genes_not_in_child_list = parent2.tolist()
  for i in parent1[sequence_start:sequence_end]:
    genes_not_in_child_list.remove(i)
  print(parent2)
  print(genes_not_in_child_list)
  genes_not_in_child = np.array(genes_not_in_child_list)
  if genes_not_in_child.shape[0] >= 1:
    for gene in genes_not_in_child:
      lookup = gene
      not_in_sequence = True
      while not_in_sequence:
        position_in_parent2 = np.where(parent2 == lookup)[0][0]
        parent2[position_in_parent2] = 0
        #print(position_in_parent2)

        if position_in_parent2 in range(sequence_start, sequence_end):
          lookup = parent1[position_in_parent2]

        else:
          child[position_in_parent2] = gene
          not_in_sequence = False

  return child

print(pmx_crossover(parent1, parent2, sequence_start, sequence_end))
print(parent2)