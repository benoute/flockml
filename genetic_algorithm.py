
# Code adapted from Stephen Marsland (Machine Learning: An Algorithmic Perspective)
# Benoit Fontaine


# The Genetic algorithm with support for parallel execution using IPython

import pylab as pl
import numpy as np
from IPython.parallel import interactive

@interactive
def do_fitness(fitnessFunction, population):
    return fitnessFunction(population)

class GeneticAlgorithm:

    def __init__(self, stringLength, fitnessFunction, nEpochs, populationSize=100,
                mutationProb=-1, crossover='un', nElite=4, tournament=True,
                initialPop=[]):
        """ Constructor"""
        self.stringLength = stringLength
        self.lb_view= None

        # Population size should be even
        if np.mod(populationSize,2)==0:
            self.populationSize = populationSize
        else:
            self.populationSize = populationSize+1

        if mutationProb < 0:
             self.mutationProb = 1/stringLength
        else:
             self.mutationProb = mutationProb

        self.nEpochs = nEpochs
        self.curEpochs = 0
        self.stop = False

        self.fitnessFunction = fitnessFunction

        self.crossover = crossover
        self.nElite = nElite
        self.tournment = tournament

        if np.shape(initialPop)[0]:
            self.population = initialPop
        else:
            self.population = np.random.rand(self.populationSize, self.stringLength)
            self.population = np.where(self.population<0.5,0,1)

        self.bestfit = np.zeros(self.nEpochs)
        self.best = []
        self.generation_fit = np.zeros(self.nEpochs)
        self.generation_num_feature = np.zeros(self.nEpochs)

    def set_lb_view(self, lb_view):
        self.lb_view = lb_view

    def runGA(self):
        from thread import start_new_thread
        start_new_thread(self.__runGA,())

    def report(self, start=0):
        best_num_feature       = self.best[self.curEpochs-1].shape[0]
        best_fitness           = self.bestfit[self.curEpochs-1]
        progress               = int(100 * float(self.curEpochs)/self.nEpochs)
        generation_fitness     = self.generation_fit[self.curEpochs-1]
        generation_num_feature = self.generation_num_feature[self.curEpochs-1]

        print "%i generation / %i (%i %%): %f (%i features)/ %f (avg. %i features)" \
                        % (self.curEpochs,
                        self.nEpochs, progress, best_fitness, best_num_feature,
                        generation_fitness, generation_num_feature)
        pl.plot(self.bestfit[start:self.curEpochs],'kx-', c='g', label='Best fitness')
        pl.plot(self.generation_fit[start:self.curEpochs],'kx-', c='b',
            label='Avg generation fitness')
        pl.xlabel('Generation')
        pl.ylabel('Fitness')
        pl.legend(loc='best')
        pl.show()

        pl.plot(self.generation_num_feature[start:self.curEpochs], 'kx-',
                    label='Avg number of features per generation')
        pl.xlabel('Generation')
        pl.ylabel('Number of features')

        pl.show()

    def abort(self):
        print "Sending abort signal..."
        self.stop = True
        if self.lb_view:
            self.lb_view.abort()


    def add_more(self, nEpochs):
        self.bestfit = np.concatenate((self.bestfit, np.zeros(nEpochs)))
        self.generation_fit = np.concatenate((self.generation_fit, np.zeros(nEpochs)))
        self.generation_num_feature = np.concatenate((self.generation_num_feature, np.zeros(nEpochs)))
        self.nEpochs += nEpochs


    def run_more(self, nEpochs, mutationProb=-1, crossover='', nElite=-1, tournament=-1):
        if mutationProb > 0:
            self.mutationProb = mutationProb
        if crossover:
            self.crossover = crossover
        if nElite >= 0:
            self.nElite = nElite
        if tournament != -1:
            self.tournment = tournament

        self.add_more(nEpochs)
        self.runGA()

    def __fitness(self, population):
        fitness = np.zeros(population.shape[0])
        if self.lb_view:
            tasks = []
            for individual in population:
                task = self.lb_view.apply(do_fitness, self.fitnessFunction,
                                            individual)
                tasks.append(task)
            for i,task in enumerate(tasks):
                self.lb_view.wait(task)
                fitness[i] = task.get()
        else:
            for i,individual in enumerate(population):
                fitness[i] = self.fitnessFunction(individual)

        return fitness

    def __runGA(self):
        """The basic loop"""
        while self.curEpochs < self.nEpochs:
        #for i in range(self.nEpochs):
            if self.stop:
                print "Aborting..."
                break

            i = self.curEpochs

            # First time only: compute fitness of the population if none exists
            if self.curEpochs == 0:
                #print "Compute population fitness ..."
                self.fitness = self.__fitness(self.population)

            # Record statistics
            best_index = self.fitness.argmax()
            self.bestfit[i] = self.fitness[best_index]
            self.best.append(np.where(self.population[best_index, :])[0])
            self.generation_fit[i] = np.mean(self.fitness)
            self.generation_num_feature[i] = np.mean([np.where(individual == 1)[0].shape[0]
                                                        for individual in self.population])

            # Pick parents -- can do in order since they are randomised
            #print "Fitness Proportional Selection ..."
            newPopulation = self.fps(self.population,self.fitness)

            # Apply the genetic operators
            #print "Crossover ..."
            if self.crossover == 'sp':
                newPopulation = self.spCrossover(newPopulation)
            elif self.crossover == 'un':
                newPopulation = self.uniformCrossover(newPopulation)
            elif self.crossover == 'ssocf':
                newPopulation = self.ssocfCrossover(newPopulation)
            newPopulation = self.mutate(newPopulation)

            # Compute fitness of the new population
            newFitness = self.__fitness(newPopulation)

            # Apply elitism and tournaments if using
            if self.nElite>0:
                #print "Elitism ..."
                newPopulation, newFitness = self.elitism(self.population,
                                                            newPopulation,
                                                            self.fitness,
                                                            newFitness)
            if self.tournment:
                #print "Tournament ..."
                newPopulation, newFitness = self.tournament(self.population,
                                                                newPopulation,
                                                                self.fitness,
                                                                newFitness)
            # Switch population
            self.population = newPopulation
            self.fitness = newFitness

            self.curEpochs+=1


    def fps(self,population,fitness):
        """ Fitness Proportional Selection """

        # Scale fitness by total fitness
        fitness = fitness/sum(fitness)
        fitness = 10*fitness/fitness.max()

        # Put repeated copies of each string in according to fitness
        # Deal with strings with very low fitness
        j=0
        while round(fitness[j])<1:
            j = j+1

        newPopulation_inds = [j] * int(round(fitness[j]))

        # Add multiple copies of strings into the newPopulation
        for i in range(j+1,self.populationSize):
            if round(fitness[i])>=1:
                newPopulation_inds.extend([i] * int(round(fitness[i])))

        np.random.shuffle(newPopulation_inds)
        newPopulation = population[newPopulation_inds[:self.populationSize],:]

        return newPopulation

    def spCrossover(self,population):
        # Single point crossover
        newPopulation = np.zeros(np.shape(population))
        crossoverPoint = np.random.randint(0,self.stringLength,self.populationSize)
        for i in range(0,self.populationSize,2):
            newPopulation[i,:crossoverPoint[i]] = population[i,:crossoverPoint[i]]
            newPopulation[i+1,:crossoverPoint[i]] = population[i+1,:crossoverPoint[i]]
            newPopulation[i,crossoverPoint[i]:] = population[i+1,crossoverPoint[i]:]
            newPopulation[i+1,crossoverPoint[i]:] = population[i,crossoverPoint[i]:]
        return newPopulation

    def uniformCrossover(self,population):
        # Uniform crossover
        newPopulation = np.zeros(np.shape(population))
        which = np.random.rand(self.populationSize,self.stringLength)
        which1 = which>=0.5
        for i in range(0,self.populationSize,2):
            newPopulation[i,:] = population[i,:]*which1[i,:] + population[i+1,:]*(1-which1[i,:])
            newPopulation[i+1,:] = population[i,:]*(1-which1[i,:]) + population[i+1,:]*which1[i,:]
        return newPopulation

    def ssocfCrossover(self, population):
        newPopulation = np.zeros(np.shape(population))

        for i in range(0,self.populationSize,2):
            newPopulation[i,:] = np.zeros(self.stringLength)
            newPopulation[i+1,:] = np.zeros(self.stringLength)

            common_inds = np.intersect1d(np.where(population[i,:] == population[i+1,:])[0],
                                            np.where(population[i,:] == 1)[0])
            non_common_inds = np.union1d(np.where(population[i,:] != population[i+1,:])[0],
                                            np.where(population[i,:] == 0)[0])
            newPopulation[i,common_inds] = population[i,common_inds]
            newPopulation[i+1,common_inds] = population[i+1,common_inds]

            n1 = np.where(population[i,:] == 1)[0].shape[0]
            n2 = np.where(population[i+1,:] == 1)[0].shape[0]
            nc = common_inds.shape[0]
            nu = self.stringLength - nc

            p1 = float(n1-nc)/nu
            p2 = float(n2-nc)/nu

            newPopulation[i,non_common_inds] = np.random.binomial(1, p1, non_common_inds.shape[0])
            newPopulation[i+1,non_common_inds] = np.random.binomial(1, p2, non_common_inds.shape[0])

        return newPopulation

    def mutate(self,population):
        # Mutation
        whereMutate = np.random.rand(np.shape(population)[0],np.shape(population)[1])
        population[np.where(whereMutate < self.mutationProb)] = 1 - population[np.where(whereMutate < self.mutationProb)]
        return population

    def elitism(self, oldPopulation, population, oldFitness, fitness):
        best_inds = np.argsort(oldFitness)
        best = np.squeeze(oldPopulation[best_inds[-self.nElite:],:])
        indices = range(np.shape(population)[0])
        np.random.shuffle(indices)
        population = population[indices,:]
        fitness = fitness[indices]
        population[0:self.nElite,:] = best
        fitness[0:self.nElite] = oldFitness[best_inds[-self.nElite:]]

        return population, fitness

    def tournament(self,oldPopulation,population,oldFitness,fitness):
        for i in range(0,np.shape(population)[0],2):
            f = np.concatenate((oldFitness[i:i+2],fitness[i:i+2]))
            indices = np.argsort(f)
            if indices[-1]<2 and indices[-2]<2:
                population[i,:] = oldPopulation[i,:]
                population[i+1,:] = oldPopulation[i+1,:]
                fitness[i] = oldFitness[i]
                fitness[i+1] = oldFitness[i+1]
            elif indices[-1]<2:
                if indices[0]>=2:
                    population[i+indices[0]-2,:] = oldPopulation[i+indices[-1]]
                    fitness[i+indices[0]-2] = oldFitness[i+indices[-1]]
                else:
                    population[i+indices[1]-2,:] = oldPopulation[i+indices[-1]]
                    fitness[i+indices[1]-2] = oldFitness[i+indices[-1]]
            elif indices[-2]<2:
                if indices[0]>=2:
                    population[i+indices[0]-2,:] = oldPopulation[i+indices[-2]]
                    fitness[i+indices[0]-2] = oldFitness[i+indices[-2]]
                else:
                    population[i+indices[1]-2,:] = oldPopulation[i+indices[-2]]
                    fitness[i+indices[1]-2] = oldFitness[i+indices[-2]]
        return population, fitness

