import csv
import os

class CustomCSVReporter:
    def __init__(self, filename):
        self.filename = filename
        self.current_generation = 0
        if not os.path.isfile(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Gen', 'Prom_fit', 'Prom_positive_fit', 'Prom_negative_fit'])

    def start_generation(self, generation):
        self.current_generation = generation

    def info(self, msg):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def end(self):
        pass

    def end_generation(self, config, population, species_set):
        pass



    def post_evaluate(self, config, population, species, best_genome):
        fitness_values = [genome.fitness for genome in population.values()]

        if not fitness_values:
            return

        generation = self.current_generation

        prom_fit = sum(fitness_values) / len(fitness_values)
        positives = [f for f in fitness_values if f > 0]
        negatives = [f for f in fitness_values if f < 0]

        prom_pos = sum(positives) / len(positives) if positives else 0
        prom_neg = sum(negatives) / len(negatives) if negatives else 0

        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([generation, prom_fit, prom_pos, prom_neg])
