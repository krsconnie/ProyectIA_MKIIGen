import csv
import os
from lifes_laws.damage_store import damage_store


class DamageCSVReporter:
    def __init__(self, filename):
        self.filename = filename
        self.current_generation = 0
        if not os.path.isfile(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['NumGen', 'Prom_damage'])

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

    def species_stagnant(self, sid, species):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        damages = [
            damage_store.get(genome.key, 0)
            for genome in population.values()
        ]

        if not damages:
            return

        generation = self.current_generation
        prom_damage = sum(damages) / len(damages)

        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([generation, prom_damage])
