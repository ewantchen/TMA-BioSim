import numpy as np 
import random
from typing import List, Dict, Tuple



ACTIONS = [
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "NORTH WEST",
    "NORTH EAST",
    "SOUTH WEST",
    "SOUTH EAST"
]

n_ACTIONS = len(ACTIONS)

SENSORS = [
   "X_POS",
   "Y_POS",
   "RANDOM",
   "BOUNDARY_DIST_X",
   "BOUNDARY_DIST_Y",
]

n_SENSORS = len(SENSORS)

sensor_values = {
    # Position X normalisée [0, 1] où 0 = bord gauche, 1 = bord droit
    "X_POS": lambda agent_x, world_size: agent_x / (world_size - 1),
    
    # Position Y normalisée [0, 1] où 0 = bord bas, 1 = bord haut
    "Y_POS": lambda agent_y, world_size: agent_y / (world_size - 1),
    
    # Valeur aléatoire pour l'aléatoire [0, 1]
    "RANDOM": lambda *_: random.random(),
    
    # Distance au bord X normalisée [0, 1] où 0 = au bord, 1 = au centre
    "BOUNDARY_DIST_X": lambda agent_x, world_size: min(agent_x, world_size - 1 - agent_x) / (world_size // 2),
    
    # Distance au bord Y normalisée [0, 1] où 0 = au bord, 1 = au centre
    "BOUNDARY_DIST_Y": lambda agent_y, world_size: min(agent_y, world_size - 1 - agent_y) / (world_size // 2)
}

"""défini une connection possible dans l'agent"""
class Gene : 
    def __init__(self):
        self.sourceType: int = 0 #0=NEURON, 1=SENSOR
        self.sourceNum: int = 0 # Index de la source (d'où vient l'input)
        self.sinkType: int = 0 #0=NEURON, 1=ACTION
        self.sinkNum: int = 0 #Index de la cible (où va l'output)
        self.weight: int = 0 # Poids (int16)

    def weightAsFloat(self) -> float : 
        #Converti le poids entier en float [-1.0 , 1.0]
        return self.weight / 8192.0 #Même méthode dans BioSim4
    

    def makeRandomWeight() -> int :
        #Poid aléatoire (comme dans BioSim4)
        return np.random.randint(-32768, 32767) # int16 signé

    def make_random_gene() -> "Gene" :
        #crée un gène (comme dans BioSim4)
        gene = Gene()
        gene.sourceType = np.random.randint(0,2) # 0=NEURON, 1=SENSOR
        gene.sourceNum = np.random.randint(0, 0x7FFF) # 15 bits (comme BioSim)
        gene.sinkType = np.random.randint(0,2) # 0=NEURON, 1=ACTION
        gene.sinkNum = np.random.randint(0,n_ACTIONS) #15 bits
        gene.weight = Gene.makeRandomWeight()
        return gene
    
    @staticmethod
    def make_random_genome(min_len=10, max_len=50) -> "List[Gene]" :
        #Crée un génome aléatoire
        length = np.random.randint(min_len, max_len + 1)
        return [Gene.make_random_gene() for _ in range(length)]
    
    #Gestion de l'aléatoire dans le génome des individus
    @staticmethod
    def random_bit_flip(gene : "Gene") -> "Gene" :
        chance = np.random.rand()
        if chance < 0.2 :
            gene.sourceType ^= 1 #Flip entre 0 et 1 
        elif chance < 0.4 : 
            gene.sinkType ^= 1 
        elif chance < 0.6 :
            gene.sourceNum ^= (1 << np.random.randint(0,15)) 
        elif chance < 0.8 : 
            gene.sinkNum ^= (1 << random.randint(0,15))
        else :
            gene.weight ^= (1 << np.random.randint(0,15))
        return gene
    
    def apply_point_mutations(genome : List["Gene"], mutation_rate = 0.01) -> List["Gene"] :
        #applique des mutations aléatoires (comme BioSim)
        for gene in genome : 
            if np.random.rand() < mutation_rate : 
                Gene.random_bit_flip(gene)
        return genome

    
    def random_insert_deletion(genome: List["Gene"], max_length=100) -> List["Gene"] :
        #ajoute ou supprime un gène aléatoirement
        if np.random.rand() < 0.05 : # 5% de chance 
            if np.random.rand() < 0.5 and len(genome) > 1 :
                genome.pop(np.random.randint(0,len(genome))) #supprime
            elif len(genome) < max_length :
                genome.append(Gene.make_random_gene())
        return genome
        
"""Conversion du génome en réseau neuronal"""
class Neuron :
    def __init__(self):
        self.output : float = 0.5 #Valeur par défaut
        self.driven: bool = False
        self.input : float = 0.0 #pour l'instant


class NeuralNet :
    def __init__(self):
        self.connections : List[Gene] = []
        self.neurons : List[Neuron] = []
    
    def create_wiring_from_genome(genome: List[Gene], max_neurons=1000) -> "NeuralNet" :
        """convertit le génome en réseau neuronal"""
        # étape 1 : filtrer les neurones inutiles
        used_neurons = set()
        for gene in genome : 
            if gene.sinkType == 0 : 
                used_neurons.add(gene.sinkNum)
            if gene.sourceType == 0 :
                used_neurons.add(gene.sourceNum)

        # Étape 2: Renumérotation (ex: [5, 10, 15] → [0, 1, 2])
        neuron_remap = {old: new for new, old in enumerate(sorted(used_neurons))}

        #Étape 3 : Construire le réseau 
        net = NeuralNet()
        net.neurons = [Neuron() for _ in range(len(used_neurons))]
        
        for gene in genome : 
            #Ignorer les connexions vers les neurones supprimés 
            if gene.sinkType == 0 and gene.sinkNum not in neuron_remap :
                continue
            
            new_gene = Gene()
            
            new_gene.sourceType = gene.sourceType
            new_gene.sinkType = gene.sinkType
            new_gene.weight = gene.weight

            if gene.sourceType == 0 : 
                new_gene.sourceNum = neuron_remap[gene.sourceNum]
            else : 
                new_gene.sourceNum = gene.sourceNum

            if gene.sinkType == 0 :
                new_gene.sinkNum = neuron_remap[gene.sinkNum]
            else : 
                new_gene.sinkNum = gene.sinkNum
            
            net.connections.append(new_gene)

        return net
    
    def get_action_outputs(self) -> Dict[int, float] :
        #Retourne les activations des neurones d'action
        action_outputs = {}
        for gene in self.connections : 
            if gene.sinkType == 1 :#vers une action
                output_value = 0.0
                if gene.sourceType == 0 : 
                    #si ça va vers une action, et viens d'un neurone (pas sensor), alors on calcule
                    # l'output sur l'action
                    output_value = self.neurons[gene.sourceNum].output * gene.weightAsFloat()
                elif gene.sourceType == 1 :
                    output_value = self.neurons[gene.sourceNum].output * gene.weightAsFloat()
                else :
                    output_value = 0.0
            
            if gene.sinkNum <= n_ACTIONS :
                if gene.sinkNum in action_outputs :
                    action_outputs[gene.sinkNum] += output_value
                else :
                    action_outputs[gene.sinkNum] = output_value
        return action_outputs
    
    def _get_sensor_values(self, agent_position, world_size) -> Dict[int, float]:
        """Calcule toutes les valeurs des capteurs en utilisant le dictionnaire"""
        x, y = agent_position
    
        return {
                0: sensor_values["X_POS"](x, world_size),
                1: sensor_values["Y_POS"](y, world_size),
                2: sensor_values["RANDOM"](),
                3: sensor_values["BOUNDARY_DIST_X"](x, world_size),
                4: sensor_values["BOUNDARY_DIST_Y"](y, world_size)
                }


    def get_sensors_input(self, sensor_values: Dict[int, float]) -> None:
        """Transmet les entrées sensorielles aux neurones connectés"""
        for gene in self.connections:
        # Seulement les connexions SENSOR -> NEURON
            if gene.sourceType == 1 and gene.sinkType == 0:
                if gene.sourceNum in sensor_values:
                # Accumule les entrées pondérées dans le neurone cible
                    self.neurons[gene.sinkNum].input += (
                        sensor_values[gene.sourceNum] * gene.weightAsFloat()
                )
                    self.neurons[gene.sinkNum].driven = True


    def feed_forward(self) -> None:
        """Propage les signaux à travers le réseau"""
    # 1. Traitement des neurones
        for neuron in self.neurons:
            if neuron.driven:
                neuron.output = np.tanh(neuron.input)  # Fonction d'activation
                neuron.input = 0.0  # Réinitialise pour le prochain cycle
    



    






