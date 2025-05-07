from brain import (
    Neuron,
    Gene,
    NeuralNet,
    ACTIONS
)
import functools
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv
import random
import numpy as np


class BioSim(ParallelEnv):
    metadata = {
        "name": "BioSim",
    }

    def __init__(self, size = 128, n_agents = 10):
        self.n_agents = n_agents
        
        
        self.agent_position = None
        self.agent_genome = None

        self.timestep = None
        self.size = size

    def reset(self, seed=None, options=None):
        #on prend les agents de la liste
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        #on leur donne une position aléatoire 
        
        self.agent_position = {
            agent: np.array([
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1)
                ])
                    for agent in self.agents
                }
        self.agent_genome = {
            agents : Gene.make_random_genome()
            for agents in self.agents
        }
        self.agent_brains = {
            agent: NeuralNet.create_wiring_from_genome(self.agent_genome[agent])
            for agent in self.agents
        }


        self.timestep = 0
        #pas besoin des observations au début
        observations = {
            agents : self._get_observation(agents)
            for agents in self.agents
        }
        return observations

    def step(self, actions):
        rewards = {agents : 0 for agents in self.agents}
        termination = {agent: False for agent in self.agents}
        truncationns = {agents : False for agents in self.agents}
        infos = {agents : {} for agents in self.agents}

        for agents in self.agents :
            action_outputs = NeuralNet.get_action_outputs(self.agent_brains[agents])
            if action_outputs :
                best_action_index = max(action_outputs, key = action_outputs.get)
                action_name = ACTIONS[best_action_index]

        #le max et le min permet d'avoir des bordures en comparant 0 et position
        # Le 0, 0 commence en haut à gauche, donc y doit être rapproché de 0 pour monter.
        if action_name == "UP":
            self.agent_position[agents][1] = max(0, self.agent_position[agents][1] - 1)

        elif action_name == "DOWN":
            self.agent_position[agents][1] = min(self.size - 1, self.agent_position[agents][1] + 1)

        elif action_name == "WEST":
            self.agent_position[agents][0] = max(0, self.agent_position[agents][0] - 1)

        elif action_name == "EAST":
            self.agent_position[agents][0] = min(self.size - 1, self.agent_position[agents][0] + 1)

        elif action_name == "NORTH WEST" :
            self.agent_position[agents][0,1]

        elif action_name == "NORTH WEST":
            self.agent_position[agents][0] = max(0, self.agent_position[agents][0] - 1)  # Déplace l'agent vers la gauche
            self.agent_position[agents][1] = max(0, self.agent_position[agents][1] - 1)  # Déplace l'agent vers le haut

            # North East : Déplacement vers le haut à droite
        elif action_name == "NORTH EAST":
                self.agent_position[agents][0] = min(self.size - 1, self.agent_position[agents][0] + 1)  # Déplace l'agent vers la droite
                self.agent_position[agents][1] = max(0, self.agent_position[agents][1] - 1)  # Déplace l'agent vers le haut

            # South West : Déplacement vers le bas à gauche
        elif action_name == "SOUTH WEST":
                self.agent_position[agents][0] = max(0, self.agent_position[agents][0] - 1)  # Déplace l'agent vers la gauche
                self.agent_position[agents][1] = min(self.size - 1, self.agent_position[agents][1] + 1)  # Déplace l'agent vers le bas

            # South East : Déplacement vers le bas à droite
        elif action_name == "SOUTH EAST":
                self.agent_position[agents][0] = min(self.size - 1, self.agent_position[agents][0] + 1)  # Déplace l'agent vers la droite
                self.agent_position[agents][1] = min(self.size - 1, self.agent_position[agents][1] + 1)  # Déplace l'agent vers le bas



        self.timestep += 1 
        observations = {
            agents : self._get_observation(agents)
            for agents in self.agents
        }
        return observations, rewards, termination, truncationns, infos
    
    def _get_observation(self, agents) :
     pass

    
     @functools.lru_cache(maxsize=None)
     def observation_space(self, agent):
        return MultiDiscrete(np.array([self.size, self.size]))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(ACTIONS))