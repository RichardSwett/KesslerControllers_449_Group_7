# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time
from graphics_both import GraphicsBoth
from src.kesslergame import Scenario, KesslerGame, GraphicsType
from FuzzyController import GroupSevenController
from EasyGA.structure import Chromosome
from Controller_Defs import read_list_from_file
import os


my_test_scenario = Scenario(name='Test Scenario',
                            num_asteroids=10,
                            ship_states=[
                                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1}
                            ],
                            map_size=(1000, 800),
                            time_limit=120,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.Tkinter,
                 'realtime_multiplier': 1,
                 'graphics_obj': None}

game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
# game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

pre = time.perf_counter()

if os.path.exists("best_chromosome.pk1"):

    chromosome = Chromosome(read_list_from_file("best_chromosome.pk1"))

else: 

    chromosome = Chromosome([[[0.004909619761891005, 0.024494051314874955, 0.056946675102348465], 
                   [35, 95, 120], [59.419856828504486, 98.7598202838615, 159.2490580996916], 
                   [43.313031350255784, 52.58580234991643, 278.8065261700964, 493.1566041420365, 683.5075146580989], 
                   [58.243503838056945, 101.23232459042296, 142.76971633525383], 
                   [186.99693276638138], 
                   [2.9522636948690675, 5.805887299076666, 8.211396101189177]]])

score, perf_data = game.run(scenario=my_test_scenario, controllers = [GroupSevenController(chromosome = chromosome)])

print('Scenario eval time: '+str(time.perf_counter()-pre))
print(score.stop_reason)
print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
print('Deaths: ' + str([team.deaths for team in score.teams]))
print('Accuracy: ' + str([team.accuracy for team in score.teams]))
print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
print('Evaluated frames: ' + str([controller.eval_frames for controller in score.final_controllers]))
