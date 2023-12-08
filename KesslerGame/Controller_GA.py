from EasyGA import EasyGA
from EasyGA.structure import Chromosome
import time
from src.kesslergame import Scenario, GraphicsType,TrainerEnvironment
from Controller_Defs import write_list_to_file, read_list_from_file
from FuzzyController import GroupSevenController
import numpy as np

def fitness(chromosome) :
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
    
    #game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
    game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

    pre = time.perf_counter()
    score, perf_data = game.run(scenario=my_test_scenario, controllers = [GroupSevenController(chromosome=chromosome)])

    eval_time = time.perf_counter()-pre
    asteroids_hit = [team.asteroids_hit for team in score.teams]
    accuracy = [team.accuracy for team in score.teams]
    print('score: ' + str((eval_time/100) + (asteroids_hit[0]/my_test_scenario.max_asteroids) + accuracy[0]))

    return (eval_time/50) + (asteroids_hit[0]/my_test_scenario.max_asteroids) + accuracy[0]

def generate_chromosome():

    bullet_time = sorted([np.random.uniform(0, 0.1),
                   np.random.uniform(0, 0.1),
                   np.random.uniform(0, 0.1)])
    
    theta_delta = [35, 95, 120]
    
    ship_turn = [np.random.uniform(35, 60),
                 np.random.uniform(60, 135),
                 np.random.uniform(135, 180)]
    
    ast_dist = [np.random.uniform(0,50), 
                np.random.uniform(50, 225),
                np.random.uniform(225, 425),
                np.random.uniform(425, 625),
                np.random.uniform(625, 700)]
    
    ast_angle = [np.random.uniform(15, 60),
                 np.random.uniform(60, 135),
                 np.random.uniform(135, 180)]
    
    thrust = [np.random.uniform(150, 225)]
              
    danger = [np.random.uniform(0, 3),
              np.random.uniform(0, 10),
              np.random.uniform(7, 10)]
        
    return [bullet_time, theta_delta, ship_turn, 
            ast_dist, ast_angle, thrust, danger]


ga = EasyGA.GA()

ga.chromosome_length = 1
ga.population_size = 1
ga.gene_impl = lambda: generate_chromosome()
ga.target_fitness_type = 'max'
ga.generation_goal = 1
ga.fitness_function_impl = fitness

ga.evolve()

ga.print_best_chromosome()
best_chromosome = ga.population[0]
write_list_to_file("best_chromosome.pk1", best_chromosome)





