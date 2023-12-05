from EasyGA import EasyGA
from EasyGA.structure import Chromosome
import random
import time
from src.kesslergame import Scenario, KesslerGame, GraphicsType,TrainerEnvironment
from scottdickcontroller import ScottDickController
from ivancontroller_copy import IvanControllercopy
from ivancontroller import IvanController
from YashController import YashController
from yash_utils import write_list_to_file, read_list_from_file
from KavishController import KavishController
from test_controller import TestController
from graphics_both import GraphicsBoth
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
    score, perf_data = game.run(scenario=my_test_scenario, controllers = [YashController(chromosome=chromosome)])

    # print('Scenario eval time: '+str(time.perf_counter()-pre))
    # print(score.stop_reason)
    # print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
    # print('Deaths: ' + str([team.deaths for team in score.teams]))
    # print('Accuracy: ' + str([team.accuracy for team in score.teams]))
    # print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
    # print('Evaluated frames: ' + str([controller.eval_frames for controller in score.final_controllers]))
    # print('asteroids: ' + str(my_test_scenario.max_asteroids))

    eval_time = time.perf_counter()-pre
    asteroids_hit = [team.asteroids_hit for team in score.teams]
    accuracy = [team.accuracy for team in score.teams]
    print('score: ' + str((eval_time/100) + (asteroids_hit[0]/my_test_scenario.max_asteroids) + accuracy[0]))

    return (eval_time/100) + (asteroids_hit[0]/my_test_scenario.max_asteroids) + accuracy[0]

def generate_chromosome():
    bullet_time_s = random.uniform(0, 2)
    bullet_time_m = random.uniform(0, 2)
    bullet_time_l = random.randrange(-10, 10)

    theta_delta_nl = random.randrange(-10, 10)
    theta_delta_ns = random.randrange(-10, 10)
    theta_delta_z = random.randrange(-10, 10)
    theta_delta_ps = random.randrange(-10, 10)
    theta_delta_pl = random.randrange(-10, 10)

    ship_turn_nl = random.randrange(0, 59)
    ship_turn_ns = random.randrange(-29, 29)
    ship_turn_z = random.randrange(-59, 59)
    ship_turn_ps = random.randrange(-29, 29)
    ship_turn_pl = random.randrange(-59, 0)

    ship_fire_n = random.uniform(-0.9, 0.9)
    #ship_fire_y = random.uniform(-10, 10)
    ship_fire_y = ship_fire_n

    rel_ast_dist_nl = random.randrange(0, 99)
    rel_ast_dist_ns = random.randrange(-49, 74)
    rel_ast_dist_z = random.randrange(-74, 74)
    rel_ast_dist_ps = random.randrange(-74 , 49)
    rel_ast_dist_pl = random.randrange(0, 99)

    rel_ast_angl_nl = random.randrange(0, 59)
    rel_ast_angl_ns = random.randrange(-29, 29)
    rel_ast_angl_z = random.randrange(-59, 59)
    rel_ast_angl_ps = random.randrange(-29, 29)
    rel_ast_angl_pl = random.randrange(-59, 0)

    ship_thrust_nl = random.randrange(0, 24)
    ship_thrust_ns = random.randrange(-25, 24)
    ship_thrust_z = random.randrange(-25, 24)
    ship_thrust_ps = random.randrange(-25, 24)
    ship_thrust_pl = random.randrange(0, 99)

    return np.array([
        bullet_time_s,
        bullet_time_m,
        bullet_time_l,
        theta_delta_nl,
        theta_delta_ns,
        theta_delta_z,
        theta_delta_ps,
        theta_delta_pl,
        ship_turn_nl,
        ship_turn_ns,
        ship_turn_z,
        ship_turn_ps,
        ship_turn_pl,
        ship_fire_n,
        ship_fire_y,
        rel_ast_dist_nl,
        rel_ast_dist_ns,
        rel_ast_dist_z,
        rel_ast_dist_ps,
        rel_ast_dist_pl,
        rel_ast_angl_nl,
        rel_ast_angl_ns,
        rel_ast_angl_z,
        rel_ast_angl_ps,
        rel_ast_angl_pl,
        ship_thrust_nl,
        ship_thrust_ns,
        ship_thrust_z,
        ship_thrust_ps,
        ship_thrust_pl
    ])

ga = EasyGA.GA()

ga.chromosome_length = 1
ga.population_size = 1
ga.gene_impl = lambda: generate_chromosome()
ga.target_fitness_type = 'max'
ga.generation_goal = 1
ga.fitness_function_impl = fitness

ga.evolve()

ga.print_best_chromosome()
best_chromsome = ga.population[0]
write_list_to_file("best_chromosome.pk1", best_chromsome.gene_value_list)
print(read_list_from_file("best_chromosome.pk1"))
print(Chromosome(read_list_from_file("best_chromosome.pk1")))
