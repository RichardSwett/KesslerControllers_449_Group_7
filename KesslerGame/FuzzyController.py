# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from src.kesslergame import KesslerController
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
from Controller_Defs import relative_pos, find_intercept


class GroupSevenController(KesslerController):
    def __init__(self, chromosome = None):
        self.eval_frames = 0
        
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-180,180,1), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
 
        bullet_time_param = chromosome[0].value[0]
        theta_param = chromosome[0].value[1]
        turn_param = chromosome[0].value[1]

        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0, 0, bullet_time_param[0]])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0, bullet_time_param[1], 0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe, bullet_time_param[2], 0.1)
        
        #Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        theta_delta['NXL'] = fuzz.zmf(theta_delta.universe, -180, -theta_param[2])
        theta_delta['NL'] = fuzz.trimf(theta_delta.universe, [-135, -theta_param[1], -55])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-70, -theta_param[0], 0])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1,0,1])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [0, theta_param[0], 70])
        theta_delta['PL'] = fuzz.trimf(theta_delta.universe, [55, theta_param[1], 135])
        theta_delta['PXL'] = fuzz.smf(theta_delta.universe, theta_param[2], 180)
        
        #Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        ship_turn['NXL'] = fuzz.zmf(ship_turn.universe, -180, -turn_param[2])
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-135,-turn_param[1],-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-60,-turn_param[0], 0])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-35,0,35])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [0,turn_param[0],60])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [60,turn_param[1],135])    
        ship_turn['PXL'] = fuzz.smf(ship_turn.universe, turn_param[2], 180)
        
        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1]) 
                
        #Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NXL'], (ship_turn['NXL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NXL'], ship_fire['Y']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NXL'], ship_fire['Y']))
        rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PXL'], ship_fire['Y']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PXL'], ship_fire['Y']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PXL'], (ship_turn['PXL'], ship_fire['N']))

        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NXL'], (ship_turn['NXL'], ship_fire['N']))     
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NXL'], ship_fire['N']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NL'], ship_fire['Y']))
        rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))    
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PL'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PXL'], ship_fire['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PXL'], (ship_turn['PXL'], ship_fire['N']))

        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NXL'], (ship_turn['NXL'], ship_fire['N']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NXL'], ship_fire['N']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NL'], ship_fire['Y']))
        rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PL'], ship_fire['Y']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PXL'], ship_fire['N']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PXL'], (ship_turn['PXL'], ship_fire['N']))
     
        self.targetting = ctrl.ControlSystem()
        self.targetting.addrule(rule1)
        self.targetting.addrule(rule2)
        self.targetting.addrule(rule3)
        self.targetting.addrule(rule4)
        self.targetting.addrule(rule5)
        self.targetting.addrule(rule6)
        self.targetting.addrule(rule7)
        self.targetting.addrule(rule8)
        self.targetting.addrule(rule9)
        self.targetting.addrule(rule10)
        self.targetting.addrule(rule11)
        self.targetting.addrule(rule12)
        self.targetting.addrule(rule13)
        self.targetting.addrule(rule14)
        self.targetting.addrule(rule15)
        self.targetting.addrule(rule16)
        self.targetting.addrule(rule17)
        self.targetting.addrule(rule18)
        self.targetting.addrule(rule19)
        self.targetting.addrule(rule20)
        self.targetting.addrule(rule21)

        rel_ast_dist = ctrl.Antecedent(np.arange(0,700,1),'rel_ast_dist')
        rel_ast_angl = ctrl.Antecedent(np.arange(-180,180,1), 'rel_ast_angl')
        ship_thrust = ctrl.Consequent(np.arange(-250,250,1),'ship_thrust')

        ast_dist_param = chromosome[0].value[3]
        ast_angle_param = chromosome[0].value[4]
        thurst_param = chromosome[0].value[5]

        #Define each of the new inputs and outputs
        rel_ast_dist['NL'] = fuzz.zmf(rel_ast_dist.universe,0, 50)
        rel_ast_dist['NS'] = fuzz.trimf(rel_ast_dist.universe, [50,ast_dist_param[1],225])
        rel_ast_dist['Z'] = fuzz.trimf(rel_ast_dist.universe, [225,ast_dist_param[2],425])
        rel_ast_dist['PS'] = fuzz.trimf(rel_ast_dist.universe, [425,ast_dist_param[3],625])
        rel_ast_dist['PL'] = fuzz.smf(rel_ast_dist.universe, 625,700)

        rel_ast_angl['NXL'] = fuzz.trimf(rel_ast_angl.universe, [-180,-ast_angle_param[2],-135])
        rel_ast_angl['NL'] = fuzz.trimf(rel_ast_angl.universe, [-135,-ast_angle_param[1],-60])
        rel_ast_angl['NS'] = fuzz.trimf(rel_ast_angl.universe, [-60,-ast_angle_param[0],-15])
        rel_ast_angl['Z'] = fuzz.trimf(rel_ast_angl.universe, [-15,0,15])
        rel_ast_angl['PS'] = fuzz.trimf(rel_ast_angl.universe, [15,ast_angle_param[0],60])
        rel_ast_angl['PL'] = fuzz.trimf(rel_ast_angl.universe, [60,ast_angle_param[1],135])
        rel_ast_angl['PXL'] = fuzz.trimf(rel_ast_angl.universe, [135,ast_angle_param[2],180])
          
        ship_thrust['NL'] = fuzz.zmf(ship_thrust.universe, -350, -225)
        ship_thrust['NS'] = fuzz.trimf(ship_thrust.universe, [-225,-thurst_param[0],-75])
        ship_thrust['Z'] = fuzz.trimf(ship_thrust.universe, [-150, 0, 150])
        ship_thrust['PS'] = fuzz.trimf(ship_thrust.universe, [150,thurst_param[0],225])
        ship_thrust['PL'] = fuzz.smf(ship_thrust.universe, 225, 350)

        #Declare each fuzzy rule for the Evade Mode
        rule1_evade = ctrl.Rule((rel_ast_angl['NXL'] & rel_ast_dist['NL']), (ship_turn['Z'], ship_thrust['PL']))
        rule2_evade = ctrl.Rule((rel_ast_angl['NXL'] & rel_ast_dist['NS']), (ship_turn['Z'], ship_thrust['PL']))
        rule3_evade = ctrl.Rule((rel_ast_angl['NXL'] & rel_ast_dist['Z']), (ship_turn['PXL'], ship_thrust['PL']))
        rule4_evade = ctrl.Rule((rel_ast_angl['NXL'] & rel_ast_dist['PS']), (ship_turn['PXL'], ship_thrust['PL']))
        rule5_evade = ctrl.Rule((rel_ast_angl['NXL'] & rel_ast_dist['PL']), (ship_turn['PXL'], ship_thrust['PS']))

        rule6_evade = ctrl.Rule((rel_ast_angl['NL'] & rel_ast_dist['NL']), (ship_turn['PXL'], ship_thrust['PL']))
        rule7_evade = ctrl.Rule((rel_ast_angl['NL'] & rel_ast_dist['NS']), (ship_turn['PXL'], ship_thrust['PL']))
        rule8_evade = ctrl.Rule((rel_ast_angl['NL'] & rel_ast_dist['Z']), (ship_turn['PXL'], ship_thrust['PS']))
        rule9_evade = ctrl.Rule((rel_ast_angl['NL'] & rel_ast_dist['PS']), (ship_turn['PL'], ship_thrust['PS']))
        rule10_evade = ctrl.Rule((rel_ast_angl['NL'] & rel_ast_dist['PL']), (ship_turn['PL'], ship_thrust['PS']))

        rule11_evade = ctrl.Rule((rel_ast_angl['NS'] & rel_ast_dist['NL']), (ship_turn['NL'], ship_thrust['NS']))
        rule12_evade = ctrl.Rule((rel_ast_angl['NS'] & rel_ast_dist['NS']), (ship_turn['NL'], ship_thrust['NS']))
        rule13_evade = ctrl.Rule((rel_ast_angl['NS'] & rel_ast_dist['Z']), (ship_turn['NL'], ship_thrust['NS']))
        rule14_evade = ctrl.Rule((rel_ast_angl['NS'] & rel_ast_dist['PS']), (ship_turn['NL'], ship_thrust['NS']))
        rule15_evade = ctrl.Rule((rel_ast_angl['NS'] & rel_ast_dist['PL']), (ship_turn['NL'], ship_thrust['NS']))

        rule16_evade = ctrl.Rule((rel_ast_angl['Z'] & rel_ast_dist['NL']), (ship_turn['PXL'], ship_thrust['NL']))
        rule17_evade = ctrl.Rule((rel_ast_angl['Z'] & rel_ast_dist['NS']), (ship_turn['Z'], ship_thrust['NL']))
        rule18_evade = ctrl.Rule((rel_ast_angl['Z'] & rel_ast_dist['Z']), (ship_turn['Z'], ship_thrust['NL']))
        rule19_evade = ctrl.Rule((rel_ast_angl['Z'] & rel_ast_dist['PS']), (ship_turn['Z'], ship_thrust['PS']))
        rule20_evade = ctrl.Rule((rel_ast_angl['Z'] & rel_ast_dist['PL']), (ship_turn['Z'], ship_thrust['PL']))

        rule21_evade = ctrl.Rule((rel_ast_angl['PS'] & rel_ast_dist['NL']), (ship_turn['PL'], ship_thrust['NS']))
        rule22_evade = ctrl.Rule((rel_ast_angl['PS'] & rel_ast_dist['NS']), (ship_turn['PL'], ship_thrust['NS']))
        rule23_evade = ctrl.Rule((rel_ast_angl['PS'] & rel_ast_dist['Z']), (ship_turn['PL'], ship_thrust['NS']))
        rule24_evade = ctrl.Rule((rel_ast_angl['PS'] & rel_ast_dist['PS']), (ship_turn['PL'], ship_thrust['NS']))
        rule25_evade = ctrl.Rule((rel_ast_angl['PS'] & rel_ast_dist['PL']), (ship_turn['PL'], ship_thrust['NS']))

        rule26_evade = ctrl.Rule((rel_ast_angl['PL'] & rel_ast_dist['NL']), (ship_turn['NXL'], ship_thrust['PL']))
        rule27_evade = ctrl.Rule((rel_ast_angl['PL'] & rel_ast_dist['NS']), (ship_turn['NXL'], ship_thrust['PL']))
        rule28_evade = ctrl.Rule((rel_ast_angl['PL'] & rel_ast_dist['Z']), (ship_turn['NXL'], ship_thrust['PL']))
        rule29_evade = ctrl.Rule((rel_ast_angl['PL'] & rel_ast_dist['PS']), (ship_turn['NL'], ship_thrust['PS']))
        rule30_evade = ctrl.Rule((rel_ast_angl['PL'] & rel_ast_dist['PL']), (ship_turn['NL'], ship_thrust['PS']))

        rule31_evade = ctrl.Rule((rel_ast_angl['PXL'] & rel_ast_dist['NL']), (ship_turn['Z'], ship_thrust['PL']))
        rule32_evade = ctrl.Rule((rel_ast_angl['PXL'] & rel_ast_dist['NS']), (ship_turn['Z'], ship_thrust['PL']))
        rule33_evade = ctrl.Rule((rel_ast_angl['PXL'] & rel_ast_dist['Z']), (ship_turn['PXL'], ship_thrust['PL']))
        rule34_evade = ctrl.Rule((rel_ast_angl['PXL'] & rel_ast_dist['PS']), (ship_turn['NXL'], ship_thrust['PS']))
        rule35_evade = ctrl.Rule((rel_ast_angl['PXL'] & rel_ast_dist['PL']), (ship_turn['NXL'], ship_thrust['PS']))

        #Evade mode Ruleset activation
        self.evade_control = ctrl.ControlSystem()
        self.evade_control.addrule(rule1_evade)
        self.evade_control.addrule(rule2_evade)
        self.evade_control.addrule(rule3_evade)
        self.evade_control.addrule(rule4_evade)
        self.evade_control.addrule(rule5_evade)
        self.evade_control.addrule(rule6_evade)
        self.evade_control.addrule(rule7_evade)
        self.evade_control.addrule(rule8_evade)
        self.evade_control.addrule(rule9_evade)
        self.evade_control.addrule(rule10_evade)
        self.evade_control.addrule(rule11_evade)
        self.evade_control.addrule(rule12_evade)
        self.evade_control.addrule(rule13_evade)
        self.evade_control.addrule(rule14_evade)
        self.evade_control.addrule(rule15_evade)
        self.evade_control.addrule(rule16_evade)
        self.evade_control.addrule(rule17_evade)
        self.evade_control.addrule(rule18_evade)
        self.evade_control.addrule(rule19_evade)
        self.evade_control.addrule(rule20_evade)
        self.evade_control.addrule(rule21_evade)
        self.evade_control.addrule(rule22_evade)
        self.evade_control.addrule(rule23_evade)
        self.evade_control.addrule(rule24_evade)
        self.evade_control.addrule(rule25_evade)
        self.evade_control.addrule(rule26_evade)
        self.evade_control.addrule(rule27_evade)
        self.evade_control.addrule(rule28_evade)
        self.evade_control.addrule(rule29_evade)
        self.evade_control.addrule(rule30_evade)
        self.evade_control.addrule(rule31_evade)
        self.evade_control.addrule(rule32_evade)
        self.evade_control.addrule(rule33_evade)
        self.evade_control.addrule(rule34_evade)
        self.evade_control.addrule(rule35_evade)

        num_aster_close = ctrl.Antecedent(np.arange(0, 10, 1), 'num_aster_close')
        num_aster_far = ctrl.Antecedent(np.arange(0, 10, 1), 'num_aster_far')

        danger = ctrl.Consequent(np.arange(0, 10, 1), 'danger')
        danger_param = chromosome[0].value[6]

        num_aster_close['S'] = fuzz.zmf(num_aster_close.universe, 0, danger_param[0])
        num_aster_close['M'] = fuzz.trimf(num_aster_close.universe, [0, danger_param[1], 10])
        num_aster_close['L'] = fuzz.smf(num_aster_close.universe, danger_param[2], 10)

        num_aster_far['S'] = fuzz.zmf(num_aster_far.universe, 0, danger_param[0])
        num_aster_far['M'] = fuzz.trimf(num_aster_far.universe, [0, danger_param[1], 10])
        num_aster_far['L'] = fuzz.smf(num_aster_far.universe, danger_param[2], 10)

        danger['S'] = fuzz.zmf(danger.universe, 0, danger_param[0])
        danger['M'] = fuzz.trimf(danger.universe, [0, danger_param[1], 10])
        danger['L'] = fuzz.smf(danger.universe, danger_param[2], 10)

        rule1_danger = ctrl.Rule((num_aster_close["S"] & num_aster_far["S"]), danger["S"])
        rule2_danger = ctrl.Rule((num_aster_close["M"] & num_aster_far["S"]), danger["M"])
        rule3_danger = ctrl.Rule((num_aster_close["L"] & num_aster_far["S"]), danger["L"])

        rule4_danger = ctrl.Rule((num_aster_close["S"] & num_aster_far["M"]), danger["S"])
        rule5_danger = ctrl.Rule((num_aster_close["M"] & num_aster_far["M"]), danger["M"])
        rule6_danger = ctrl.Rule((num_aster_close["L"] & num_aster_far["M"]), danger["L"])

        rule7_danger = ctrl.Rule((num_aster_close["S"] & num_aster_far["L"]), danger["M"])
        rule8_danger = ctrl.Rule((num_aster_close["M"] & num_aster_far["L"]), danger["L"])
        rule9_danger = ctrl.Rule((num_aster_close["L"] & num_aster_far["L"]), danger["L"])

        self.assess_danger = ctrl.ControlSystem()
        self.assess_danger.addrule(rule1_danger)
        self.assess_danger.addrule(rule2_danger)
        self.assess_danger.addrule(rule3_danger)
        self.assess_danger.addrule(rule4_danger)
        self.assess_danger.addrule(rule5_danger)
        self.assess_danger.addrule(rule6_danger)
        self.assess_danger.addrule(rule7_danger)
        self.assess_danger.addrule(rule8_danger)
        self.assess_danger.addrule(rule9_danger)

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        """
        Method processed each time step by this controller.
        """
        bullet_speed = 800

        #get all asteroids and sort based on distances
        asteroids = game_state['asteroids']
        close_asteroids = []
        num_aster_close = 0
        num_aster_far = 0
        for asteroid in asteroids:
            
            relative_dist, _, _ = relative_pos(asteroid, ship_state)
            close_asteroids.append((asteroid, relative_dist))

            if relative_dist < 100:
                num_aster_close += 1
            elif num_aster_far < 10:
                num_aster_far += 1


        k = int(len(asteroids) / 5)
        if k > 10: 
            k = 10

        sorted_asteroids = sorted(close_asteroids, key = lambda x:x[1])
        sorted_asteroids = sorted_asteroids[0:k]
        closest_asteroid = sorted_asteroids[0]

        #determine danger
        danger_sim = ctrl.ControlSystemSimulation(self.assess_danger, flush_after_run=1)
        danger_sim.input["num_aster_close"] = num_aster_close
        danger_sim.input["num_aster_far"] = num_aster_far
        danger_sim.compute()
        danger_level = danger_sim.output["danger"]

        #if still -> get closest asteroid and shoot
        #if moving -> find safe spot -> move and shoot if one of the closest asteroid in the way

        if danger_level > 5:
            evade = ctrl.ControlSystemSimulation(self.evade_control, flush_after_run=1)

            ship_speed = math.sqrt(ship_state["velocity"][0]**2 + ship_state["velocity"][1]**2) 
            evade_angle, intercept_angle, _ = find_intercept(closest_asteroid[0],closest_asteroid[1], ship_state, ship_speed)

            evade.input['rel_ast_angl'] = evade_angle
            evade.input['rel_ast_dist'] = closest_asteroid[1]
            evade.compute()
            thrust = evade.output['ship_thrust']
            turn_rate = evade.output['ship_turn']

            fire = False

            if danger_level > 8:
                fire = True
            else: 
                for asteroid in sorted_asteroids:
                    _, intercept_angle, _ = find_intercept(asteroid[0], asteroid[1], ship_state, ship_speed)
                    if abs(intercept_angle - ship_state['heading']) < 5:
                        fire = True
                
        else: 
            thrust = 0
            #shoot closest asteroid
            shooting_angle, intercept_angle, bullet_time = find_intercept(closest_asteroid[0], closest_asteroid[1], ship_state, bullet_speed)

            if shooting_angle != None: 
                target = ctrl.ControlSystemSimulation(self.targetting, flush_after_run = 1)
                target.input["bullet_time"] = bullet_time
                target.input["theta_delta"] = shooting_angle
                target.compute()
                turn_rate = target.output["ship_turn"]
            
                if target.output["ship_fire"] >= 0:
                    
                    if abs(intercept_angle - ship_state["heading"]) < 5:
                        fire = True
                    else:
                        fire = False
                else: 
                    fire = False
            else: 
                fire = False
                turn_rate = 0

        self.eval_frames +=1
        return thrust, turn_rate, fire

    @property
    def name(self) -> str:
        return "Group 7's Controller"
    


