# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick

# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
#   detailed discussion of this source code.

from src.kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt
from ivan_defs import relative_pos,time_to_col
from src.kesslergame.asteroid import Asteroid
import pdb


class IvanController(KesslerController):
    
        
    def __init__(self):
        self.eval_frames = 0 #What is this?

        # self.targeting_control is the targeting rulebase, which is static in this controller.      
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi,math.pi,0.1), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        
        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)
        
        #Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/3,-1*math.pi/6)
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/3,-1*math.pi/6,0])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/6,0,math.pi/6])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [0,math.pi/6,math.pi/3])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,math.pi/6,math.pi/3)
        
        #Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-90,-60])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [60,90,120])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])
        
        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1]) 
                
        #Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['N']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))   

        rule6 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))    
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        
        rule11 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule14 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))
     
        #DEBUG
        #bullet_time.view()
        #theta_delta.view()
        #ship_turn.view()
        #ship_fire.view()

        # Declare the fuzzy controller, add the rules 
        # This is an instance variable, and thus available for other methods in the same object. See notes.                         
        # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
             
        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule1)
        self.targeting_control.addrule(rule2)
        self.targeting_control.addrule(rule3)
        self.targeting_control.addrule(rule4)
        self.targeting_control.addrule(rule5)
        self.targeting_control.addrule(rule6)
        self.targeting_control.addrule(rule7)
        self.targeting_control.addrule(rule8)
        self.targeting_control.addrule(rule9)
        self.targeting_control.addrule(rule10)
        self.targeting_control.addrule(rule11)
        self.targeting_control.addrule(rule12)
        self.targeting_control.addrule(rule13)
        self.targeting_control.addrule(rule14)
        self.targeting_control.addrule(rule15)

        #Start of Ivan's Fuzzy Logic for Evade Mode
        rel_ast_dist = ctrl.Antecedent(np.arange(0,700,1),'rel_ast_dist')
        rel_ast_angl = ctrl.Antecedent(np.arange(-180,180,1), 'rel_ast_angl')
        #t_to_col = ctrl.Antecedent(np.arange(0,10,11),'t_col')
        ship_thrust = ctrl.Consequent(np.arange(0,200,1),'ship_thrust')

        rel_ast_dist['NL'] = fuzz.zmf(rel_ast_dist.universe,0,100)
        rel_ast_dist['NS'] = fuzz.trimf(rel_ast_dist.universe, [50,150,275])
        rel_ast_dist['Z'] = fuzz.trimf(rel_ast_dist.universe, [225,350,475])
        rel_ast_dist['PS'] = fuzz.trimf(rel_ast_dist.universe, [425,550,650])
        rel_ast_dist['PL'] = fuzz.smf(rel_ast_dist.universe,600,700)

        rel_ast_angl['NL'] = fuzz.trimf(rel_ast_angl.universe, [-180,-180,-120])
        rel_ast_angl['NS'] = fuzz.trimf(rel_ast_angl.universe, [-120,-90,-60])
        rel_ast_angl['Z'] = fuzz.trimf(rel_ast_angl.universe, [-60,0,60])
        rel_ast_angl['PS'] = fuzz.trimf(rel_ast_angl.universe, [60,90,120])
        rel_ast_angl['PL'] = fuzz.trimf(rel_ast_angl.universe, [120,180,180])
          
        ship_thrust['NL'] = fuzz.zmf(ship_thrust.universe,0,50)
        ship_thrust['NS'] = fuzz.trimf(ship_thrust.universe, [25,50,75])
        ship_thrust['Z'] = fuzz.trimf(ship_thrust.universe, [50,75,100])
        ship_thrust['PS'] = fuzz.trimf(ship_thrust.universe, [75,100,125])
        ship_thrust['PL'] = fuzz.smf(ship_thrust.universe,100,200)

        #Declare each fuzzy rule for the Evade Mode
        rule1_evade = ctrl.Rule((rel_ast_angl['NL'] & rel_ast_dist['NL']), (ship_turn['PS'], ship_thrust['PL']))
        rule2_evade= ctrl.Rule((rel_ast_angl['NL'] & rel_ast_dist['NS']), (ship_turn['PS'], ship_thrust['PL']))
        rule3_evade = ctrl.Rule((rel_ast_angl['NL'] & rel_ast_dist['Z']), (ship_turn['Z'], ship_thrust['PS']))
        rule4_evade = ctrl.Rule((rel_ast_angl['NL'] & rel_ast_dist['PS']), (ship_turn['Z'], ship_thrust['PS']))
        rule5_evade = ctrl.Rule((rel_ast_angl['NL'] & rel_ast_dist['PL']), (ship_turn['Z'], ship_thrust['Z']))

        rule6_evade = ctrl.Rule((rel_ast_angl['NS'] & rel_ast_dist['NL']), (ship_turn['PL'], ship_thrust['PL']))
        rule7_evade = ctrl.Rule((rel_ast_angl['NS'] & rel_ast_dist['NS']), (ship_turn['PS'], ship_thrust['PS']))
        rule8_evade = ctrl.Rule((rel_ast_angl['NS'] & rel_ast_dist['Z']), (ship_turn['PS'], ship_thrust['Z']))
        rule9_evade = ctrl.Rule((rel_ast_angl['NS'] & rel_ast_dist['PS']), (ship_turn['Z'], ship_thrust['NS']))
        rule10_evade = ctrl.Rule((rel_ast_angl['NS'] & rel_ast_dist['PL']), (ship_turn['Z'], ship_thrust['NS']))

        rule11_evade = ctrl.Rule((rel_ast_angl['Z'] & rel_ast_dist['NL']), (ship_turn['PL'], ship_thrust['NL']))
        rule12_evade = ctrl.Rule((rel_ast_angl['Z'] & rel_ast_dist['NS']), (ship_turn['PL'], ship_thrust['NL']))
        rule13_evade = ctrl.Rule((rel_ast_angl['Z'] & rel_ast_dist['Z']), (ship_turn['PL'], ship_thrust['NL']))
        rule14_evade = ctrl.Rule((rel_ast_angl['Z'] & rel_ast_dist['PS']), (ship_turn['PS'], ship_thrust['NL']))
        rule15_evade = ctrl.Rule((rel_ast_angl['Z'] & rel_ast_dist['PL']), (ship_turn['PS'], ship_thrust['NL']))

        rule16_evade = ctrl.Rule((rel_ast_angl['PS'] & rel_ast_dist['NL']), (ship_turn['NL'], ship_thrust['PL']))
        rule17_evade = ctrl.Rule((rel_ast_angl['PS'] & rel_ast_dist['NS']), (ship_turn['NS'], ship_thrust['PS']))
        rule18_evade = ctrl.Rule((rel_ast_angl['PS'] & rel_ast_dist['Z']), (ship_turn['NS'], ship_thrust['Z']))
        rule19_evade = ctrl.Rule((rel_ast_angl['PS'] & rel_ast_dist['PS']), (ship_turn['Z'], ship_thrust['NS']))
        rule20_evade = ctrl.Rule((rel_ast_angl['PS'] & rel_ast_dist['PL']), (ship_turn['Z'], ship_thrust['NS']))

        rule21_evade = ctrl.Rule((rel_ast_angl['PL'] & rel_ast_dist['NL']), (ship_turn['NS'], ship_thrust['PL']))
        rule22_evade = ctrl.Rule((rel_ast_angl['PL'] & rel_ast_dist['NS']), (ship_turn['NS'], ship_thrust['PL']))
        rule23_evade = ctrl.Rule((rel_ast_angl['PL'] & rel_ast_dist['Z']), (ship_turn['Z'], ship_thrust['PS']))
        rule24_evade = ctrl.Rule((rel_ast_angl['PL'] & rel_ast_dist['PS']), (ship_turn['Z'], ship_thrust['PS']))
        rule25_evade = ctrl.Rule((rel_ast_angl['PL'] & rel_ast_dist['PL']), (ship_turn['Z'], ship_thrust['Z']))

        #rel_ast_angl.view()
        #rel_ast_dist.view()
        #ship_turn.view()
        #ship_thrust.view()
        #pdb.set_trace()

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
        

        

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        """
        Method processed each time step by this controller.
        """
        # These were the constant actions in the basic demo, just spinning and shooting.
        #thrust = 0 <- How do the values scale with asteroid velocity vector?
        #turn_rate = 90 <- How do the values scale with asteroid velocity vector?
        
        # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
        # So are the ship position and velocity, and bullet position and velocity. 
        # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
        # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded in many places.
        # So, position is updated by multiplying velocity by delta_time, and adding that to position.
        # Ship velocity is updated by multiplying thrust by delta time.
        # Ship position for this time increment is updated after the the thrust was applied.
        

        # My demonstration controller does not move the ship, only rotates it to shoot the nearest asteroid.
        # Goal: demonstrate processing of game state, fuzzy controller, intercept computation 
        # Intercept-point calculation derived from the Law of Cosines, see notes for details and citation.

        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]       
        closest_asteroid = None
        asteroid_distances = []

        for a in game_state["asteroids"]:

            #Loop through all asteroids, find minimum Eudlidean distance
            relative = relative_pos(a["position"][0], a["position"][1], a["velocity"][0], a["velocity"][1], ship_state["position"][0], ship_state["position"][1], ship_state["velocity"][0], ship_state["velocity"][1],a["radius"])
            t = time_to_col(a["position"][0], a["position"][1], a["velocity"][0], a["velocity"][1], ship_state["position"][0], ship_state["position"][1], ship_state["velocity"][0], ship_state["velocity"][1])

            asteroid_distances.append((a, relative[0], relative[1], t))

            if relative[0] > 500: 
                continue

            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = relative[0],dot = relative[1], ti = t)  
            else:    
                # closest_asteroid exists, and is thus initialized.
                if ((closest_asteroid["dist"] > relative[0]) & (relative[1] <= 0)):
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = relative[0]
                    closest_asteroid["dot"] = relative[1]
                    closest_asteroid["ti"] = t

        k = int(len(game_state["asteroids"])/4)
        sorted_asteroids = sorted(asteroid_distances, key = lambda x:x[1])
        closest_asteroids = sorted_asteroids[0:k]

        if closest_asteroid == None: 
            closest_asteroid = dict(aster = closest_asteroids[0][0], dist = closest_asteroids[0][1], dot = closest_asteroids[0][2], ti = closest_asteroids[0][3])

        # closest_asteroid is now the nearest asteroid object. 
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.
        #print(closest_asteroid["dist"], closest_asteroid["dot"])
        danger_low = ((closest_asteroid["dist"] + closest_asteroid["aster"]["radius"])<200)
        danger_medium = ((closest_asteroid["dist"] + closest_asteroid["aster"]["radius"])<150)
        danger_high = ((closest_asteroid["dist"] + closest_asteroid["aster"]["radius"])<100)
 
        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!
        
        def shooting_params(closest_asteroid): 
            asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
            asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
            
            asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
            
            asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
            my_theta2 = asteroid_ship_theta - asteroid_direction
            cos_my_theta2 = math.cos(my_theta2)
            # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
            asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
            bullet_speed = 800 # Hard-coded bullet speed from bullet.py
            
            # Determinant of the quadratic formula b^2-4ac
            targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * closest_asteroid["dist"])
            
            # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
            intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
            intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))

            # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
            if intrcpt1 > intrcpt2:
                if intrcpt2 >= 0:
                    bullet_t = intrcpt2
                else:
                    bullet_t = intrcpt1
            else:
                if intrcpt1 >= 0:
                    bullet_t = intrcpt1
                else:
                    bullet_t = intrcpt2
                    
            # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
            intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * bullet_t
            intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * bullet_t
            
            my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
            
            # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
            shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
            
            # Wrap all angles to (-pi, pi)
            shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

            return (shooting_theta, bullet_t)

        if(danger_low):
            if(danger_medium):
                if(danger_high):
                    print("DangerHigh")
                else:
                    print("DangerMedium")
            else:
                print("DangerLow")        


        shooting_param = shooting_params(closest_asteroid)
        shooting_theta = shooting_param[0]
        bullet_t = shooting_param[1]
        
        # Pass the inputs to the rulebase and fire it
        if (danger_high):
            evade = ctrl.ControlSystemSimulation(self.evade_control,flush_after_run=1)
            print("Evade Mode")
            evade.input['rel_ast_angl'] = ((180/math.pi)*shooting_theta)
            evade.input['rel_ast_dist'] = closest_asteroid["dist"]
            evade.compute()
            thrust = evade.output['ship_thrust']
            turn_rate = evade.output['ship_turn']
            current_orien = (math.pi/180)*ship_state["heading"]

            #check if current turn angle will hit an asteroid
            for a in closest_asteroids:
                Aster_shooting_param = shooting_params(dict(aster = a[0], dist = a[1], dot = a[2], ti = a[3]))
                shooting_theta = Aster_shooting_param[0]

                if abs(current_orien - shooting_theta) < 0.5:
                    fire = True  
                    break
                else: 
                    fire = False

        else:
            print("Shooting")
            shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)
            shooting.input['bullet_time'] = bullet_t
            shooting.input['theta_delta'] = shooting_theta
            shooting.compute()
            thrust = 0
            turn_rate = shooting.output['ship_turn']

            if shooting.output['ship_fire'] >= 0:
                fire = True
            else:
                fire = False  

        
        self.eval_frames +=1
        
        #DEBUG
        #print(thrust, bullet_t, shooting_theta, turn_rate, fire)
        
        return thrust, turn_rate, fire

    @property
    def name(self) -> str:
        return "Tester Controller"