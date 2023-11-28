# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick

# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
#   detailed discussion of this source code.

from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt
from ivan_defs import relative_pos,time_to_col
from kesslergame.asteroid import Asteroid


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
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-30])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-90,-30,0])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-30,0,30])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [0,30,90])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [30,180,180])
        
        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1]) 
                
        #Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
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
        rel_ast_dist = ctrl.Antecedent(np.arange(0,700,701),'ast_d')
        rel_ast_angl = ctrl.Antecedent(np.arange(-1*math.pi,math.pi,0.1), 'ast_a')
        t_to_col = ctrl.Antecedent(np.arange(0,10,11),'t_col')
        ship_thrust = ctrl.Consequent(np.arange(0,450,451),'thrust')

        rel_ast_dist['NL'] = fuzz.zmf(rel_ast_dist.universe,0,100)
        rel_ast_dist['NS'] = fuzz.trimf(rel_ast_dist.universe, [50,150,250])
        rel_ast_dist['Z'] = fuzz.trimf(rel_ast_dist.universe, [250,350,450])
        rel_ast_dist['PS'] = fuzz.trimf(rel_ast_dist.universe, [450,550,650])
        rel_ast_dist['PL'] = fuzz.smf(rel_ast_dist.universe,600,700)

        rel_ast_angl['NL'] = fuzz.zmf(rel_ast_angl.universe, -1*math.pi/3,-1*math.pi/6)
        rel_ast_angl['NS'] = fuzz.trimf(rel_ast_angl.universe, [-1*math.pi/3,-1*math.pi/6,0])
        rel_ast_angl['Z'] = fuzz.trimf(rel_ast_angl.universe, [-1*math.pi/6,0,math.pi/6])
        rel_ast_angl['PS'] = fuzz.trimf(rel_ast_angl.universe, [0,math.pi/6,math.pi/3])
        rel_ast_angl['PL'] = fuzz.smf(rel_ast_angl.universe,math.pi/6,math.pi/3)
        
        t_to_col['S'] = fuzz.trimf(t_to_col.universe,[0,0,2])
        t_to_col['M'] = fuzz.trimf(t_to_col.universe, [1,3,5])
        t_to_col['L'] = fuzz.smf(t_to_col.universe,5,7)
          
        ship_thrust['NL'] = fuzz.zmf(ship_thrust.universe,0,100)
        ship_thrust['NS'] = fuzz.trimf(ship_thrust.universe, [50,150,200])
        ship_thrust['Z'] = fuzz.trimf(ship_thrust.universe, [175,250,300])
        ship_thrust['PS'] = fuzz.trimf(ship_thrust.universe, [275,350,400])
        ship_thrust['PL'] = fuzz.smf(ship_thrust.universe,375,400)

        #Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        #theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/3,-1*math.pi/6)
        #theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/3,-1*math.pi/6,0])
        #theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/6,0,math.pi/6])
        #theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [0,math.pi/6,math.pi/3])
        #theta_delta['PL'] = fuzz.smf(theta_delta.universe,math.pi/6,math.pi/3)
        
        #Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        #ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-30])
        #ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-90,-30,0])
        #ship_turn['Z'] = fuzz.trimf(shi                #Declare each fuzzy rulep_turn.universe, [-30,0,30])
        #ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [& rel_ast_angl['NL']0,30,90])
        #ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [& rel_ast_angl['NS']30,180,180])

                #Declare each fuzzy rule& rel_ast_angl['PS']
        #rule1 = ctrl.Rule(t_to_col['L'] & rel_ast_angl['NL'] & rel_ast_dist['PL'] 
        #rule2 = ctrl.Rule(t_to_col['L'] & rel_ast_angl['NS'] & rel_ast_dist['NL']
        #rule3 = ctrl.Rule(t_to_col['L'] & rel_ast_angl['Z'] & rel_ast_dist['NS']
        #rule4 = ctrl.Rule(t_to_col['L'] & rel_ast_angl['PS'] & rel_ast_dist['Z']
        #rule5 = ctrl.Rule(t_to_col['L'] & rel_ast_angl['PL'] & rel_ast_dist['PS']  
        #rule6 = ctrl.Rule(t_to_col['M'] & rel_ast_angl['NL'] & rel_ast_dist['PL']
        #rule7 = ctrl.Rule(t_to_col['M'] & rel_ast_angl['NS'] & rel_ast_dist['NL']
        #rule8 = ctrl.Rule(t_to_col['M'] & rel_ast_angl['Z']  & rel_ast_dist['NS']
        #rule9 = ctrl.Rule(t_to_col['M'] & rel_ast_angl['PS'] & rel_ast_dist['Z']
        #rule10 = ctrl.Rule(t_to_col['M'] & rel_ast_angl['PL'] & rel_ast_dist['PS']
        #rule11 = ctrl.Rule(t_to_col['S'] & rel_ast_angl['NL'] & rel_ast_dist['PL']
        #rule12 = ctrl.Rule(t_to_col['S'] & rel_ast_angl['NS'] & rel_ast_dist['PL']              #Declare each fuzzy rule
        #rule13 = ctrl.Rule(t_to_col['S'] & rel_ast_angl['Z'] & rel_ast_dist['PL']
        #rule14 = ctrl.Rule(t_to_col['S'] & rel_ast_angl['PS'] & rel_ast_dist['PL']
        #rule15 = ctrl.Rule(t_to_col['S'] & rel_ast_angl['PL'] & rel_ast_dist['PL']


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
        
        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            relative = relative_pos(a["position"][0], a["position"][1], a["velocity"][0], a["velocity"][1], ship_state["position"][0], ship_state["position"][1], ship_state["velocity"][0], ship_state["velocity"][1])
            t = time_to_col(a["position"][0], a["position"][1], a["velocity"][0], a["velocity"][1], ship_state["position"][0], ship_state["position"][1], ship_state["velocity"][0], ship_state["velocity"][1])
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = relative[0],dot = relative[1], ti = t)  
            else:    
                # closest_asteroid exists, and is thus initialized.
                if((round(t[0], 2) == round(t[1], 2)) & (closest_asteroid["ti"][0] > t[0])):
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = relative[0]
                    closest_asteroid["dot"] = relative[1]
                    closest_asteroid["ti"] = t
                elif ((closest_asteroid["dist"] > relative[0]) & (relative[1] <= 0)):
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = relative[0]
                    closest_asteroid["dot"] = relative[1]
                    closest_asteroid["ti"] = t

        # closest_asteroid is now the nearest asteroid object. 
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.
        print(closest_asteroid["dist"], closest_asteroid["dot"])
        danger_low = ((closest_asteroid["dist"] + closest_asteroid["aster"]["radius"])<200)
        danger_medium = ((closest_asteroid["dist"] + closest_asteroid["aster"]["radius"])<150)
        danger_high = ((closest_asteroid["dist"] + closest_asteroid["aster"]["radius"])<100)
 
        if(danger_low):
            if(danger_medium):
                if(danger_high):
                    print("DangerHigh")
                    print(closest_asteroid["ti"])
                else:
                    print("DangerMedium")
            else:
                print("DangerLow")
        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!
        
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

        #Speed up the theta angle
        if (shooting_theta <= 0):
          shooting_theta = shooting_theta-((math.pi/180)*10)
        else:
           shooting_theta = shooting_theta+((math.pi/180)*10)

        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        shooting.compute()
        thrust = 0
        # Get the defuzzified outputs
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