import pickle
import math

def write_list_to_file(file_path, data_list):
    with open(file_path, 'wb') as file:
        pickle.dump(data_list, file)

def read_list_from_file(file_path):
    with open(file_path, 'rb') as file:
        data_list = pickle.load(file)
    return data_list

def distance(coords1, coords2):
    return math.sqrt((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2)

def relative_pos(asteroid, ship):

    asteroid_coords = asteroid["position"]
    ship_coords = ship["position"]

    asteroid_velocity = asteroid["velocity"]
    ship_velocity = ship["velocity"]

    rel_pos = [ship_coords[0] - asteroid_coords[0], ship_coords[1] - asteroid_coords[1]]
    rel_vel = [ship_velocity[0] - asteroid_velocity[0], ship_velocity[1] - asteroid_velocity[1]]
    future_ast_pos = [asteroid_coords[0] + rel_vel[0] * 1/30, asteroid_coords[1] + rel_vel[1] * 1/30]

    return distance(future_ast_pos, ship_coords), rel_pos, future_ast_pos

def find_intercept(asteroid, ast_distance, ship, object_speed):

    _, rel_pos, ast_pos = relative_pos(asteroid, ship)
    rel_angle = math.atan2(rel_pos[0], rel_pos[1])
    ast_vel_angle = math.atan2(asteroid["velocity"][1], asteroid["velocity"][0])
    ast_distance = ast_distance - (asteroid["size"]/2)

    theta = rel_angle - ast_vel_angle
    cos_theta = math.cos(theta)
    ast_speed = math.sqrt(asteroid["velocity"][0]**2 + asteroid["velocity"][1]**2)

    a = ast_speed**2 - object_speed**2
    b = -2 * ast_distance * ast_speed * cos_theta
    c = ast_distance

    determinant = abs(b**2 - (4*a*c))

    intercept_time = [(-b + math.sqrt(determinant))/(2*a*c), (-b - math.sqrt(determinant))/(2*a*c)]
    object_time = min(intercept_time)

    intercept_pos = [ast_pos[0] + asteroid["velocity"][0] * object_time, 
                     ast_pos[1] + asteroid["velocity"][1] * object_time]
    
    intercept_angle = math.atan2((intercept_pos[1] - ship["position"][1]), 
                                 (intercept_pos[0] - ship["position"][0]))  
    
    warped_intercept_angle = ((intercept_angle + 2*math.pi) % (2*math.pi)) * 180/math.pi           
    
    target_angle = intercept_angle - (math.pi/180) * ship["heading"] 
    target_angle = ((target_angle + math.pi) % (2* math.pi) - math.pi) * 180/math.pi

    return target_angle, warped_intercept_angle, object_time