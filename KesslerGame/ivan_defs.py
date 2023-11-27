from cmath import sqrt
import math
import numpy as np


def calc_dist(x1, y1, x2, y2):
    return (math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
        
def relative_pos(ast_x, ast_y, ast_v_x, ast_v_y, space_x, space_y, space_v_x, space_v_y):
    # Calculate relative position vector
    relative_pos_x = ast_x - space_x
    relative_pos_y = ast_y - space_y
    # Calculate relative velocity components
    relative_vel_x = ast_v_x - space_v_x        
    relative_vel_y = ast_v_y - space_v_y

    #Return dot product
    dot_product = relative_pos_x * relative_vel_x + relative_pos_y * relative_vel_y
    # Predict future position of the asteroid
    future_ast_x = ast_x + relative_vel_x*1/30
    future_ast_y = ast_y + relative_vel_y*1/30

    # Calculate Euclidean distance between current spaceship and predicted asteroid positions
    distance = calc_dist(future_ast_x, future_ast_y, space_x, space_y)
    return (distance,dot_product)


class QuadTree:
    def __init__(self, x, y, width, height, capacity):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.capacity = capacity
        self.objects = []
        self.children = []

    def insert(self, asteroid):
        if len(self.children) == 0 and len(self.objects) < self.capacity:
            self.objects.append(asteroid)
        else:
            if len(self.children) == 0:
                self.split()

            for child in self.children:
                if self.check_collision(asteroid, child):
                    child.insert(asteroid)
                    break

    def split(self):
        half_width = self.width / 2
        half_height = self.height / 2

        self.children.append(QuadTree(self.x, self.y, half_width, half_height, self.capacity))
        self.children.append(QuadTree(self.x + half_width, self.y, half_width, half_height, self.capacity))
        self.children.append(QuadTree(self.x, self.y + half_height, half_width, half_height, self.capacity))
        self.children.append(QuadTree(self.x + half_width, self.y + half_height, half_width, half_height, self.capacity))

        for obj in self.objects:
            for child in self.children:
                if self.check_collision(obj, child):
                    child.insert(obj)
                    break

        self.objects = []

    def check_collision(self, obj, quad):
        return (
            obj.position[0] + obj.radius > quad.x and
            obj.position[0] - obj.radius < quad.x + quad.width and
            obj.position[1] + obj.radius > quad.y and
            obj.position[1] - obj.radius < quad.y + quad.height
        )

    def get_safest_region(self):
        if len(self.children) == 0:
            return self
        else:
            safest_child = min(self.children, key=lambda child: len(child.objects))
            return safest_child.get_safest_region()

