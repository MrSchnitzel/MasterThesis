import numpy as np
import math
class vec(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def findClockwiseAngle(self, other):
       # using cross-product formula
       return -math.degrees(math.asin((self.a * other.b - self.b * other.a)/(self.length()*other.length())))
       # the dot-product formula, left here just for comparison (does not return angles in the desired range)
       # return math.degrees(math.acos((self.a * other.a + self.b * other.b)/(self.length()*other.length())))

    def length(self):
       return math.sqrt(self.a**2 + self.b**2)

    def unit_vector(self,vector):
        """ Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    def angle_between(self,v1, v2):
        """Finds angle between two vectors"""
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        # return np.arctan2(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def x_rotation(self,vector,theta):
        """Rotates 3-D vector around x-axis"""
        R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
        return np.dot(R,vector)

    def y_rotation(self,vector,theta):
        """Rotates 3-D vector around y-axis"""
        R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
        return np.dot(R,vector)

    def z_rotation(self,vector,theta):
        """Rotates 3-D vector around z-axis"""
        R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
        return np.dot(R,vector)

    def findClockwiseAngle180(self, other):
       # using cross-product formula
    #    return -math.degrees(math.asin((self.a * other.b - self.b * other.a)/(self.length()*other.length())))
       # the dot-product formula, left here just for comparison (does not return angles in the desired range)
       angle = math.degrees(math.acos((self.a * other.a + self.b * other.b)/(self.length()*other.length())))
       det = self.determinant(other)
       if det > 0:
           angle = -angle
       return angle
    
    def determinant(self,w):
        return self.a*w.b-self.b*w.a