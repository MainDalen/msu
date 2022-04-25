from itertools import combinations
from math import sqrt

class V(): # Вектор скорости
    def __init__(self, x, y):
        
        self.x = x
        self.y = y
    
    def __add__(self, vector):
        
        if isinstance(vector, V) or isinstance(vector, A):
            return V(self.x + vector.x, self.y + vector.y)
        else:
            raise Exception('Wrong addition')
    
    def __sub__(self, vector):

        if isinstance(vector, V):
            return V(self.x - vector.x, self.y - vector.y)
        elif isinstance(vector, A):
            return A(self.x - vector.x, self.y - vector.y)
        else:
            raise Exception('Wrong subtraction')
    
    def __mul__(self, v):

        if isinstance(v, V):
            return self.x * v.x + self.y * v.y
        elif isinstance(v, float) or isinstance(v, int):
            return V(self.x * v, self.y * v)
        else:
            raise  Exception('Wrong multiplication')

    def __str__(self):

        return f'V = {{{self.x}, {self.y}}}'

class A(): # Вектор ускорения
    def __init__(self, x, y):

        self.x = x
        self.y = y
    
    def __add__(self, vector):

        if isinstance(vector, V):
            return V(self.x + vector.x, self.y + vector.y)
        elif isinstance(vector, A):
            return A(self.x + vector.x, self.y + vector.y)
        else:
            raise Exception('Wrong addition')
    
    def __sub__(self, vector):

        if isinstance(vector, V):
            return V(self.x - vector.x, self.y - vector.y)
        elif isinstance(vector, A):
            return A(self.x - vector.x, self.y - vector.y)
        else:
            raise Exception('Wrong subtraction')
    
    def __mul__(self, a):

        if isinstance(a, A):
            return self.x * a.x + self.y * a.y
        elif isinstance(a, float) or isinstance(a, int):
            return V(self.x * a, self.y * a)
        else:
            raise  Exception('Wrong multiplication')
        
    def __str__(self):

        return f'A = {{{self.x}, {self.y}}}'

class particle(): # частица

    v = V(0, 0)
    a = A(0, 0)
    q = 0
    r = 0

    def __init__(self, x, y, m):

        self.x = x
        self.y = y
        self.m = 1
    
    def __str__(self):

        return f"""
        x = {self.x}\n
        y = {self.y}\n
        m = {self.m}\n
        {self.v}\n
        {self.a}\n
        """

def potential(p, k=1):
    Fx, Fy = -k*p.x, -k*p.y
    Ax, Ay = Fx/p.m, Fy/p.m
    return A(Ax, Ay)


def kulon(p1, p2):
    k = 9 * 10**9
    r = sqrt( (p1.x - p2.x)**2 + (p1.y - p2.y)**2 )
    F = (k * p1.q * p2.q) / (p1.m * r**3)
    # return (p1.a - p2.a) * F
    return A(p1.x - p2.x, p1.y - p2.y) * F

def create_walls(size):
    return {
        'xmin': -size / 2,
        'xmax': size / 2,
        'ymin': -size / 2,
        'ymax': size / 2
    }

def bounce(p, xmin, xmax, ymin, ymax):

    if p.x <= xmin:
        p.v.x = -p.v.x
        p.x = p.x - 2 * (p.x - xmin)
    
    if p.x >= xmax:
        p.v.x = -p.v.x
        p.x = p.x - 2 * (p.x - xmax)
    
    if p.y <= ymin:
        p.v.y = -p.v.y
        p.y = p.y - 2 * (p.y - ymin)
    
    if p.y >= ymax:
        p.v.y = -p.v.y
        p.y = p.y - 2 * (p.y - ymax)

def tdt(dt, parts, walls, k):
    for p in parts:
        p.a = potential(p, k)
        bounce(p, **walls)
    
    for z in combinations(parts, 2):
        z[0].a += kulon(z[0], z[1])
        z[1].a += kulon(z[1], z[0])

    for p in parts:
        p.x += (p.v.x * dt) + ((p.a.x * dt**2) / 2)
        p.y += (p.v.y * dt) + ((p.a.y * dt**2) / 2)
        p.v += (p.a * dt)