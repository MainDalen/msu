import numpy as np
from tqdm import tqdm
import os
import pandas as pd

# KB = 1.380649 * (10 ** -16) # ЭРГ/К
KB = 1 # ЭРГ/К
# M = 10 ** -24 # г
M = 1 # г
# A = 10 ** -8 # Ангстрем -> см
A = 1 # Ангстрем -> м

ELEMENTS = {
    'He': {
        'eps': 6.03 * KB,
        'sig': 2.63 * A,
        'm': 6.67 * M,
        'r': 1.4 * A
    },
    'H2': {
        'eps': 29.2 * KB,
        'sig': 2.87 * A,
        'm': 1.71 * M,
        'r': 1.2 * A
    },
    'Ne': {
        'eps': 35.6 * KB,
        'sig': 2.75 * A,
        'm': 33.55 * M,
        'r': 1.5 * A
    },
    'N2': {
        'eps': 95.05 * KB,
        'sig': 3.69 * A,
        'm': 46.43 * M,
        'r': 1.6 * A
    },
    'O2': {
        'eps': 99.2 * KB,
        'sig': 3.52 * A,
        'm': 26.68 * M,
        'r': 1.5 * A
    }
}

def generate_points_with_min_distance(n, size, min_dist):
    # compute grid shape based on number of points
    size = size - 2 * min_dist
    num_y = np.int32(np.sqrt(n))
    num_x = np.int32(n / num_y) + 1

    # create regularly spaced neurons
    x = np.linspace(-size, size, num_x, dtype=np.float32)
    y = np.linspace(-size, size, num_y, dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1,2)[0:n]

    # compute spacing
    init_dist = np.min((x[1]-x[0], y[1]-y[0]))

    # perturb points
    max_movement = (init_dist - min_dist)/2
    noise = np.random.uniform(low=-max_movement,
                                high=max_movement,
                                size=(len(coords), 2))
    coords += noise

    return coords

def writeOutput(filename, natoms, timestep, box, **data):
    """ Writes the output (in dump format) """
    
    axis = ('x', 'y', 'z')
    
    with open(filename, 'a') as fp:
        
        fp.write('ITEM: TIMESTEP\n')
        fp.write('{}\n'.format(timestep))
        
        fp.write('ITEM: NUMBER OF ATOMS\n')
        fp.write('{}\n'.format(natoms))
        
        fp.write('ITEM: BOX BOUNDS' + ' f' * len(box) + '\n')
        for box_bounds in box:
            fp.write('{} {}\n'.format(*box_bounds))

        for i in range(len(axis) - len(box)):
            fp.write('0 0\n')
            
        keys = list(data.keys())
        
        for key in keys:
            isMatrix = len(data[key].shape) > 1
            
            if isMatrix:
                _, nCols = data[key].shape
                
                for i in range(nCols):
                    if key == 'pos':
                        data['{}'.format(axis[i])] = data[key][:,i]
                    else:
                        data['{}_{}'.format(key,axis[i])] = data[key][:,i]
                        
                del data[key]
                
        keys = data.keys()
        
        fp.write('ITEM: ATOMS' + (' {}' * len(data)).format(*data) + '\n')
        
        output = []
        for key in keys:
            output = np.hstack((output, data[key]))
            
        if len(output):
            np.savetxt(fp, output.reshape((natoms, len(data)), order='F'))


def wallHitCheck(pos, vels, box, element):
    """ Эта функция обеспечивает выполнение отражающих граничных условий.
    Все частицы, которые ударяются о стену, обновляют свою скорость
    в противоположном направлении.
    @pos: позиции молекул (ndarray)
    @vels: скорости частиц (ndarray)
    @box: размер бокса (tuple)
    """
    ndims = len(box)
    radius = element['r']

    for i in range(ndims):
        vels[((pos[:,i] - radius <= box[i][0]) | (pos[:,i] + radius >= box[i][1])),i] *= -1
        pos[pos[:,i] - radius <= box[i][0],i] = box[i][0] + radius
        pos[pos[:,i] + radius >= box[i][1],i] = box[i][1] - radius



def integrate_old(pos, vels, forces, mass,  dt):
    """Метод Верле, который перемещает систему во времени
    @pos: позиции частиц (ndarray)
    @vels: скорость частицы (ndarray)
    """
    pos += vels * dt + forces * dt * dt / mass / 2 
    vels += forces * dt / mass

def integrate(pos, vels, element, dt, k):
    force  = computeForce(pos, vels, element, k)
    half_vels = vels + force * dt / (2 * element['m'])
    pos += half_vels * dt
    force2 = computeForce(pos, vels, element, k)
    vels = half_vels + force2 * dt / (2 * element['m'])

def computeForce_old(pos, vels, charge, k):
    """Вычисляет силы Кулона, действующие на каждую частицу
    @vels: скорости частиц (ndarray)
    @charge: заряд (float)
    @k: коэффициент пропорцональности (float)
    возвращает силы (ndarray)
    """

    natoms, ndims = vels.shape
    forces = np.zeros(vels.shape)

    for i in range(natoms): # для каждой частицы i
        for j in range(natoms): # перебираем все частицы кроме выбранной j
            if j == i:
                continue
            Rij = np.array(pos[i,:]-pos[j,:], dtype=np.double) # радиус вектор от частицы j к частице i
            Rij_module = np.sqrt(np.dot(Rij,Rij)) # модуль радиус вектора
            forces[i] += k * charge * charge * Rij / np.power(Rij_module, 3) # вычисление силы, действующей со стороны частицы j на частицу i
    return forces

def computeForce(pos, vels, element, k):

    natoms, ndims = vels.shape
    forces = np.zeros(vels.shape)

    for i in range(natoms): # для каждой частицы i
        for j in range(natoms): # перебираем все частицы кроме выбранной j
            if j == i:
                continue
            Rij = pos[i,:]-pos[j,:] # радиус вектор от частицы j к частице i
            Rij = np.array(Rij, dtype=np.float64)
            # Rij *= 10**-4
            Rij_module = np.sqrt(np.dot(Rij,Rij)) # модуль радиус вектора
            forces[i] += k * 24 * element['eps'] * (2 * (element['sig']**12) - (element['sig']**6)) * Rij / (Rij_module**14)
    return forces
  
def run(**args):
    """Это основная функция, которая решает уравнения движения для
    системы молекул, использующая алгоритм Верле, и записывает в файл положения частиц
    в разные моменты времени

    @natoms (int): количество частиц
    @mass (float): масса частиц
    @charge (float): заряд частиц
    @dt (float): временной шаг моделирования
    @nsteps (int): общее количество шагов, выполняемых решателем
    @box (tuple): размер коробки для моделирования
    @ofname (str): имя файла для записи выходных данных в
    @freq (int): запись выходных данных через freg шагов
    @[radius]: радиус частицы (для визуализации)
    возвращает состояния частиц в каждый шаг времени
    """

    natoms, box, dt = args['natoms'], args['box'], args['dt']
    element, k, nsteps   = args['element'], args['k'], args['steps']
    ofname, freq, T = args['ofname'], args['freq'], args['T']

    radius = element['r']
    
    dim = len(box)
    # pos = np.random.rand(natoms,dim)

    # for i in range(dim):
        # pos[:,i] = box[i][0] + (box[i][1] -  box[i][0]) * pos[:,i]

    pos = generate_points_with_min_distance(natoms, box[0][1], 2 * element['r'] + element['sig'])

    vels = np.array(np.random.rand(natoms, dim) - 0.5, dtype=np.float64)
    vels = vels * np.sqrt(natoms * KB * T / element['m'])


    wallHitCheck(pos, vels, box, element)

    radius = np.ones(natoms) * radius

    # while step <= nsteps:
    for step in tqdm(range(nsteps+1)):

        # Перенос системы во времени
        integrate(pos, vels, element, dt, k)

        # Проверка столкновения частиц со стеной
        wallHitCheck(pos, vels, box, element)

        # Запись состояний в файл
        if not step%freq:
            writeOutput(ofname, natoms, step, box, radius=radius, pos=pos, v=vels)
        
        # backup = pd.DataFrame(index=np.ones(natoms)*step)
        # backup['id'] = range(natoms)
        # backup[['x', 'y']] = pos
        # backup[['vx', 'vy']] = vels
        # backup[['fx', 'fy']] = forces
        # back_path = 'backup.csv'
        # if not os.path.exists(back_path):
        #     backup.to_csv(back_path)
        # else:
        #     backup.to_csv(back_path, mode='a', header=False)


if __name__ == '__main__':

    element = ELEMENTS['He']
    size = element['r'] * 50
    box = ((-size, size), (-size, size))
    name_2d = 'MD_2D.dump'
    params = {
        'steps': 1000, # количество шагов
        'natoms': 100, # количество атомов (частиц)
        'element': element,
        'box': box,
        'T': 300,
        'dt': 0.005, # временной шаг
        'k': 50000, # коэф. пропорциональности
        'freq': 1, # через какое число шагов вести запись состояний
        'ofname': name_2d
        }
    if os.path.exists(name_2d):
        os.remove(name_2d)
    if os.path.exists('backup.csv'):
        os.remove('backup.csv')
    run(**params)