import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from datetime import datetime

# KB = 1.380649 * (10 ** -16) # erg/k
KB = 1 # erg/k
# M = 10 ** -24 # g
M = 1 # g
# A = 10 ** -8 # Angstrem -> cm
A = 1 # Angstrem -> m

ELEMENTS = {
    'He': {
        'eps': 6.03 * KB,
        'sig': 2.63 * A,
        # 'sig': 3.0 * A,
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

def wallHitCheck(pos, box):
    ndims = len(box)

    for i in range(ndims):
        teleport = box[i][1] - box[i][0]
        pos[pos[:,i] <= box[i][0], i] += teleport
        pos[pos[:,i] >= box[i][1], i] -= teleport

def integrate(pos, vels, element, dt, k, T=None, Tt=None):
    force  = computeForce(pos, vels, element, size, k)
    half_vels = vels + force * dt / (2 * element['m'])
    pos += half_vels * dt
    force2 = computeForce(pos, vels, element, size, k)
    vels = half_vels + force2 * dt / (2 * element['m'])
    return vels

def computeForce(pos, vels, element, size, k, return_e_pot = False):

    natoms, ndims = vels.shape
    forces = np.zeros(vels.shape)
    E_pot = np.zeros(natoms)
    size2 = 2 * size
    teleports = [
        np.array([     0,      0]),
        np.array([ size2,      0]),
        np.array([-size2,      0]),
        np.array([     0,  size2]),
        np.array([     0, -size2]),
        ]
    for i in range(natoms):
        for j in range(natoms):
            if j == i:
                continue
            for tlprt in teleports:
                Rij = pos[i,:]-(pos[j,:] + tlprt)
                Rij_module = np.sqrt(np.dot(Rij,Rij))
                if Rij_module < 2.5 * element['sig']:
                    E_pot[i] += k * 24 * element['eps'] * ((2 * (element['sig'] / Rij_module)**12) - ((element['sig'] / Rij_module)**6))
                    forces[i] += E_pot[i] * Rij / Rij_module**2
    if return_e_pot:
        return forces, E_pot
    else:
        return forces
  
def run(**args):

    natoms, box, dt = args['natoms'], args['box'], args['dt']
    element, k, nsteps   = args['element'], args['k'], args['steps']
    dumpname, freq, T = args['dumpname'], args['freq'], args['T']
    csvname = args['csvname']

    radius = element['r']
    
    dim = len(box)

    pos = generate_points_with_min_distance(natoms, box[0][1], 2 * element['r'] + element['sig'])

    vels = np.array(np.random.rand(natoms, dim) - 0.5, dtype=np.float64)
    vels = vels * np.sqrt(natoms * KB * T / element['m'])


    wallHitCheck(pos, box)

    radius = np.ones(natoms) * radius

    for step in tqdm(range(nsteps+1)):

        forces, E_pot = computeForce(pos, vels, element, size, k, True)

        vels = integrate(pos, vels, element, dt, k)

        wallHitCheck(pos, box)

        if not step%freq:
            writeOutput(dumpname, natoms, step, box, radius=radius, pos=pos, v=vels)
        
            backup = pd.DataFrame(index=np.ones(natoms)*step)
            backup['id'] = range(natoms)
            backup[['x', 'y']] = pos
            backup[['vx', 'vy']] = vels
            backup[['fx', 'fy']] = forces
            backup['E_pot'] = E_pot
            backup['E_kin'] = 0.5 * element['m'] * (backup['vx']**2 + backup['vy']**2)
            if not os.path.exists(csvname):
                backup.to_csv(csvname)
            else:
                backup.to_csv(csvname, mode='a', header=False)

if __name__ == '__main__':

    element = ELEMENTS['He']
    size = element['r'] * 30
    box = ((-size, size), (-size, size))
    params = {
        'steps': 10000,
        'natoms': 100,
        'element': element,
        'box': box,
        'T': 273,
        'Tt': 1,
        'dt': 0.001,
        'k': 10,
        'freq': 1
        }
    now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    name_dump = f'dump/N{params["natoms"]}_{now}.dump'
    name_csv = f'dump/N{params["natoms"]}_{now}.csv'
    if os.path.exists(name_dump):
        os.remove(name_dump)
    run(**params, dumpname=name_dump, csvname=name_csv)