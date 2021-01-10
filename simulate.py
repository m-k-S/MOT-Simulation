# Simulation
from scipy import constants
import numpy as np
from scipy.integrate import solve_ivp

# Optimization
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# Plotting
from tqdm import tqdm

#########################
#                       #
#   GLOBAL PARAMETERS   #
#                       #
#########################

# Physical Constants
hbar = constants.hbar
kB = constants.Boltzmann
muB = constants.value('proton mag. mom.') / constants.value('proton mag. mom. to Bohr magneton ratio')
Da = constants.value('atomic mass constant')

# Strontium 88 parameters
m = 88 * Da
Gamma = 2 * np.pi * 32 * 10**6 # linewidth of 32MHz (angular frequency)
k = (2 * np.pi) / (0.461 * 10**-6)
IO = 42.5 # Saturation intensity, mW/cm^2
gJ = 1.

# Fixed parameters
MOT3dXoffset = 0.
MOT3dYoffset = -0.003
MOT3dZoffset = 0.43
offset = np.array([MOT3dXoffset, MOT3dYoffset, MOT3dZoffset])
g = 9.8
pushbeamMaxZ = 0.1

# Capture condition (at what position / speed do we count atoms as captured?)
targetZoneSize = 2 * 10**-3
maxSpeed = 10

# Example free parameters (unused!)
example_params = {
    's0': 2.,
    's03D': 2.,
    's0push': 0.01,
    'Delta2D': -1.5,
    'Delta3D': -2.5,
    'DeltaPush': -0.9,
    'Bgrad2D': 130. * 10**2,
    'Bgrad3D': 50. * 10**2,
    'wtransverse2D': 4. * 10**-3,
    'wlongitude2D': 10. * 10**-3,
    'w3D': 0.01,
    'wPush': 3. * 10**-3
}

###############################
#                             #
#   MOT Equations of Motion   #
#                             #
###############################

r22 = lambda s0, delta: (0.5 * s0) / (1 + s0 + 4 * delta**2)

def ax2DMOT(delta, v, x, y, z, sim_params):
    s0l = sim_params['s0']
    s0r = sim_params['s0']
    wtransverse2D = sim_params['wtransverse2D']
    wlongitude2D = sim_params['wlongitude2D']
    Bgrad2D = sim_params['Bgrad2D']
    return np.exp(-2 * y**2 / wtransverse2D**2) \
            * np.exp(-2 * z**2 / wlongitude2D**2) * (hbar * k * Gamma / m) \
            * (r22(s0l, delta - ((k * v) / Gamma) - (x * Bgrad2D * muB * gJ / (10**4 * hbar * Gamma))) \
             - r22(s0r, delta + ((k * v) / Gamma) + (x * Bgrad2D * muB * gJ / (10**4 * hbar * Gamma))))

def ax3DMOT(delta, v, x, y, z, sim_params):
    s03DMOTX1 = sim_params['s03D']
    s03DMOTX2 = sim_params['s03D']
    w3D = sim_params['w3D']
    Bgrad3D = sim_params['Bgrad3D']
    return np.exp(-2 * y**2 / w3D**2) \
            * np.exp(-2 * z**2 / w3D**2) * (hbar * k * Gamma / m) \
            * (r22(s03DMOTX1, delta - ((k * v) / Gamma) - (x * Bgrad3D * muB * gJ / (10**4 * hbar * Gamma))) \
             - r22(s03DMOTX2, delta + ((k * v) / Gamma) + (x * Bgrad3D * muB * gJ / (10**4 * hbar * Gamma))))

def ay2DMOT(delta, v, x, y, z, sim_params):
    s0l = sim_params['s0']
    s0r = sim_params['s0']
    wtransverse2D = sim_params['wtransverse2D']
    wlongitude2D = sim_params['wlongitude2D']
    Bgrad2D = sim_params['Bgrad2D']
    return np.exp(-2 * x**2 / wtransverse2D**2) \
            * np.exp(-2 * z**2 / wlongitude2D**2) * (hbar * k * Gamma / m) \
            * (r22(s0l, delta - ((k * v) / Gamma) - (y * Bgrad2D * muB * gJ / (10**4 * hbar * Gamma))) \
             - r22(s0r, delta + ((k * v) / Gamma) + (y * Bgrad2D * muB * gJ / (10**4 * hbar * Gamma))))

def ay3DMOT(delta, v, x, y, z, sim_params):
    s03DMOTY1 = sim_params['s03D']
    s03DMOTY2 = sim_params['s03D']
    w3D = sim_params['w3D']
    Bgrad3D = sim_params['Bgrad3D']
    return np.exp(-2 * x**2 / w3D**2) \
            * np.exp(-2 * z**2 / w3D**2) * (hbar * k * Gamma / m) \
            * (r22(s03DMOTY1, delta - ((k * v) / Gamma) - (y * Bgrad3D * muB * gJ / (10**4 * hbar * Gamma))) \
             - r22(s03DMOTY2, delta + ((k * v) / Gamma) + (y * Bgrad3D * muB * gJ / (10**4 * hbar * Gamma))))

az2DMOT = lambda delta, v, x, y, z, sim_params: 0

def azPush(delta, v, x, y, z, sim_params):
    wPush = sim_params['wPush']
    s0push = sim_params['s0push']
    if z < pushbeamMaxZ:
        return np.exp(-2 * x**2 / wPush**2) * np.exp(-2 * y**2 / wPush**2) * (hbar * k * Gamma / m) \
                * r22(s0push, delta - ((k * v) / Gamma))
    else:
        return 0

def az3DMOT(delta, v, x, y, z, sim_params):
    s03DMOTZ1 = sim_params['s03D']
    s03DMOTZ2 = sim_params['s03D']
    w3D = sim_params['w3D']
    Bgrad3D = sim_params['Bgrad3D']
    return np.exp(-2 * x**2 / w3D**2) * np.exp(-2 * y**2 / w3D**2) * np.exp(-2 * z**2 / w3D**2) * (hbar * k * Gamma / m) \
            * (r22(s03DMOTZ1, delta - ((k * v) / Gamma) - (z * Bgrad3D * muB * gJ / (10**4 * hbar * Gamma))) \
             - r22(s03DMOTZ2, delta + ((k * v) / Gamma) + (z * Bgrad3D * muB * gJ / (10**4 * hbar * Gamma))))

def axMOT(v, x, y, z, sim_params):
    Delta2D = sim_params['Delta2D']
    Delta3D = sim_params['Delta3D']
    return ax2DMOT(Delta2D, v, x, y, z, sim_params) \
            + ax3DMOT(Delta3D, v, x - MOT3dXoffset, y - MOT3dYoffset, z - MOT3dZoffset, sim_params)

def ayMOT(v, x, y, z, sim_params):
    Delta2D = sim_params['Delta2D']
    Delta3D = sim_params['Delta3D']
    return ay2DMOT(Delta2D, v, x, y, z, sim_params) \
            + ay3DMOT(Delta3D, v, x - MOT3dXoffset, y - MOT3dYoffset, z - MOT3dZoffset - g, sim_params)

def azMOT(v, x, y, z, sim_params):
    Delta2D = sim_params['Delta2D']
    Delta3D = sim_params['Delta3D']
    DeltaPush = sim_params['DeltaPush']
    return az2DMOT(Delta2D, v, x, y, z, sim_params) \
            + az3DMOT(Delta3D, v, x - MOT3dXoffset, y - MOT3dYoffset, z - MOT3dZoffset, sim_params) \
            + azPush(DeltaPush, v, x, y, z, sim_params)

#################################
#                               #
#   Create Initial Conditions   #
#                               #
#################################

# Jet Loading Parameters
jetParams = {
    'T': 600,
    'Sigmav': np.sqrt(kB * 600 / m),
    'Mux': -9 * 10**-3,
    'Muy': -9 * 10**-3,
    'Muz': 0., # 10**-3?
    'Sigmax': 0.1 * 10**-3,
    'Sigmay': 0.1 * 10**-3,
    'Sigmaz': 1.25 * 10**-3,
    'fullacceptanceAngle': 40.
}

runs = 2000000
tFinal = 0.05
preFilterMaxAxialSpeed = 30.
preFilterMaxTranverseSpeed = 100.

# Jet Initialization
initialPositions = np.vstack((
    np.random.normal(jetParams['Mux'], jetParams['Sigmax'], runs),
    np.random.normal(jetParams['Muy'], jetParams['Sigmay'], runs),
    np.random.normal(jetParams['Muz'], jetParams['Sigmaz'], runs)
)).T

initialVelocities = np.vstack((
    np.random.normal(0, jetParams['Sigmav'], runs),
    np.random.normal(0, jetParams['Sigmav'], runs),
    np.random.normal(0, jetParams['Sigmav'], runs)
)).T

initPositionsFiltered = []
initVelocitiesFiltered = []

print ("Initializing atom positions and velocities...\n")
for idx in tqdm(range(runs)):
    angle_cutoff = jetParams['fullacceptanceAngle']
    if initialVelocities[idx, 1] > 0 \
    and (np.pi / 180) * (45 - angle_cutoff / 2) <= np.arctan(initialVelocities[idx, 1] / initialVelocities[idx, 0]) \
    and np.arctan(initialVelocities[idx, 1] / initialVelocities[idx, 0]) <= (np.pi / 180) * (45 + angle_cutoff / 2) \
    and np.abs(initialVelocities[idx, 2]) <= preFilterMaxAxialSpeed \
    and np.sqrt(initialVelocities[idx, 0]**2 + initialVelocities[idx, 1]**2) <= preFilterMaxTranverseSpeed:
        initPositionsFiltered.append(initialPositions[idx, :])
        initVelocitiesFiltered.append(initialVelocities[idx, :])

initPositionsFiltered = np.asarray(initPositionsFiltered)
initVelocitiesFiltered =  np.asarray(initVelocitiesFiltered)

assert (len(initPositionsFiltered) == len(initVelocitiesFiltered))
num_atoms = len(initPositionsFiltered)
print ("Number of atoms: {}\n".format(num_atoms))

########################
#                      #
#       Simulate       #
#                      #
########################

# Optimization Objective
def simulate_MOT(params, init_positions=initPositionsFiltered, init_velocities=initVelocitiesFiltered):
    init_conditions = np.hstack((init_positions, init_velocities))
    num_atoms = init_conditions.shape[0]

    # Create IVP
    def ivp(t, vec, sim_params=params):
        x, y, z, vx, vy, vz = tuple(vec)
        return [vx, vy, vz, axMOT(vx, x, y, z, sim_params), ayMOT(vy, x, y, z, sim_params), azMOT(vz, x, y, z, sim_params)]

    outputs = []
    for idx in range(num_atoms):
        output = solve_ivp(ivp, (0, 0.05), init_conditions[idx, :], vectorized=True)
        outputs.append(output)

    final = np.array([output.y[:, -1] for output in outputs])

    finalPositions = final[:, 0:3]
    finalVelocities = final[:, 3:6]

    captured = []

    # Capture Analysis
    for idx in range(num_atoms):
        if np.linalg.norm(finalPositions[idx] - offset) <= targetZoneSize:
            if np.linalg.norm(finalVelocities[idx]) <= maxSpeed:
                captured.append(idx)

    return len(captured)

########################
#                      #
#       Optimize       #
#                      #
########################

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1, 1)

def initialize_parameters(bounds, fn):
    dim = len(list(bounds.keys()))
    initial_params = [{k:0 for k in bounds.keys()} for _ in range(2**dim)]
    idx = 0
    for key, val in bounds.items():
        kdx = 0
        for jdx in range(0, 2**dim, 2**idx):
            for d in initial_params[jdx:jdx+2**idx]:
                d[key] = val[kdx % 2]
            kdx += 1
        idx += 1

    io_pairs = []
    for params in tqdm(initial_params):
        captured_atoms = fn(params) # fn is the end2end simulation function; returns number of captured atomss
        io_pairs.append([params, captured_atoms])

    return io_pairs

m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
gpr = GaussianProcessRegressor(kernel=m52, alpha=0.04)

# Bounds on free parameters
bounds = {
    's0': [0.1, 3],
    's0push': [0, 20],
    's03D': [0.1, 3],
    'Delta2D': [-5, 1],
    'Delta3D': [-5, 1],
    'DeltaPush': [-5, 1],
    'Bgrad2D': [0, 300],
    'Bgrad3D': [0, 300],
    'wtransverse2D': [0.002, 0.005],
    'wlongitude2D': [0.002, 0.010],
    'w3D': [0.002, 0.008],
    'wPush': [0.0005, 0.002]
}

# Initialize samples
print ("Seeding optimization data from parameter bounds...\n")
initial_data = initialize_parameters(bounds, simulate_MOT)

X_sample = np.array([
    [v for v in p[0].values()] for p in initial_data
])
Y_sample = np.array([
    p[1] for p in initial_data
])

bounds_np = np.array([
    b for b in self.bounds.values()
])

# Number of iterations
n_iter = 1000

print ("Running optimization...\n")
for i in tqdm(range(n_iter)):
    # Update Gaussian process with existing samples
    gpr.fit(X_sample, Y_sample)

    # Obtain next sampling point from the acquisition function (expected_improvement)
    X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds_np)
    next_params = {k:v for k, v in zip(bounds.keys(), X_next)}

    # Obtain next noisy sample from the objective function
    Y_next = simulate_MOT(next_params)

    # Add sample to previous samples
    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.vstack((Y_sample, Y_next))

print (Y_sample)
