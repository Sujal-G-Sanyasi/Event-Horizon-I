# physics_server.py
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import math

app = Flask(__name__)
CORS(app)


# Kepler constant squared 
G = 0.00029591220828559104

# idk this u can create your own solar body if u want lol
class Body:
    def __init__(self, name, mass, pos, vel, radius=0.01):
        self.name = name
        self.mass = float(mass)
        self.radius = float(radius)
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)

# Planet masses (solar masses)
Mercury_m = 1.651e-7
Venus_m    = 2.447e-6
Earth_m    = 3.003e-6
Mars_m     = 3.227e-7
Jupiter_m  = 0.0009543
Saturn_m   = 0.0002857
Uranus_m   = 4.366e-5
Neptune_m  = 5.151e-5

# Initializer function that creates fresh bodies (used on start & reset)
def make_initial_bodies():
    sun = Body("Sun", 1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], radius=0.1)

    # orbital radii (AU) - increased for more spacing
    data = [
        ("Mercury", 2.0, Mercury_m),
        ("Venus",   4.0, Venus_m),
        ("Earth",   6.0, Earth_m),
        ("Mars",    8.0, Mars_m),
        ("Jupiter", 20.0, Jupiter_m),
        ("Saturn",  30.0, Saturn_m),
        ("Uranus",  40.0, Uranus_m),
        ("Neptune", 50.0, Neptune_m),
    ]

    bodies = [sun]
    for name, r, m in data:
        # circular orbital speed consistent with chosen G: v = sqrt(G * M_sun / r)
        v_circ = math.sqrt(G / r)
        pos = np.array([0.0, 0.0, r])  # Start on Z axis
        vel = np.array([v_circ, 0.0, 0.0])  # Velocity in X direction
        bodies.append(Body(name, m, pos, vel, radius=0.01))

    return bodies


all_bodies = make_initial_bodies()

# Physics helpers funcs
def compute_accelerations(bodies):
    """Return list of accelerations (numpy arrays) for each body given current positions."""
    n = len(bodies)
    accs = [np.zeros(3, dtype=float) for _ in range(n)]
    for i in range(n):
        bi = bodies[i]
        for j in range(n):
            if i == j:
                continue
            bj = bodies[j]
            r_vec = bj.pos - bi.pos
            dist = np.linalg.norm(r_vec)
            if dist == 0:
                continue
            # a = G * other.mass / dist^3 * r_vec
            accs[i] += (G * bj.mass) * r_vec / (dist**3)
    return accs

def step_bodies(bodies, dt):
    """Literally leafrog is godly which make orbital planetary movement stable by
       1st Half Velocity -> Move position -> 2nd Half Velocity this corrects unusual
       spike of velocity atleast better tha Euler   
    """
    # a(t)
    a1 = compute_accelerations(bodies)

    # half-kick
    for i, b in enumerate(bodies):
        b.vel += 0.5 * a1[i] * dt

    # drift
    for b in bodies:
        b.pos += b.vel * dt

    # a(t+dt)
    a2 = compute_accelerations(bodies)

    # second half-kick
    for i, b in enumerate(bodies):
        b.vel += 0.5 * a2[i] * dt


@app.route('/get_positions')
def get_positions():
    
    out = []
    for b in all_bodies:
        out.append({
            'name': b.name,
            'pos': b.pos.tolist(),
            'vel': b.vel.tolist()
        })
    return jsonify(out)

@app.route('/step')
def step_endpoint():
   
    global all_bodies

    # parse dt
    dt_param = request.args.get('dt', default=None, type=float)
    if dt_param is None:
        # default: advance 1 day
        sim_dt = 1.0
    else:
        sim_dt = float(dt_param)
        if sim_dt <= 0:
            return jsonify({'error': 'dt must be positive'}), 400

    # substep if dt is large for stability
    MAX_SUBDT = 0.2  # days -- tuneable; smaller => more stable but slower
    steps = max(1, int(math.ceil(sim_dt / MAX_SUBDT)))
    sub_dt = sim_dt / steps

    # clamp steps to avoid runaway
    steps = min(1000, steps)

    for _ in range(steps):
        step_bodies(all_bodies, sub_dt)

    # prepare response
    out = []
    for b in all_bodies:
        out.append({
            'name': b.name,
            'pos': b.pos.tolist(),
            'vel': b.vel.tolist()
        })
    return jsonify(out)

@app.route('/reset')
def reset_endpoint():
    global all_bodies
    all_bodies = make_initial_bodies()
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    print("Starting physics server on http://localhost:5000")
    app.run(debug=True, port=5000)

