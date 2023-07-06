import numpy as np
import dPF as lib

Ub = 20e3
Sb = 1e6
Zb = Ub**2/Sb

# Nodes
nodes = [{'id': 0, 'Umax': 1.05, 'Umin': 0.95, 'slack': True  },
         {'id': 1, 'Umax': 1.05, 'Umin': 0.95, 'slack': False },
         {'id': 2, 'Umax': 1.05, 'Umin': 0.95, 'slack': False },
         {'id': 3, 'Umax': 1.05, 'Umin': 0.95, 'slack': False }]
# Lines
lines = [{'id': 0, 'From': 0, 'To': 1, 'R': 0.150/Zb, 'X': 0.600/Zb, 'Long': 1.00, 'Imax': 1e15 },    
         {'id': 1, 'From': 1, 'To': 2, 'R': 0.150/Zb, 'X': 0.700/Zb, 'Long': 1.00, 'Imax': 1e15 },    
         {'id': 2, 'From': 1, 'To': 3, 'R': 0.150/Zb, 'X': 0.700/Zb, 'Long': 1.00, 'Imax': 1e15 }]
# Loads
loads = [{'node': 1, 'P': 1e6/Sb, 'Q': 0 },
         {'node': 2, 'P': 3e6/Sb, 'Q': 0 },
         {'node': 2, 'P': 4e6/Sb, 'Q': 0 }]

# Generating the network
net = lib.grid(nodes, lines, loads, 1)
net.power_flow.solve()
net.report()
net.check()


sol = net.solve_pf_SOCP()
net.report()
net.check()


net.solve_pf_dSOCP(rho = 7, niter = 20)
net.report()
net.check()

ZM = net.ZM


