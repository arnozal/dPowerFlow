import numpy as np
from scipy.optimize import fsolve, minimize, NonlinearConstraint, LinearConstraint, Bounds
import matplotlib.pyplot as plt


class grid:
    def __init__(self, nodes, lines, loads, Ub):
        self.Ub = Ub
        self.nodes = self.add_nodes(nodes)                                      
        self.lines = self.add_lines(lines, self.nodes)                                                  
        self.loads = self.add_loads(loads, self.nodes)       
        self.power_flow = power_flow(self)        
        self.n = len(self.nodes) - 1
        self.m = len(self.lines)
        for node in self.nodes:
            node.m = len(node.lines)
            node.n = node.m
                
    def add_nodes(self, nodes):
        nodes_list = list()
        for item in nodes:
            nodes_list.append(node(item['id'], item['Umax'], item['Umin'], item['slack']))
        return nodes_list
        
    def add_lines(self, lines, nodes):
        lines_list = list()
        for item in lines:
            lines_list.append(line(item['id'], item['From'], item['To'], item['R'], item['X'], item['Long'], item['Imax'], nodes))
        return lines_list
        
    def add_loads(self, loads, nodes):
        loads_list = list()
        for item in loads:
            loads_list.append(load(item['node'], item['P'], item['Q'], nodes))
        return loads_list
    
    def check(self):
        print('Printing residuals...')
        res = 0
        for node in self.nodes[1:]:
            print(f'Node {node.ref}: {node.check():.5f}')
            res += np.linalg.norm(node.check())
        for line in self.lines:
            print(f'Line {line.nodes[0].ref}-{line.nodes[1].ref}: {line.check():.5f}')
            res += np.linalg.norm(line.check())
        print('')
        return res
            
    def report(self):
        print('Reporting the state of the network...')
        for node in self.nodes:
            print(f'Node {node.ref}: U = {node.U:.5f}')
        for line in self.lines:
            print(f'Line {line.nodes[0].ref}-{line.nodes[1].ref}: I = {line.I:.5f}')
        print('')
    
    def set_SOCP_vars(self, pr = False):             
        for item in self.nodes:
            if hasattr(item, 'U'):
                item.Ckk = np.real(item.U)**2 + np.imag(item.U)**2
            else:
                item.Ckk = 1
                item.U = 1
        for line in self.lines:
            n1 = line.nodes[0]
            n2 = line.nodes[1]
            if hasattr(n1, 'U') and hasattr(n2, 'U'):
                line.Ckt = np.real(n1.U)*np.real(n2.U) + np.imag(n1.U)*np.imag(n2.U) 
                line.Skt = np.real(n1.U)*np.imag(n2.U) - np.imag(n1.U)*np.real(n2.U) 
            else:
                line.Ckt = 1
                line.Skt = 0
        self.x = np.array([node.Ckk for node in self.nodes[1:]] + [line.Ckt for line in self.lines] + [line.Skt for line in self.lines])
        if pr:
            print('Printing SOCP variables...')
            for node in self.nodes:
                print(f'C{node.ref} = {node.Ckk}')
            for line in self.lines:
                print(f'C{line.nodes[0].ref}{line.nodes[1].ref} = {line.Ckt}')
                print(f'S{line.nodes[0].ref}{line.nodes[1].ref} = {line.Skt}') 
            print('')
        return self.x
    
    def set_SOCP_lc(self):
        self.A = np.zeros((2*self.n, self.n+2*self.m)) 
        self.B = np.zeros(2*self.n) 
        index_constraint = 0
        for index_node, node in enumerate(self.nodes[1:]):
            for line in node.lines:
                self.A[index_constraint, index_node] -= line.G 
                self.A[index_constraint, self.n + line.ref] += line.G     
                if line.nodes[0] == node:
                    self.A[index_constraint, self.n + self.m + line.ref] += line.B 
                else:
                    self.A[index_constraint, self.n + self.m + line.ref] -= line.B 
                self.A[index_constraint + 1, index_node] -= line.B 
                self.A[index_constraint + 1, self.n + line.ref] += line.B
                if line.nodes[0] == node:
                    self.A[index_constraint + 1, self.n + self.m + line.ref] -= line.G
                else:
                    self.A[index_constraint + 1, self.n + self.m + line.ref] += line.G
            self.B[index_constraint] = np.sum([item.P for item in node.loads])
            self.B[index_constraint + 1] = np.sum([item.Q for item in node.loads])
            index_constraint += 2
        return self.A, self.B    
    
    def set_x2SOCP(self, x):
        index = 0        
        for node in self.nodes[1:]:
            node.Ckk = x[index]
            index += 1
        for line in self.lines:
            line.Ckt = x[index]
            index += 1
        for line in self.lines:
            line.Skt = x[index]
            index += 1
        
    def set_SOCP2obj(self):
        for line in self.lines:
            sol = minimize(self.SOCP2vars, 
                           [1, 0], 
                           args = ([np.real(line.nodes[0].U), 
                                    np.imag(line.nodes[0].U),
                                    line.nodes[1].Ckk,
                                    line.Ckt,
                                    line.Skt],))
            x = sol.x
            line.nodes[1].U = complex(x[0],x[1])
        for line in self.lines:               
            line.I = (line.nodes[0].U - line.nodes[1].U)/line.Z
        for node in self.nodes:
            for load in node.loads:
                load.I = complex(load.P, -load.Q)/(np.conj(node.U))
        
    def solve_pf_SOCP(self):
        # Generating auxiliar matrices for linear constraints
        self.set_SOCP_lc()
        # Setting variables Ckk, Ckt and skt and cosntructing search vector x
        x = self.set_SOCP_vars()
        # Defining Objective function
        of = lambda x : - np.sum(x[self.n:self.n + self.m])
        # Defining constraints and solving the optimization problem
        cons = [NonlinearConstraint(self.ineq_con, 0, np.inf),
                LinearConstraint(self.A, lb = self.B, ub = self.B)]
        bnd = Bounds(lb = [0]*self.n + [0]*self.m + [-10]*self.m, # Ckk, Ckt, Skt
                     ub = [10]*self.n + [10]*self.m + [10]*self.m)
        sol = minimize(of, 
                       x, 
                       method='SLSQP',
                       bounds = bnd, 
                       constraints = cons)
        print('SOCP: ' + sol.message)
        print('')
        x = sol.x
        # Assigning solution to objects
        self.set_x2SOCP(x)        
        # Setting values to the electrical magnitudes form SOCP vars
        self.set_SOCP2obj()        
        return sol
    
    def SOCP2vars(self, x, args):
        e, f, Ckk, Ckt, Skt = args
        residual = np.array([np.abs(- Ckk + x[0]**2 + x[1]**2),
                             np.abs(- Ckt + e*x[0] + f*x[1]),
                             np.abs(- Skt + e*x[1] - f*x[0])])
        return 1e3*np.sum(residual.dot(residual))    
        
    def ineq_con(self, x):
        residuals = list()
        for line in self.lines:
            n1 = line.nodes[0].ref
            n2 = line.nodes[1].ref
            if n1 == 0:
                c1 = line.nodes[0].Ckk
            else:
                c1 = x[n1 - 1]
            c2 = x[n2 - 1]
            c12 = x[self.n + line.ref]    
            s12 = x[self.n + self.m + line.ref]     
            residuals.append(c1*c2 - c12**2 - s12**2)      
        return residuals
    
    def solve_pf_dSOCP(self, rho = 1, niter = 10):
        self.set_SOCP_vars()
        self.z = np.array([node.Ckk for node in self.nodes] + [line.Ckt for line in self.lines] + [line.Skt for line in self.lines])      
        self.xk2z()  
        for node in self.nodes[1:]:
            # Generating auxiliar matrices for linear constraints
            node.set_SOCP_lc()      
            node.set_SOCP_vars()
        for it in range(niter):
            print(f'Iteration \t {it+1}/{niter} \t norm(z) = {np.linalg.norm(self.z):.5}')            
            # Solving power flow problem
            for node in self.nodes[1:]:
                node.B = node.update_b_lc()
                node.solve_pf_SOCP(self.z, rho)
            # Updating z            
            self.ZM = np.array([node.M.T.dot(node.x) for node in self.nodes[1:]])
            self.z = np.divide(np.sum(self.ZM, axis=0), np.count_nonzero(self.ZM, axis=0))         
            # Updating dual variables
            for node in self.nodes[1:]:
                node.update_lmb(self.z, rho)
        # Assigning solution to objects
        self.set_x2SOCP(self.z[1:])        
        # Setting values to the electrical magnitudes form SOCP vars
        self.set_SOCP2obj()  
        print('')
            
    def xk2z(self):
        nz = self.n + self.m*2 + 1
        for node in self.nodes[1:]:
            node.M = np.zeros(( 1 + node.m*3, nz ))
            node.M[0, node.ref] = 1
            index_M = 1
            for item in zip(node.neigh, node.lines):
                node.M[index_M, item[0].ref] = 1
                node.M[index_M + 1, self.n + 1 + item[1].ref] = 1
                node.M[index_M + 2, self.n + 1 + self.m + item[1].ref] = 1
                index_M += 3
            node.lmb = np.zeros(1 + node.m*3)
                
        

class node:
    def __init__(self, ref, Umax, Umin, slack):
        self.ref = ref                      
        self.Umax = Umax
        self.Umin = Umin
        self.slack = slack
        self.lines = list()
        self.neigh = list()
        self.loads = list()
           
    def check(self):
        Ilines = np.sum([line.I if line.nodes[0] == self else -line.I for line in self.lines])
        Iloads = np.sum([load.I for load in self.loads])
        return Ilines + Iloads   
        
    def solve_pf_SOCP(self, z, rho):
        # Defining Objective function
        of = lambda x : - np.sum(x[2::3]) + self.lmb.dot(x - self.M.dot(z)) + (rho/2)*np.linalg.norm(x - self.M.dot(z))**2
        # Defining constraints and solving the optimization problem
        cons = [NonlinearConstraint(self.ineq_con, 0, np.inf),
                LinearConstraint(self.A, lb = self.B, ub = self.B)]
        bnd = Bounds(lb = [0] + [0, 0, -10]*self.m,
                     ub = [10] + [10, 10, 10]*self.m)
        sol = minimize(of, 
                       self.x, 
                       method='SLSQP',
                       bounds = bnd, 
                       constraints = cons)
        if sol.success == False:
            print(f'dSOCP at node {self.ref}: ' + sol.message)
        self.x = sol.x
        self.Ckk = self.x[0]
        return sol 
    
    def update_lmb(self, z, rho):
        self.lmb = self.lmb + rho*(self.x - self.M.dot(z))
    
    def update_lmb(self, z, rho):
        self.lmb += rho*(self.x - self.M.dot(z))
        
    def ineq_con(self, x):
        residuals = list()
        for index_line, line in enumerate(self.lines):
            n1 = line.nodes[0].ref
            if n1 == self:
                c1 = x[0]
                c2 = x[1 + 3*index_line]
            else:
                c1 = x[1 + 3*index_line]
                c2 = x[0]
            c12 = x[1 + 3*index_line + 1]    
            s12 = x[1 + 3*index_line + 2]     
            residuals.append(c1*c2 - c12**2 - s12**2)      
        return residuals
        
    
    def set_SOCP_vars(self):
        if hasattr(self, 'U'):
            self.Ckk = np.real(self.U)**2 + np.imag(self.U)**2
        else:
            self.Ckk = 1
            self.U = 1
        for line in self.lines:
            n1 = line.nodes[0]
            n2 = line.nodes[1]
            if hasattr(n1, 'U') and hasattr(n2, 'U'):
                line.Ckt = np.real(n1.U)*np.real(n2.U) + np.imag(n1.U)*np.imag(n2.U) 
                line.Skt = np.real(n1.U)*np.imag(n2.U) - np.imag(n1.U)*np.real(n2.U) 
            else:
                line.Ckt = 1
                line.Skt = 0
        self.x = [[self.Ckk]] + [[line.nodes[1].Ckk, line.Ckt, line.Skt] if line.nodes[0] == self else [line.nodes[0].Ckk, line.Ckt, line.Skt] for line in self.lines]
        self.x = np.array([item for sublist in self.x for item in sublist])
        return self.x
        
    def set_SOCP_lc(self):
        self.A = np.zeros((2, 1 + self.n+2*self.m)) 
        self.B = np.zeros(2) 
        for line_index, line in enumerate(self.lines):
            self.A[0, 0] -= line.G 
            self.A[0, 1 + line_index*3 + 1] += line.G     
            if line.nodes[0] == self:
                self.A[0, 1 + line_index*3 + 2] += line.B 
            else:
                self.A[0, 1 + line_index*3 + 2] -= line.B 
            self.A[1, 0] -= line.B 
            self.A[1, 1 + line_index*3 + 1] += line.B
            if line.nodes[0] == self:
                self.A[1, 1 + line_index*3 + 2] -= line.G
            else:
                self.A[1, 1 + line_index*3 + 2] += line.G
        self.B[0] = np.sum([item.P for item in self.loads])
        self.B[1] = np.sum([item.Q for item in self.loads])
        # Setting the value sent from adjacent nodes
        a_line = np.eye(1 + self.n+2*self.m)[1:len(self.neigh)*3:3, :]  
        self.A = np.block([[self.A], [a_line]])
        self.B = np.concatenate(( self.B, np.zeros(len(self.neigh)) ))
        self.B = self.update_b_lc()
        return self.A, self.B    
        
    def update_b_lc(self):
        for index_n, n in enumerate(self.neigh):
            self.B[-len(self.neigh) + index_n] = n.Ckk
        return self.B
        if 0 in [node.ref for node in self.neigh]:
            self.A = np.block([[self.A],
                               [np.eye(1 + self.n+2*self.m)[[node.ref for node in self.neigh].index(0)*3+1,:]]])
            self.B = np.concatenate((self.B, [1]))            
        return self.A, self.B    
        
        
class load:
    def __init__(self, node, P, Q, nodes_list):
        self.node = next((item for item in nodes_list if item.ref == node), None)
        self.node.loads.append(self)
        self.P = P
        self.Q = Q
    
class line:
    def __init__(self, ref, From, To, R, X, long, Imax, nodes_list):
        self.ref = ref     
        self.Z = complex(R, X)*long  
        self.G, self.B = np.real(1/self.Z), -np.imag(1/self.Z)
        self.Imax = Imax                
        self.nodes = [next((item for item in nodes_list if item.ref == From), None), 
                      next((item for item in nodes_list if item.ref == To), None)]   
        self.nodes[0].lines.append(self)
        self.nodes[1].lines.append(self)
        self.nodes[1].neigh.append(self.nodes[0])
        self.nodes[0].neigh.append(self.nodes[1])
        
    def check(self):
        res = self.Z*self.I - (self.nodes[0].U - self.nodes[1].U)
        return res
            
class power_flow: 
    def __init__(self, net):
        self.net = net
        
    def generate_Y(self):
        self.Y = np.zeros((len(self.net.nodes), len(self.net.nodes)), dtype = complex)
        for node in self.net.nodes:
            for line in node.lines:
                for line_con in line.nodes:
                    if node.ref != line_con.ref:
                        Z = complex(np.real(line.Z), np.imag(line.Z))
                        self.Y[node.ref, line_con.ref] = -1/Z
        for index in range(self.Y.shape[0]):
            self.Y[index, index] = -np.sum(self.Y[index,:])             
        self.Yrx = np.kron(np.real(self.Y), np.eye(2)) + np.kron(np.imag(self.Y), [[0, -1], [1, 0]])       
        return self.Y, self.Yrx   
    
    def generate_S(self):
        self.S = list()
        self.Srx = np.zeros(((len(self.net.nodes)-1)*2, (len(self.net.nodes)-1)*2))
        for index, node in enumerate(self.net.nodes[1:]):
            P = - np.sum([load.P for load in node.loads])
            Q = - np.sum([load.Q for load in node.loads])            
            self.S.append(complex(P, Q))
            self.Srx[index*2, index*2] = P
            self.Srx[index*2, index*2 + 1] = Q
            self.Srx[index*2 + 1, index*2] = - Q
            self.Srx[index*2 + 1, index*2 + 1] = P
        self.S = np.array(self.S)        
        return self.S, self.Srx
    
    def pf_constraints(self, x):
        U = np.concatenate(([1, 0], x))
        res = np.dot(self.Yrx[2:, :], U) - np.dot(self.Srx, U[2:])/np.dot(np.kron(np.eye(int(len(U[2:])/2)), np.ones((2,2)) ), U[2:]*U[2:])                                
        return list(res)
    
    def solve(self):
        self.generate_Y()
        self.generate_S()        
        x = [1, 0]*(len(self.net.nodes) - 1)
        x, infodict, ier, mesg = fsolve(self.pf_constraints, x, full_output = True)
        print('')  
        print('Power flow:', mesg)     
        self.net.nodes[0].U = self.net.Ub
        for node_index, node in enumerate(self.net.nodes[1:]):
            node.U = complex(x[node_index*2], x[node_index*2 + 1])
            for load in node.loads:
                load.I = complex(load.P, -load.Q)/(np.conj(node.U))
        for line in self.net.lines:
            line.I = (line.nodes[0].U - line.nodes[1].U)/line.Z
        residual = np.sum([node.check() for node in self.net.nodes[1:]])
        if np.linalg.norm(residual) > 0.01:
            print('Inconsistent solution')
        print('')
        return x, infodict
    
        