import numpy as np
import pdb
import Delan_H
from scipy.linalg import sqrtm


class DMP(object):
    def __init__(self, pastor_mod=False):
        self.pastor_mod = pastor_mod
        # Transformation system
        self.alpha = 25.0             # K = D = 20.0
        self.beta = self.alpha / 4.0  # = K / D = 100.0 / 20.0 = 5.0
        # Canonical system
        self.alpha_t = self.alpha / 3.0
        # Obstacle avoidance
        self.gamma_o = 1000.0
        self.beta_o = 20.0 / np.pi

    def phase(self, n_steps, t=None):
        """The phase variable replaces explicit timing.

        It starts with 1 at the beginning of the movement and converges
        exponentially to 0.
        """
        phases = np.exp(-self.alpha_t * np.linspace(0, 1, n_steps))
        if t is None:
            return phases
        else:
            return phases[t]

    def spring_damper(self, x0, g, tau, s, X, Xd,delan=False):
        """The transformation system generates a goal-directed movement."""
        if self.pastor_mod:
            # Allows smooth adaption to goals, in the original version also the
            # forcing term is multiplied by a constant alpha * beta which you
            # can of course omit since the weights will simply be scaled
            mod = -self.beta * (g - x0) * s
        else:
            mod = 0.0
        if delan == False:
            #pdb.set_trace()
            #return self.alpha * (self.beta * (g - X) - tau * Xd + mod) / tau ** 2
            beta = 0.35*(self.alpha)**0.5
            return self.alpha*(g - X) - tau*beta*Xd  - self.alpha*(g-x0)*s 
        if delan == True:
            #pdb.set_trace()
            #print("IN DELAN")
            #X = np.asanyarray([X[0] , X[1] ,0 ])
            #g = np.asanyarray([g[0] , g[1] ,0 ])
            alpha = Delan_H.value(X,g)

            #np.interp(alpha, (alpha.min(), alpha.max()), (0, 1))
            beta = 0.35*sqrtm(alpha)
            #pdb.set_trace()
            next_coord =alpha@(g - X) - tau*beta@Xd  - alpha@(g-x0)*s
            return next_coord

        

    def forcing_term(self, x0, g, tau, w, s, X, scale=False):
        """The forcing term shapes the movement based on the weights."""
        n_features = w.shape[1]
        f = np.dot(w, self._features(tau, n_features, s))
        if scale:
            f *= g - x0

        if X.ndim == 3:
            F = np.empty_like(X)
            F[:, :] = f
            return F
        else:
            return f



    def obstacle(self, o, X, Xd):
        """Obstacle avoidance is based on point obstacles."""
        if X.ndim == 1:
          X = X[np.newaxis, np.newaxis, :]
        if Xd.ndim == 1:
          Xd = Xd[np.newaxis, np.newaxis, :]

        C = np.zeros_like(X)
        R = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                      [np.sin(np.pi / 2.0),  np.cos(np.pi / 2.0)]])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                obstacle_diff = o - X[i, j]
                theta = (np.arccos(obstacle_diff.dot(Xd[i, j]) /
                                   (np.linalg.norm(obstacle_diff) *
                                    np.linalg.norm(Xd[i, j]) + 1e-10)))
                C[i, j] = (self.gamma_o * R.dot(Xd[i, j]) * theta *
                           np.exp(-self.beta_o * theta))

        return np.squeeze(C)




def trajectory(dmp, w, x0, g, tau, dt, o=None, shape=True, avoidance=False,
               verbose=0,delan=False):
    """Generate trajectory from DMP in open loop."""
    if verbose >= 1:
        print("Trajectory with x0 = %s, g = %s, tau=%.2f, dt=%.3f"
              % (x0, g, tau, dt))

    x = x0.copy()
    xd = np.zeros_like(x, dtype=np.float64)
    xdd = np.zeros_like(x, dtype=np.float64)
    X = [x0.copy()]
    Xd = [xd.copy()]
    Xdd = [xdd.copy()]

    # Internally, we do Euler integration usually with a much smaller step size
    # than the step size required by the system
    internal_dt = min(0.001, dt)
    n_internal_steps = int(tau / internal_dt)
    steps_between_measurement = int(dt / internal_dt)

    # Usually we would initialize t with 0, but that results in floating point
    # errors for very small step sizes. To ensure that the condition t < tau
    # really works as expected, we add a constant that is smaller than
    # internal_dt.
    t = 0.5 * internal_dt
    ti = 0
    S = dmp.phase(n_internal_steps + 1)
    while t < tau:
        t += internal_dt
        ti += 1
        s = S[ti]
        #pdb.set_trace()
        x += internal_dt * xd
        xd += internal_dt * xdd

        sd = dmp.spring_damper(x0, g, tau, s, x, xd,delan)
        f = dmp.forcing_term(x0, g, tau, w, s, x) if shape else 0.0
        #pdb.set_trace()
        
        C = dmp.obstacle(o[0:2], x[0:2], xd[0:2]) if avoidance else 0.0
        if avoidance:
            xdd = sd + 3*np.asarray([C[0],C[1],0]) #+ f
        else:
            xdd = sd + 2*C


        if ti % steps_between_measurement == 0:
            X.append(x.copy())
            Xd.append(xd.copy())
            Xdd.append(xdd.copy())

    return np.array(X), np.array(Xd), np.array(Xdd)

def cartesian_product(*arrs):
    return np.transpose(np.meshgrid(*arrs)).reshape(-1, len(arrs))


def potential_field(dmp, t, T, v, w, x0, g, tau, dt, o, x_range, y_range,
                    n_tics):
    """xx, yy,zz = np.meshgrid(np.linspace(x_range[0], x_range[1], n_tics),
                         np.linspace(y_range[0], y_range[1], n_tics),
                         np.zeros([0]))"""
    #pdb.set_trace()
    x_step = (x_range[1]-x_range[0])/n_tics
    y_step = (y_range[1] - y_range[0])/n_tics
    xx, yy, zz =T[:,0],T[:,1],T[:,2] 
    """np.meshgrid(np.arange(x_range[0], x_range[1], x_step),
                        np.arange(y_range[0], y_range[1], y_step),
                        np.asarray([0.0]))"""

    #cart_productarray = cartesian_product(xx, yy, zz)
    x = T[0:-1:20] #np.unique(cart_productarray,axis=0)
    #pdb.set_trace()
    #x = np.array((xx, yy,zz)).transpose((1, 3, 0))
    xd = np.empty_like(x)
    xd[:, :] = v[0:-1:20]
    #pdb.set_trace()
    n_steps = int(tau / dt)

    s = dmp.phase(n_steps, t)
    sd = dmp.spring_damper(x0, g, tau, s, x, xd)
    #f = dmp.forcing_term(x0, g, tau, w, s, x)
    #pdb.set_trace()
    C = np.zeros((sd.shape[0]  ,  sd.shape[1] ))
    for i in range (C.shape[0] ) :
        C[i,:2] = dmp.obstacle(o[0:2], x[i,0:2], xd[i,0:2])
    acc = sd  + C
    return xx, yy, sd,-1, C, acc,x,xd


