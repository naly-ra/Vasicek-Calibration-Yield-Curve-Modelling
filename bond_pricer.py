

def A(t,T,theta):
    return 1 - np.exp(-theta*(T-t)) / theta

def D(t,T,theta,mu,sigma):
    first_par = (mu - (sigma**2/2*theta))
    middle_par = A(t,T,theta) - (T-t)
    last_par = (sigma**2 * A(t,T,theta)**2) / 4*theta
    
    return first_par*middle_par - last_par


def bond_pricer(t,T,rates,theta,mu,sigma):
    At = A(t,T,theta)
    Dt = D(t,T,theta,mu,sigma) 
    return np.exp(-At*rates + Dt)