import numpy as np

from scipy.special import logsumexp


def simulate_clv(ntaxa, ndays, ss_var=0.01):
    """Simulates data under compositional Lotka-Volterra.
        
        Let p = (p_1, ..., p_D) be the relative proportions
        of D taxa (species).

        Let x = alr(p), the additive log-ratio of p. Note 
        x is in R^{D-1} and p is in S^D.

        The state space model is:
            x_t ~ Normal(x_{t-1} + g + Ap_{t-1}, e)
        
        The observation model is:
            y_t ~ Multinomial(C_t, p_t = alr^{-1}(x_t))

        The count parameter C_t is chosen to simulate the
        varying sequencing depths observed across real samples.


    Parameters
    ----------
        ntaxa  : number of species to simulate
        ndays  : number of days to simulate
        ss_var : state space variance

    Returns
    -------
        x  : an ndays by ntaxa-1 matrix of latent states
        y  : an ndays by ntaxa matrix of observed sequencing counts
        A  : simulated interaction matrix A in R^{D-1 x D}
        g  : simulated growth rate vector g in R^{D-1}
        mu : initial mean

    """
    A  = np.random.normal(loc=0,scale=0.2,size=(ntaxa-1, ntaxa))
    g  = np.random.normal(loc=0,scale=0.1,size=ntaxa-1)
    mu = np.random.normal(loc=0,scale=0.1)
    N  = 10000 # sequencing reads parameter

    latent_dim = A.shape[0]
    x = []
    y = []
    mu  = np.random.multivariate_normal(mean=np.zeros(latent_dim), cov=np.eye(latent_dim))
    for t in range(ndays):
        xt = mu

        # increase dimension by 1
        xt1 = np.concatenate((xt, np.array([0])))
        pt = np.exp(xt1 - logsumexp(xt1))

        # simulate total number of reads with over-dispersion
        logN = np.random.normal(loc=np.log(N), scale=0.5)
        Nt = np.random.poisson(np.exp(logN))
        yt = np.random.multinomial(Nt, pt).astype(float)

        x.append(xt)
        y.append(yt)

        mu  = xt + g + A.dot(pt)
    return x, y, A, g, mu


if __name__ == "__main__":
    # simulate 10 species across 50 days
    x, y, A, g, mu = simulate_clv(10, 50, ss_var=0.01)