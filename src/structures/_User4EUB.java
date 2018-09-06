package structures;

import utils.Utils;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * Each user has a set of variational parameters:
 * variational parameters (\mu, \simga) for user embedding u
 * variational parameters (\mu, \sigma) for affinity \delta
 * variational parameters for taylor parameter \epsilon
 */
public class _User4EUB extends _User {
    public double[] m_mu_u;
    public double[] m_sigma_u;

    public double[] m_epsilon;

    // the variational paramters for the affinity with other users
    public double[] m_mu_delta;
    public double[] m_sigma_delta;

    public _User4EUB(String userID){
        super(userID);
    }

    public void setTopics4Variational(int dim, int userSize, double mu, double sigma){
        m_mu_u = new double[dim];
        m_sigma_u = new double[dim];

        m_mu_delta = new double[userSize];
        m_sigma_delta = new double[userSize];

        Utils.randomize(m_mu_u, mu);
        Utils.randomize(m_sigma_u, sigma);

        Utils.randomize(m_mu_delta, mu);
        Utils.randomize(m_sigma_delta, sigma);

        for(int i=0; i<userSize; i++){
            m_epsilon[i] = Math.exp(m_mu_delta[i] + 0.5*m_sigma_delta[i]*m_sigma_delta[i]) + 1;
        }
    }
}
