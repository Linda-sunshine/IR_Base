package structures;

import utils.Utils;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * Each document has a set of variational parameters:
 * variational parameters (\mu, \simga) for user preference \theta
 * variational parameters (\eta) for topic indicator z_{i,d,n}
 * variational parameters for taylor parameter \zeta
 */
public class _Doc4EUB extends _Review {

    public double m_logZeta = 0;
    public double[] m_mu_theta;
    public double[] m_sigma_theta; // only diagonal values are non-zero

    public _Doc4EUB(int ID, String source, int ylabel, String userID, String productID, String category, long timeStamp) {
        super(ID, source, ylabel, userID, productID, category, timeStamp);
    }

    public void setTopics4Variational(int k, double alpha, double mu, double sigma){
        super.setTopics4Variational(k, alpha);
        m_mu_theta = new double[k];
        m_sigma_theta = new double[k];

        for(int i=0; i<k; i++){
            m_mu_theta[i] = mu + Math.random();
            m_sigma_theta[i] = sigma + Math.random() * 0.5 * sigma;
            if (i == 0)
                m_logZeta = m_mu_theta[i] + 0.5 * m_sigma_theta[i];
            else
                m_logZeta = Utils.logSum(m_logZeta, m_mu_theta[i] + 0.5 * m_sigma_theta[i]);
        }
    }
}

