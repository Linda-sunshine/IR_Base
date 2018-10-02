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
    public double[] m_sigma_sqrt_theta; // square root of the sigma

    public int m_index;

    public _Doc4EUB(_Review r, int idx){
        super(r.getID(), r.getSource(), r.getYLabel(), r.getUserID(), r.getItemID(), r.getCategory(), r.getTimeStamp());
        m_mask = r.getMask4CV();
        m_index = idx;
        m_x_sparse = r.getSparse();
        m_totalLength = r.getTotalDocLength();
    }

    public void setIndex(int idx){
        m_index = idx;
    }
    public int getIndex(){
        return m_index;
    }
    public void setTopics4Variational(int k, double alpha, double mu, double sigma){
        super.setTopics4Variational(k, alpha);
        m_mu_theta = new double[k];
        m_sigma_theta = new double[k];
        m_sigma_sqrt_theta = new double[k];

        for(int i=0; i<k; i++){
            m_mu_theta[i] = mu + Math.random();
            m_sigma_theta[i] = sigma + Math.random() * 0.5 * sigma;
            m_sigma_sqrt_theta[i] = Math.sqrt(m_sigma_theta[i]);
            if (i == 0)
                m_logZeta = m_mu_theta[i] + 0.5 * m_sigma_theta[i];
            else
                m_logZeta = Utils.logSum(m_logZeta, m_mu_theta[i] + 0.5 * m_sigma_theta[i]);
        }
    }
}

