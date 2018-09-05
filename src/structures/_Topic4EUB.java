package structures;

import utils.Utils;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The data structure for topic embedding
 * We can either infer the topic embedding or pre-train the topic embedding
 * The dimension of the topic embedding is k.
 */
public class _Topic4EUB {
    public int m_index;
    public double[] m_mu_phi_k;
    public double[][] m_mu_sigma_phi_k;

    public _Topic4EUB(int index){
        m_index = index;
    }

    public void setTopics4Variational(int k, double mu, double sigma){
        m_mu_phi_k = new double[k];
        m_mu_sigma_phi_k = new double[k][k];
        for(int i=0; i<k; i++){
            Utils.randomize(m_mu_phi_k, mu);
            for(int j=0; j<k; j++){
                m_mu_sigma_phi_k[k][k] = sigma;
            }
        }
    }

}
