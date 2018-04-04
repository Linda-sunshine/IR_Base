package structures;

/**
 * Created by lulin on 3/28/18.
 */
public class _Doc4ETBIR extends _Doc{

    public double[] m_mu; // mean vector \mu in variational inference p(\theta|\mu,\Sigma)
    public double[] m_Sigma; // diagonal covariance matrix \Sigma in variational inference p(\theta|\mu,\Sigma)
    public double m_zeta; //Taylor expansion parameter \zeta related to p(\theta|\mu,\Sigma)

    public _Doc4ETBIR(int ID, String name, String prodID, String title, String source, int ylabel, long timeStamp){
        super(ID,  name,  prodID,  title,  source,  ylabel,  timeStamp);
    }
}
