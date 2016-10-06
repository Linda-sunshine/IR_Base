package structures;

import java.util.ArrayList;

/**
 * The structure wraps both \phi(_thetaStar) and \psi.
 * @author lin
 *
 */
public class _HDPThetaStar extends _thetaStar {
	// beta in _thetaStar is \phi used in HDP.
	
	//this will be in log space!
	protected double[] m_psi;// psi used in multinomal distribution of language model (may be of different dimension as \phi).
	public int m_hSize; //total number of local groups in the component.
	protected ArrayList<_Review> m_reviews; //reviews assigned to this group
	
	public _HDPThetaStar(int dim, int lmSize) {
		super(dim);
		m_reviews = new ArrayList<_Review>();
		m_psi = new double[lmSize];
	}
	
	public _HDPThetaStar(int dim) {
		super(dim);
		m_reviews = new ArrayList<_Review>();
	}
	
	public double[] getPsiModel(){
		return m_psi;
	}
	
	// Update \psi with the newly estimated prob. 
	public void updatePsiModel(double[] prob){
		System.arraycopy(prob, 0, m_psi, 0, prob.length);
	}
	
	public void addOneReview(_Review r){
		m_reviews.add(r);
	}
	
	//this might be expensive to perform
	public void rmReview(_Review r){
		m_reviews.remove(r);
	}
	
	public ArrayList<_Review> getReviews(){
		return m_reviews;
	}
}
