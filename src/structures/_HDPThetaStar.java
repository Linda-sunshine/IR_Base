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
	protected double m_gamma;
	
	// Parameters used in MMB model.
	protected int m_edgeSize;
	double[] m_B;
	
	public _HDPThetaStar(int dim, int lmSize, double gamma) {
		super(dim);
		m_psi = new double[lmSize];
		m_gamma = gamma;
		m_edgeSize = 0;
	}
	
	public _HDPThetaStar(int dim, double gamma) {
		super(dim);
		m_gamma = gamma;
		m_edgeSize = 0;
	}
	
	public void initPsiModel(int lmSize){
		m_psi = new double[lmSize];
	}
	
	public double[] getPsiModel(){
		return m_psi;
	}
	
	public void initB(int dim){
		m_B = new double[dim];
	}
	public void setGamma(double g){
		m_gamma = g;
	}
	
	public double getGamma(){
		return m_gamma;
	}
	
	public String showStat() {
		return String.format("%d(%.2f/%.3f)", m_memSize, m_pCount/(m_pCount+m_nCount), m_gamma);
	}
	
	ArrayList<String> m_reviewNames = new ArrayList<String>();
	public void resetReviewNames(){
		m_reviewNames.clear();
	}
	public void addReviewNames(String s){
		m_reviewNames.add(s);
	}
	
	public int getReviewSize(){
		return m_reviewNames.size();
	}
	public ArrayList<String> getReviewNames(){
		return m_reviewNames;
	}
	public void resetPsiModel(){
		m_psi = null;
	}
	
	
	// Functions used in MMB model.
	public void updateEdgeCount(int c){
		m_edgeSize += c;
	}
	
	public int getEdgeSize(){
		return m_edgeSize;
	}
	
	public double[] getB(){
		return m_B;
	}
	// update B with the newly estimated value.
	public void updateB(double[] b){
		System.arraycopy(b, 0, m_B, 0, b.length);
	}
	public void resetB(){
		m_B = null;
	}
}
