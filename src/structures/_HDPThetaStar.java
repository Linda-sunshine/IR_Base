package structures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import utils.Utils;

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
	protected int m_edgeSize[];//0: zero-edge count;1: one-edge count.
	
	// The count of the features inside clusters.
	double[] m_lmStat = null;
	HashMap<_HDPThetaStar, Double> m_B;
	
	public _HDPThetaStar(int dim, int lmSize, double gamma) {
		super(dim);
		m_psi = new double[lmSize];
		m_gamma = gamma;
		m_edgeSize = new int[2];
	}
	public _HDPThetaStar(int dim, double gamma) {
		super(dim);
		m_gamma = gamma;
		m_edgeSize = new int[2];
	}
	
	public void initLMStat(int lmDim){
		if(m_lmStat == null)
			m_lmStat = new double[lmDim];
		else 
			Arrays.fill(m_lmStat, 0);
	}
	public void clearLMStat(){
		Arrays.fill(m_lmStat, 0);
	}
	
	public void addLMStat(_SparseFeature[] fvs){
		for(_SparseFeature fv: fvs)
			m_lmStat[fv.getIndex()] += fv.getValue();
	}
	public void rmLMStat(_SparseFeature[] fvs){
		for(_SparseFeature fv: fvs)
			m_lmStat[fv.getIndex()] -= fv.getValue();
	}
	public double[] getLMStat(){
		return m_lmStat;
	}
	public double getOneLMStat(int index){
		return m_lmStat[index];
	}
	public double getLMSum(){
		double sum = 0;
		for(double v: m_lmStat)
			sum += v;
		return sum;
	}
	public void initPsiModel(int lmSize){
		m_psi = new double[lmSize];
	}
	
	public double[] getPsiModel(){
		return m_psi;
	}
	
	public void initB(){
		if(m_B == null)
			m_B = new HashMap<_HDPThetaStar, Double>();
		else
			m_B.clear();
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
	public void updateEdgeCount(int e, int c){
		m_edgeSize[e] += c;
	}
	
	public int getEdgeSize(int e){
		return m_edgeSize[e];
	}
	
	public int getTotalEdgeSize(){
		return Utils.sumOfArray(m_edgeSize);
	}
	// key: the thetastar, value: probability.
	public void addOneB(_HDPThetaStar t, double p){
		m_B.put(t, p);
	}
	// check if the user group existing or not.
	public boolean hasB(_HDPThetaStar theta){
		if(m_B.containsKey(theta))
			return true;
		else
			return false;
	}
	public double getOneB(_HDPThetaStar t){
		if(hasB(t))
			return m_B.get(t);
		else{
			System.out.println("The probability does not exist!");
			return 0;
		}
	}
	public HashMap<_HDPThetaStar, Double> getB(){
		return m_B;
	}
//	// update B with the newly estimated value.
//	public void updateB(double[] b){
//		System.arraycopy(b, 0, m_B, 0, b.length);
//	}
//	public void resetB(){
//		m_B = null;
//	}
}
