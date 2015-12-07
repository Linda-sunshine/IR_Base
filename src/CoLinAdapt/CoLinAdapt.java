package CoLinAdapt;

import java.util.ArrayList;

import structures._PerformanceStat;
import structures._Review;
import structures._SparseFeature;
import structures._User;

//online version of CoLinAdapt
public class CoLinAdapt extends LinAdapt {
	double m_eta3; // Weight for R2.
	double[] m_As;
	ArrayList<_User> m_neighbors;
	ArrayList<Integer> m_neighborIndexes;
	ArrayList<Double> m_neighborSims;

	public CoLinAdapt(int fg, int fn, double[] globalWeights, int[] featureGroupIndexes) {
		super(fg, fn, globalWeights, featureGroupIndexes);
		m_vSize = m_dim*2*(m_neighbors.size()+1);
		m_eta3 = 0.5;
	}

	public void setNeighbors(ArrayList<_User> neighbors){
		m_neighbors = neighbors;
	}
	
	public void setNeighborSims(ArrayList<Double> sims){
		m_neighborSims = sims;
	}
	
	// We can do A*w*x at the same time to reduce computation.
	public double logit(_SparseFeature[] fvs){
		double value = m_As[0]*m_weights[0] + m_As[m_dim];//Bias term: w0*a0+b0.
		int n = 0, k = 0;
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupIndexes[n];
			value += (m_As[k]*m_weights[n] + m_As[k + m_dim])*fv.getValue();
		}
		return 1/(1+Math.exp(-value));
	}

	// Calculate the new function value.
	public double calculateFunctionValue(ArrayList<_Review> trainSet){
		double sim = 0;
		double fValue = super.calculateFunctionValue(trainSet), R2 = 0, diff;	
		int jn;
		
		// Add the R2 part to the function value.
		for(int j=0; j<m_neighbors.size(); j++){
			sim = m_neighborSims.get(j);
			diff = 0;
			for(int k=0; k<m_dim; k++){
				
				jn = (j+1)*m_dim*2+k;
				diff += (m_As[k]-m_As[jn]) * (m_As[k]-m_As[jn]) // (a_ki-a_kj)^2
					+ (m_As[k+m_dim]-m_As[jn+m_dim]) * (m_As[k+m_dim]-m_As[jn+m_dim]); // (b_ki-b_kj)^2
			}
			R2 += sim * diff;
		}
		
		fValue += m_eta3 * R2;
		return fValue;
	}
	
	//Calculate the gradients for the use in LBFGS.
	@Override
	protected void gradientByR1(){
		//R1 regularization part
		for(int k=0; k<m_dim; k++){
			m_g[k] += 2 * m_eta1 * (m_As[k]-1);// add 2*eta1*(a_k-1)
			m_g[k+m_dim] += 2 * m_eta2 * m_As[k+m_dim]; // add 2*eta2*b_k
		}
	}
	
	protected void gradientByR2() {
		double sim;
		int jk;
		
		//Add the R2 for the current the pair (user, neighbor).
		//R2 = [a for the current user,...m_dim, b for the current user..., a for the first neighbor..., b for the first neighbor....]
		for(int j=0; j<m_neighbors.size(); j++){
			sim = m_neighborSims.get(j);
			for(int k=0; k<m_dim; k++){
				jk = (j+1)*m_dim*2+k; // get the index for corresponding user
				
				//Update for the user.
				m_g[k] += 2*m_eta3*sim*(m_As[k]-m_As[jk]); //ak for the current user.
				m_g[k+m_dim] += 2*m_eta3*sim*(m_As[m_dim+k]-m_As[jk+m_dim]); //bk for the current user.
				
				//Update for the neighbor.
				m_g[jk] += 2*m_eta3*sim*(m_As[jk]-m_As[k]);//ak for the neighbor.
				m_g[jk+m_dim] += 2*m_eta3*sim*(m_As[jk+m_dim]-m_As[m_dim+k]);//bk for the neighbor.
			}
		}
	}
	
	//Calculate the gradients for the use in LBFGS.
	public void calculateGradients(ArrayList<_Review> trainSet){
		super.calculateGradients(trainSet);
		gradientByR2();
	}
	
	//Return the transformed matrix.
	public double[] getCoLinAdaptA(){
		return m_A;
	}
	
	//Concatenate current user and neighbors' A matrix.
	public void setAs(){
		m_As = new double[(m_neighbors.size()+1)*m_dim*2];
		System.arraycopy(m_A, 0, m_As, 0, m_dim*2);
		for(int i=0; i<m_neighbors.size(); i++){
			System.arraycopy(m_neighbors.get(i).getCoLinAdaptA(), 0, m_As, m_dim*2*(i+1), m_dim*2);
		}
	}
	
	//Add one predicted result to the 
	public void addOnePredResult(int predL, int trueL){
		if(m_perfStat == null){
			m_perfStat = new _PerformanceStat(predL, trueL);
		} else{
			m_perfStat.addOnePredResult(predL, trueL);
		}
	}
	
	//Train each user's model with training reviews.
	public void train(ArrayList<_Review> trainSet){
		super.train(trainSet);
		updateAll(); //Update afterwards.
	}
	
	public void updateAll(){
		double[] Amat;
		
		System.arraycopy(m_As, 0, m_A, 0, m_dim*2);
		for(int i=0 ; i<m_neighbors.size(); i++){
			Amat = m_neighbors.get(i).getCoLinAdaptA();
			System.arraycopy(m_As, m_dim*2*(i+1), Amat, 0, m_dim*2);
		}
	}
	
	//Update the A matrix with the new value.
	public void updateA(double[] a){
		m_A = a;
	}
}
