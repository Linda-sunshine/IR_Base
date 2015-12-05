package CoLinAdapt;

import java.util.ArrayList;
import java.util.Arrays;

import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

import structures._PerformanceStat;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

public class CoLinAdapt extends LinAdapt{
	double m_eta3; // Weight for R2.
	double[] m_As;
	ArrayList<_User> m_neighbors;
	ArrayList<Integer> m_neighborIndexes;
	ArrayList<Double> m_neighborSims;

	public CoLinAdapt(int fg, int fn, double[] globalWeights, int[] featureGroupIndexes) {
		super(fg, fn, globalWeights, featureGroupIndexes);
		m_eta3 = 0.5;
	}
	
	public void initLBFGS(){
		if(m_g == null)
			m_g = new double[m_dim*2*(m_neighbors.size()+1)];
		if(m_diag == null)
			m_diag = new double[m_dim*2*(m_neighbors.size()+1)];
		
		Arrays.fill(m_diag, 0);
		Arrays.fill(m_g, 0);
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
		int featureIndex = 0, groupIndex = 0;
		for(_SparseFeature fv: fvs){
			featureIndex = fv.getIndex() + 1;
			groupIndex = m_featureGroupIndexes[featureIndex];
			value += (m_As[groupIndex]*m_weights[featureIndex] + m_As[groupIndex + m_dim])*fv.getValue();
		}
		return 1/(1+Math.exp(-value));
	}

	// Calculate the new function value.
	public double calculateFunctionValue(ArrayList<_Review> trainSet){
		int Yi;
		_SparseFeature[] fv;
		double Pi = 0, sim = 0;
		double fValue = 0, L = 0, R1 = 0, R2 = 0;

		for(_Review review: trainSet){
			Yi = review.getYLabel();
			fv = review.getSparse();
			Pi = logit(fv);
			if(Yi == 1)
				L += Math.log(Pi);
			else 
				L += Math.log(1 - Pi);
		}
		
		//Add R1.
		for(int i=0; i<m_dim; i++){
			R1 += m_eta1*(m_As[i]-1)*(m_As[i]-1);//(a[i]-1)^2
			R1 += m_eta2*(m_As[m_dim+i])*(m_As[m_dim+i]);//b[i]^2
		}
		fValue = -L + R1;
		
		// Add the R2 part to the function value.
		for(int i=0; i<m_neighbors.size(); i++){
			sim = m_neighborSims.get(i);
			for(int j=0; j<m_dim; j++){
				//(a_ki-a_kj)^2 + (b_ki-b_kj)^2
				R2 += (m_As[j]-m_As[(i+1)*m_dim*2+j])*(m_As[j]-m_As[(i+1)*m_dim*2+j])+(m_As[j+m_dim]-m_As[(i+1)*m_dim*2+j+m_dim])*(m_As[j+m_dim]-m_As[(i+1)*m_dim*2+j+m_dim]);
			}
			fValue += sim * m_eta3 * R2;
			R2 = 0;
		}
		System.out.println("Fvalue is " + fValue);
		return fValue;
	}
	
	//Calculate the gradients for the use in LBFGS.
	public void calculateGradients(ArrayList<_Review> trainSet){
		//The gradients are for the current user and neighbors, thus, we use a big matrix to represent it. 
		double Pi = 0, sim = 0;//Pi = P(yd=1|xd);
		int Yi, featureIndex = 0, groupIndex = 0;
		
		m_g = new double[(m_neighbors.size()+1) * m_dim*2];//The big gradients matrix contains current user and all neighbors.
		//Update gradients one review by one review.
		for(_Review review: trainSet){
			Yi = review.getYLabel();
			Pi = logit(review.getSparse());
			
			//Bias term.
			m_g[0] -= (Yi - Pi)*m_weights[0]; //a[0] = w0*x0; x0=1
			m_g[m_dim] -= (Yi - Pi);//b[0]
			
			//Traverse all the feature dimension to calculate the gradient.
			for(_SparseFeature fv: review.getSparse()){
				featureIndex = fv.getIndex() + 1;
				groupIndex = m_featureGroupIndexes[featureIndex];
				m_g[groupIndex] -= (Yi - Pi) * m_weights[featureIndex] * fv.getValue();
				m_g[m_dim + groupIndex] -= (Yi - Pi) * fv.getValue();  
			}
		}
		
		//Add the R1 for the current user.
		for(int i=0; i<m_dim; i++){
			m_g[i] += 2*m_eta1*(m_As[i]-1);// add 2*eta1*(ak-1)
			m_g[i+m_dim] += 2*m_eta2*m_As[i+m_dim]; // add 2*eta2*bk
		}
		
		//Add the R2 for the current the pair (user, neighbor).
		//R2 = [a for the current user,...m_dim, b for the current user..., a for the first neighbor..., b for the first neighbor....]
		for(int i=0; i<m_neighbors.size(); i++){
			sim = m_neighborSims.get(i);
			for(int j=0; j<m_dim; j++){
				//Update for the user.
				m_g[j] += 2*m_eta3*sim*(m_As[j]-m_As[(i+1)*m_dim*2+j]); //ak for the current user.
				m_g[j+m_dim] += 2*m_eta3*sim*(m_As[m_dim+j]-m_As[(i+1)*m_dim*2+j+m_dim]); //bk for the current user.
				//Update for the neighbor.
				m_g[j+2*m_dim*(i+1)] += 2*m_eta3*sim*(m_As[(i+1)*m_dim*2+j]-m_As[j]);//ak for the neighbor.
				m_g[j+2*m_dim*(i+1)+m_dim] += 2*m_eta3*sim*(m_As[(i+1)*m_dim*2+j+m_dim]-m_As[m_dim+j]);//bk for the neighbor.
			}
		}
		double magA = 0;
		for(int i=0; i<m_g.length; i++){
			magA += m_g[i]*m_g[i];
		}
		System.out.format("Gradient magnitude: %.5f\n", magA);
	}
	
	//Return the transformed matrix.
	public double[] getCoLinAdaptA(){
		return m_A;
	}
	
	//Concatenate current user and neighbors' A matrix.
	public void setAs(){
		m_As = new double[(m_neighbors.size()+1)*m_dim*2];
		Utils.fillPartOfArray(0, m_dim*2, m_As, m_A); // Add the user's own A matrix.
		for(int i=0; i<m_neighbors.size(); i++){
			Utils.fillPartOfArray((i+1)*m_dim*2, m_dim*2, m_As, m_neighbors.get(i).getCoLinAdaptA());
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
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue;
		int fSize = m_dim*2*(m_neighbors.size()+1);
		initLBFGS();
		try{
			do{
				fValue = calculateFunctionValue(trainSet);
				calculateGradients(trainSet);
				LBFGS.lbfgs(fSize, 6, m_As, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-10, iflag);//In the training process, A is updated.
			} while(iflag[0] != 0);
		} catch(ExceptionWithIflag e) {
			e.printStackTrace();
		}
		updateAll(); //Update afterwards.
	}
	public void updateAll(){
		double[] newA;
		m_A = Arrays.copyOfRange(m_As, 0, m_dim*2);
		for(int i=0 ; i<m_neighbors.size(); i++){
			newA = Arrays.copyOfRange(m_As, m_dim*2*(i+1), m_dim*2*(i+2));
			m_neighbors.get(i).updateA(newA);
		}
	}
	
	//Update the A matrix with the new value.
	public void updateA(double[] a){
		m_A = a;
	}

}
