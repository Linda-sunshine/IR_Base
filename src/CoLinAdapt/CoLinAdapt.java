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
	int m_index; //The index of this user in the user set.
	double m_eta3; // Weight for scaling in R2, a vector.
	double m_eta4; //Weight for shifting in R2, b vector.
	double[] m_As;
	ArrayList<_User> m_neighbors;
	ArrayList<Integer> m_neighborIndexes;
	
	double[] m_similarity;//It contains all user pair's similarity. Since we need to update it for current user and neighbor.

//	ArrayList<Double> m_neighborSims;
	
	double m_R1; //The R1 of the current user.
	double[] m_R2Vct; //The vector contains all the R2s of neighbors.
	
	public CoLinAdapt(int fg, int fn, double[] globalWeights, int[] featureGroupIndexes) {
		super(fg, fn, globalWeights, featureGroupIndexes);
		m_eta3 = 0.5;
		m_eta4 = 0.5;
	}
	
	public CoLinAdapt(int index, int fg, int fn, double[] globalWeights, int[] featureGroupIndexes) {
		super(fg, fn, globalWeights, featureGroupIndexes);
		m_index = index;
		m_eta3 = 0.5;
		m_eta4 = 0.5;
	}
	
	public void setCoefficients(double a4r1, double b4r1, double a4r2, double b4r2){
		m_eta1 = a4r1;
		m_eta2 = b4r1;
		m_eta3 = a4r2;
		m_eta4 = b4r2;
	}
	
	public double[] getR2Vct(){
		return m_R2Vct;
	}
	
	//Pass the reference of similarity to the CoLinAdapt model.
	public void setSimilarity(double[] sims){
		m_similarity = sims;
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
	
//	public void setNeighborSims(ArrayList<Double> sims){
//		m_neighborSims = new ArrayList<Double>(sims);
//	}
	
	public void setNeighborIndexes(ArrayList<Integer> indexes){
		m_neighborIndexes = indexes;
	}
	
//	public ArrayList<Double> getNeighborSims(){
//		return m_neighborSims;
//	}
	
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

	//Pre-compute the R1 and R2 beforehand.
	public void calculateR1R2Vct(){
		//R1
		m_R1 = 0;
		for(int k=0; k<m_dim; k++){
			m_R1 += m_eta1*(m_As[k]-1)*(m_As[k]-1);//(a[i]-1)^2
			m_R1 += m_eta2*(m_As[m_dim+k])*(m_As[m_dim+k]);//b[i]^2
		}
		
		// Add the R2 part to the function value.
		double a4r2 = 0, b4r2 = 0;
		m_R2Vct = new double[m_neighbors.size() * 2];
		for(int i=0; i<m_neighbors.size(); i++){
			for(int k=0; k<m_dim; k++){
				//(a_ki-a_kj)^2 + (b_ki-b_kj)^2
				a4r2 += (m_As[k]-m_As[(i+1)*m_dim*2+k])*(m_As[k]-m_As[(i+1)*m_dim*2+k]);
				
				a4r2 += (m_As[k+m_dim]-m_As[(i+1)*m_dim*2+k+m_dim])*(m_As[k+m_dim]-m_As[(i+1)*m_dim*2+k+m_dim]);
			}
			m_R2Vct[i] = a4r2;
			m_R2Vct[i+m_neighbors.size()] = b4r2;
			a4r2 = 0; b4r2 = 0; // Clear the two values.
		}
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
			else{
				if( Pi != 1)
					L += Math.log(1 - Pi);
			}
		}
		
		//Add R1.
//		for(int k=0; k<m_dim; k++){
//			R1 += m_eta1*(m_As[k]-1)*(m_As[k]-1);//(a[i]-1)^2
//			R1 += m_eta2*(m_As[m_dim+k])*(m_As[m_dim+k]);//b[i]^2
//		}
		fValue = -L + m_R1;
		
//		// Add the R2 part to the function value.
//		for(int i=0; i<m_neighbors.size(); i++){
//			sim = m_neighborSims.get(i);
//			for(int k=0; k<m_dim; k++){
//				//(a_ki-a_kj)^2 + (b_ki-b_kj)^2
//				R2 += (m_As[k]-m_As[(i+1)*m_dim*2+k])*(m_As[k]-m_As[(i+1)*m_dim*2+k])+(m_As[k+m_dim]-m_As[(i+1)*m_dim*2+k+m_dim])*(m_As[k+m_dim]-m_As[(i+1)*m_dim*2+k+m_dim]);
//			}
//			fValue += sim * m_eta3 * R2;
//			R2 = 0;
//		}
//		System.out.println("Fvalue is " + fValue);
		
		for(int i=0; i<m_neighbors.size(); i++){
			sim = m_similarity[getIndex(m_index, m_neighbors.get(i).getIndex())];
			fValue += sim * (m_eta3 * m_R2Vct[i] + m_eta4 * m_R2Vct[i + m_neighbors.size()]);
		}
		return fValue;
	}
	
	//Calculate the gradients for the use in LBFGS.
	public double calculateGradients(ArrayList<_Review> trainSet){
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
			sim = m_similarity[getIndex(m_index, m_neighbors.get(i).getIndex())];
			for(int j=0; j<m_dim; j++){
				//Update for the user.
				m_g[j] += 2*m_eta3*sim*(m_As[j]-m_As[(i+1)*m_dim*2+j]); //ak for the current user.
				m_g[j+m_dim] += 2*m_eta4*sim*(m_As[m_dim+j]-m_As[(i+1)*m_dim*2+j+m_dim]); //bk for the current user.
				//Update for the neighbor.
				m_g[j+2*m_dim*(i+1)] += 2*m_eta3*sim*(m_As[(i+1)*m_dim*2+j]-m_As[j]);//ak for the neighbor.
				m_g[j+2*m_dim*(i+1)+m_dim] += 2*m_eta4*sim*(m_As[(i+1)*m_dim*2+j+m_dim]-m_As[m_dim+j]);//bk for the neighbor.
			}
		}
		double magA = 0;
		for(int i=0; i<m_g.length; i++){
			magA += m_g[i]*m_g[i];
		}
//		System.out.format("Gradient magnitude: %.5f\n", magA);
		return magA;
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
	public boolean train(ArrayList<_Review> trainSet){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, curMag = 0, preMag = 0;
		int fSize = m_dim*2*(m_neighbors.size()+1);
		initLBFGS();
		try{
			do{
				calculateR1R2Vct();//Since we are using it in optimization.
				fValue = calculateFunctionValue(trainSet);
				curMag = calculateGradients(trainSet);
				if(curMag == 0 || Math.abs(curMag - preMag) < 1e-16){
					System.out.print("*");
					break;
				}
				preMag = curMag;
				LBFGS.lbfgs(fSize, 6, m_As, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-10, iflag);//In the training process, A is updated.
			} while(iflag[0] != 0);
		} catch(ExceptionWithIflag e) {
//			e.printStackTrace();
			return false;
		}
		updateAll(); //Update afterwards.
		return true;
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
	
	public int getIndex(int i, int j){
		//Swap i and j.
		if(i < j){
			int t = j;
			j = i;
			i = t;
		}
		return i*(i-1)/2+j;
	}
}
