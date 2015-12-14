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

public class SyncCoLinAdapt extends CoLinAdapt {
	double[] m_allAs; //All users transformation matrixes.
	double[] m_allGs; //All users gradients.
	ArrayList<_User> m_users;
	ArrayList<String> m_userIndexes;
	double[] m_similarity;//It contains all user pair's similarity.

	public SyncCoLinAdapt(int fg, int fn, double[] globalWeights, int[] featureGroupIndexes, ArrayList<_User> users) {
		super(fg, fn, globalWeights, featureGroupIndexes);
		m_users = users;
	}

	public void init(){
		_User user;
		m_allAs = new double[m_users.size()*m_dim*2];
		m_allGs = new double[m_users.size()*m_dim*2];
		m_userIndexes = new ArrayList<String>();
		
		// Fill the As with each user's A.
		for(int i=0; i<m_users.size(); i++){
			user = m_users.get(i);
			m_userIndexes.add(user.getUserID());
			Utils.fillPartOfArray(i*m_dim*2, m_dim*2, m_allAs, user.getCoLinAdaptA());
		}
	}
	
	public void initLBFGS(){
		if(m_allGs == null)
			m_allGs = new double[m_dim*2*m_users.size()];
		if(m_diag == null)
			m_diag = new double[m_dim*2*m_users.size()];
		
		Arrays.fill(m_diag, 0);
		Arrays.fill(m_allGs, 0);
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
	
	//Since we have a big A matrix, the calculation of logit function is different.
	public double logit(_SparseFeature[] fvs, int userIndex){
		double value = m_allAs[m_dim*2*userIndex]*m_weights[0] + m_allAs[m_dim*2*userIndex+m_dim];//Bias term: w0*a0+b0.
		int featureIndex = 0, groupIndex = 0;
		for(_SparseFeature fv: fvs){
			featureIndex = fv.getIndex() + 1;
			groupIndex = m_featureGroupIndexes[featureIndex];
			value += (m_allAs[m_dim*2*userIndex+groupIndex]*m_weights[featureIndex] + m_allAs[m_dim*2*userIndex+groupIndex + m_dim])*fv.getValue();
		}
		return 1/(1+Math.exp(-value));
	}
	
	// Calculate the new function value.
	public double calculateFunctionValue(ArrayList<_Review> trainSet){
		int Yi, userIndex;
		_SparseFeature[] fv;
		double Pi = 0, sim = 0;
		double fValue = 0, L = 0, R1 = 0, oneR2 = 0;

		//Likelihood is for every training review, sum up.
		for (_Review review : trainSet) {
			userIndex = m_userIndexes.indexOf(review.getUserID());
			Yi = review.getYLabel();
			fv = review.getSparse();
			Pi = logit(fv, userIndex);
			if (Yi == 1)
				L += Math.log(Pi);
			else
				L += Math.log(1 - Pi);
		}

		// Add R1 for all users.
		for(int i=0; i<m_users.size(); i++){
			for (int j = 0; j < m_dim; j++) {
				R1 += m_eta1*(m_allAs[i*m_dim*2+j]-1)*(m_allAs[i*m_dim*2+j]-1);// (a[k]-1)^2
				R1 += m_eta2*(m_allAs[i*m_dim*2+m_dim+j])*(m_allAs[i*m_dim*2+m_dim+j]);// b[k]^2
			}
		}
		fValue = -L + R1;

		int[] neighborIndexes;
		// Add the R2 part to the function value.
		for(int i=0; i<m_users.size(); i++){
			neighborIndexes = m_users.get(i).getNeighborIndexes();
			for(int index: neighborIndexes){//Access each neighbor.
				sim = m_similarity[getIndex(i, index)];
				for(int j=0; j<m_dim; j++){
					oneR2 += (m_allAs[m_dim*2*i+j]-m_allAs[m_dim*2*index+j])*(m_allAs[m_dim*2*i+j]-m_allAs[m_dim*2*index+j])//ak^2
						    +(m_allAs[m_dim*2*i+j+m_dim]-m_allAs[m_dim*2*index+j+m_dim])*(m_allAs[m_dim*2*i+j+m_dim]-m_allAs[m_dim*2*index+j+m_dim]);//bk^2 
				}
				fValue += m_eta3*sim*oneR2;
				oneR2 = 0;
			}
		}
//		System.out.println("Fvalue is " + fValue);
		return fValue;
	}

	// Calculate the gradients for the use in LBFGS.
	public void calculateGradients(ArrayList<_Review> trainSet) {
		double Pi = 0, sim = 0;// Pi = P(yd=1|xd);
		int Yi, userIndex = 0, featureIndex = 0, groupIndex = 0;
		//m_allGs = new double[m_users.size()*m_dim*2];
		Arrays.fill(m_allGs, 0);
		int[] neighborIndexes;
		
		// Update gradients one review by one review.
		for (_Review review : trainSet) {
			userIndex = m_userIndexes.indexOf(review.getUserID());
			Yi = review.getYLabel();
			Pi = logit(review.getSparse(), userIndex);

			// Bias term.
			m_allGs[m_dim*2*userIndex] -= (Yi - Pi) * m_weights[0]; // a[0] = w0*x0; x0=1
			m_allGs[m_dim*2*userIndex+m_dim] -= (Yi - Pi);// b[0]

			// Traverse all the feature dimension to calculate the gradient.
			for (_SparseFeature fv : review.getSparse()) {
				featureIndex = fv.getIndex() + 1;
				groupIndex = m_featureGroupIndexes[featureIndex];
				m_allGs[m_dim*2*userIndex+groupIndex] -= (Yi - Pi)*m_weights[featureIndex]*fv.getValue();
				m_allGs[m_dim*2*userIndex+m_dim+groupIndex] -= (Yi - Pi) * fv.getValue();
			}
		}

		// Add the R1 for all users.
		for (int i=0; i<m_users.size(); i++) {
			//Update one user's R1.
			for(int j=0; j<m_dim; j++){
				m_allGs[m_dim*2*i+j] += 2*m_eta1*(m_allAs[m_dim*2*i+j]-1);// add 2*eta1*(ak-1)
				m_allGs[m_dim*2*i+j+m_dim] += 2*m_eta2*m_allAs[m_dim*2*i+j+m_dim]; // add 2*eta2*bk
			}
		}

		// Add the R2 for all users.
		for (int i=0; i<m_users.size(); i++) {// Access each user.
			neighborIndexes = m_users.get(i).getNeighborIndexes();
			for(int index: neighborIndexes){//Access each neighbor.
				sim = m_similarity[getIndex(i, index)];
				for(int j=0; j<m_dim; j++){
					//Update a: current user: a - a_neighbor; neighbor: a_neihgbor - a.
					m_allGs[m_dim*2*i+j] += 2*sim*m_eta3*(m_allAs[m_dim*2*i+j]-m_allAs[m_dim*2*index+j]);
					m_allGs[m_dim*2*i+j+m_dim] += 2*sim*m_eta3*(m_allAs[m_dim*2*i+j+m_dim]-m_allAs[m_dim*2*index+j+m_dim]);
					//Update b: similar with a.
					m_allGs[m_dim*2*index+j] += 2*sim*m_eta3*(m_allAs[m_dim*2*index+j]-m_allAs[m_dim*2*i+j]);
					m_allGs[m_dim*2*index+j+m_dim] += 2*sim*m_eta3*(m_allAs[m_dim*2*index+j+m_dim]-m_allAs[m_dim*2*i+j+m_dim]);
				}
			}
		}
		double magA = 0;
		for (int i = 0; i < m_allGs.length; i++) {
			magA += m_allGs[i]*m_allGs[i];
		}
//		System.out.format("Gradient magnitude: %.5f\n", magA);
	}

	// Return the transformed matrix.
	public double[] getCoLinAdaptA() {
		return m_A;
	}

//	// Concatenate current user and neighbors' A matrix.
//	public void setAs() {
//		m_As = new double[(m_neighbors.size() + 1) * m_dim * 2];
//		Utils.fillPartOfArray(0, m_dim * 2, m_As, m_A); // Add the user's own A
//														// matrix.
//		for (int i = 0; i < m_neighbors.size(); i++) {
//			Utils.fillPartOfArray((i + 1) * m_dim * 2, m_dim * 2, m_As,
//					m_neighbors.get(i).getCoLinAdaptA());
//		}
//	}

	// Add one predicted result to the
	public void addOnePredResult(int predL, int trueL) {
		if (m_perfStat == null) {
			m_perfStat = new _PerformanceStat(predL, trueL);
		} else {
			m_perfStat.addOnePredResult(predL, trueL);
		}
	}

	// Train each user's model with training reviews.
	public void train(ArrayList<_Review> trainSet) {
		int[] iflag = { 0 }, iprint = { -1, 3 };
		double fValue;
		int fSize = m_dim * 2 * m_users.size();
		initLBFGS();
		try {
			do {
				fValue = calculateFunctionValue(trainSet);
				calculateGradients(trainSet);
				LBFGS.lbfgs(fSize, 6, m_allAs, fValue, m_allGs, false, m_diag, iprint, 1e-4, 1e-10, iflag);
			} while (iflag[0] != 0);
		} catch (ExceptionWithIflag e) {
			e.printStackTrace();
		}
		updateAll(); // Update afterwards.
	}

	public void updateAll() {
		double[] newA;
		for (int i=0; i<m_users.size(); i++) {
			newA = Arrays.copyOfRange(m_allAs, m_dim*2*i, m_dim*2*(i+1));
			m_users.get(i).updateA(newA);
		}
	}
	
	//Batch mode: given a set of reviews and accumulate the TP table.
	public void test(ArrayList<_Review> testSet) {
		int trueL = 0, predL = 0, userIndex = 0;
		_Review review;
		for (int i = 0; i < testSet.size(); i++) {
			review = testSet.get(i);
			trueL = review.getYLabel();
			userIndex = m_userIndexes.indexOf(review.getUserID());
			predL = predict(review, userIndex);
			m_users.get(userIndex).getCoLinAdapt().addOnePredResult(predL, trueL);
		}
	}
	
	//Predict a new review.
	public int predict(_Review review, int userIndex){
		_SparseFeature[] fv = review.getSparse();
		int predL = 0;
		// Calculate each class's probability.P(yd=1|xd)=1/(1+e^{-(AW)^T*xd})
		double p1 = logit(fv, userIndex);
		//Decide the label for the review.
		if(p1 > 0.5) 
			predL = 1;
		return predL;
	}
	
	public void setSimilarities(double[] sims){
		m_similarity = sims;
	}
}
