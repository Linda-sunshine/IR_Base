package CoLinAdapt;

import java.util.ArrayList;
import java.util.Arrays;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._PerformanceStat;
import structures._Review;
import structures._SparseFeature;
import structures._User;

public class OneLinAdapt {
	//The number of feature groups k, so the total number of dimensions of weights is 2k+2.
	_User m_user;
	int m_dim;//The dimension of scaling and shifting parameters.
	int m_featureNo; //The total number of features.
	int[] m_featureGroupIndexes;
	double[] m_weights; //The weights of each user, which is also the model of the user.
	_PerformanceStat m_perfStat; //Performance of the user's prediction.
	
	//Trade-off parameters.
	double[] m_A; //The transformation matrix which is 2*k+2 dimension.
	double m_eta1; // weight for scaling.
	double m_eta2; // weight for shifting.
	
	double[] m_diag; //parameter used in lbfgs.
	double[] m_g;//optimized gradients. 
	
	/****Online mode: the treu labels and predicted labels for each prediction
	 * true labels: [0, 1, 0, 1]
	 * pred labels: [1, 0, 1, 1]
	 *****Batch mode: TP table*/
//	protected int[][] m_predStat; 
	
	public OneLinAdapt(_User u, int fg, int fn){
		m_user = u;
		m_featureNo = fn;
		m_dim = fg + 1;//fg is the total number of feature groups.
		
		m_weights = new double[m_dim*2];//one term for bias
		m_A = new double[m_dim*2];//Two bias terms.
		m_g = new double[m_dim*2];
		m_diag = new double[m_dim*2];
		
		m_eta1 = 0.5;
		m_eta2 = 0.5;
	}   
	
	public OneLinAdapt(_User u, int fn, int fg, double[] globalWeights, int[] featureGroupIndexes){
		m_user = u;
		m_featureNo = fn;
		m_dim = fg + 1;//fg is the total number of feature groups.
		
		m_weights = globalWeights;//one term for bias
		m_featureGroupIndexes = featureGroupIndexes;
		m_A = new double[m_dim*2];//Two bias terms.
		m_g = new double[m_dim*2];
		m_diag = new double[m_dim*2];
		
		m_eta1 = 0.5;
		m_eta2 = 0.5;
	}  
	
	//Initialize the weights of the transformation matrix.
	public void init(){
		//Initial weights should be from global model.
		for(int i=0; i < m_dim; i++)
			m_A[i] = 1;//Initialize scaling to be 1 and shifting be 0.
	}
	
	public void initLBFGS(){
		Arrays.fill(m_diag, 0);
		Arrays.fill(m_g, 0);
	}
	
	//Create instance for the model with pred labels and true labels, used for online mode since we need the middle data.
	public void setPerformanceStat(int[] trueLs, int[] predLs){
		m_perfStat = new _PerformanceStat(trueLs, predLs);
	}
	
	//Create instance for the model with existing TPTable, used for batch mode.
	public void setPerformanceStat(int[][] TPTable){
		m_perfStat = new _PerformanceStat(TPTable);
	}
	
	//Predict a new review.
	public int predict(_Review review){
		_SparseFeature[] fv = review.getSparse();
		int predL = 0;
		// Calculate each class's probability.P(yd=1|xd)=1/(1+e^{-(AW)^T*xd})
		double p1 = logit(fv);
		//Decide the label for the review.
		if(p1 > 0.5) 
			predL = 1;
		return predL;
	}
	
	//Calculate the function value of the new added instance.
	public double calculateFunctionValue(ArrayList<_Review> trainSet){
		double fValue = 0;
		int Yi;
		_SparseFeature[] fv;
		double Pi = 0;
		//Init: R1 = (a[0]-1)^2 + b[0]^2;
		double R1 = m_eta1*(m_A[0]-1)*(m_A[0]-1) +
					m_eta2*m_A[m_dim]*m_A[m_dim];
		
		for(_Review review: trainSet){
			Yi = review.getYLabel();
			fv = review.getSparse();
			Pi = logit(fv);
			if(Yi == 1)
				fValue += Math.log(Pi);
			else 
				fValue += Math.log(1 - Pi);
		}
		//Add regularization parts.
		for(int i=0; i<m_dim; i++){
			R1 += m_eta1*(m_A[i]-1)*(m_A[i]-1);//(a[i]-1)^2
			R1 += m_eta2*(m_A[m_dim+i])*(m_dim+i);//b[i]^2
		}
		System.out.println("Fvalue is " + (-fValue+R1));
		return -fValue + R1;
	}
	
	// We can do A*w*x at the same time to reduce computation.
	public double logit(_SparseFeature[] fvs){
		double value = -m_A[0]*m_weights[0] + m_A[m_dim];//Bias term: w0*a0+b0.
		int featureIndex = 0, groupIndex = 0;
		for(_SparseFeature fv: fvs){
			featureIndex = fv.getIndex() + 1;
			groupIndex = m_featureGroupIndexes[featureIndex];
			value -= (m_A[groupIndex]*m_weights[featureIndex] + m_A[groupIndex + m_dim])*fv.getValue();
		}
		return 1/(1+Math.exp(value));
	}
	
	//Calculate the gradients for the use in LBFGS.
	public void calculateGradients(ArrayList<_Review> trainSet){
		double Pi = 0;//Pi = P(yd=1|xd);
		int Yi, featureIndex = 0, groupIndex = 0;
		
		m_g = new double[m_dim*2];
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
		//Add the regularization parts.
		for(int i=0; i<m_dim; i++){
			m_g[i] += 2*m_eta1*(m_A[i]-1);// add 2*eta1*(ak-1)
			m_g[i+m_dim] += 2*m_eta2*m_A[i+m_dim]; // add 2*eta2*bk
		}
		double magA = 0, magB = 0 ;
		for(int i=0; i<m_dim; i++){
			magA += m_g[i]*m_g[i];
			magB += m_g[i+m_dim]*m_g[i+m_dim];
		}
		System.out.format("Gradient magnitude for A: %.5f\tB: %.5f\n", magA, magB);
	}
	
	//In this function, given a review and feature index, if the review has this feature, return the value of this feature, otherwise, return 0.
	public double findFeatureValue(_Review review, int index){
		_SparseFeature[] fvs = review.getSparse();
		if(fvs == null || fvs.length == 0)
			return 0;
		int start = 0, end = fvs.length, middle = 0;
		while(middle < end && start < end){
			middle = (start + end) / 2;
			if(fvs[middle].getIndex() == index)
				return fvs[middle].getValue();
			else if(fvs[middle].getIndex() < index)
				start = middle + 1;
			else
				end = middle - 1;
		}
		return 0;
	}
	
	// The same function with different names.
	public void train(_Review review){
		ArrayList<_Review> trainSet = new ArrayList<_Review>();
		trainSet.add(review);
		train(trainSet);
	}
	//Train each user's model with training reviews.
	public void train(ArrayList<_Review> trainSet){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue;
		int fSize = m_dim * 2;
		
		initLBFGS();
		try{
			do{
				fValue = calculateFunctionValue(trainSet);
				calculateGradients(trainSet);
				LBFGS.lbfgs(fSize, 6, m_A, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-20, iflag);//In the training process, A is updated.
			} while(iflag[0] != 0);
		} catch(ExceptionWithIflag e) {
			e.printStackTrace();
		}
	}
	
//	public void fillTrueLabels(int[] trueLs){
//		if(m_predStat.length == 0 || m_predStat.equals(null))
//			m_predStat = new int[2][m_user.getReviewSize()];
//		m_predStat[0] = trueLs;
//	}
//	
//	public void fillPredLabels(int[] predLs){
//		if(m_predStat.length == 0 || m_predStat.equals(null))
//			m_predStat = new int[2][m_user.getReviewSize()];
//		m_predStat[1] = predLs;
//	}
	
	//Batch mode: given a set of reviews and accumulate the TP table.
	public void test(ArrayList<_Review> testSet){
		int trueL = 0, predL = 0;
		int[][] TPTable = new int[2][2];
		for(int i=0; i<testSet.size(); i++){
			trueL = testSet.get(i).getYLabel();
			predL = predict(testSet.get(i));
			TPTable[trueL][predL]++;
		}
		setPerformanceStat(TPTable);
	}
}
