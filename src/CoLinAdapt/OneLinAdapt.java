package CoLinAdapt;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.TreeMap;

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
	double[] m_weights; //The weights of each user, which is also the model of the user.
	TreeMap<Integer, ArrayList<Integer>> m_featureGroupIndex; //key: group index, values: feature indexes belonging to that group.
	_PerformanceStat m_perfStat; //Performance of the user's prediction.
	
	//Trade-off parameters.
	double[] m_A; //The transformation matrix which is 2*k+2 dimension.
	double m_eta1; // weight for scaling.
	double m_eta2; // weight for shifting.
	
	double[] m_diag; //parameter used in lbfgs.
	double[] m_g;//optimized gradients. 
	
	protected int[][] m_predStat;
	
	public OneLinAdapt(_User u, int fg, int fn, TreeMap<Integer, ArrayList<Integer>> featureGroupIndex){
		m_user = u;
		m_featureNo = fn;
		m_dim = fg + 1;//fg is the total number of feature groups.
		
		m_featureGroupIndex = featureGroupIndex;
		m_weights = new double[m_dim*2];//one term for bias
		m_A = new double[m_dim*2];//Two bias terms.
		m_g = new double[m_dim*2];
		m_diag = new double[m_dim*2];
		
		m_eta1 = 0.5;
		m_eta2 = 0.5;
		m_predStat = new int[2][m_user.getReviewSize()];
	}   
	
	public OneLinAdapt(_User u, int fg, int fn, TreeMap<Integer, ArrayList<Integer>> featureGroupIndex, double[] globalWeights){
		m_user = u;
		m_featureNo = fn;
		m_dim = fg + 1;//fg is the total number of feature groups.
		
		m_featureGroupIndex = featureGroupIndex;
		m_weights = globalWeights;//one term for bias
		m_A = new double[m_dim*2];//Two bias terms.
		m_g = new double[m_dim*2];
		m_diag = new double[m_dim*2];
		
		m_eta1 = 0.5;
		m_eta2 = 0.5;
		m_predStat = new int[2][m_user.getReviewSize()];
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
	
	//Global weights are fixed for all users while transformation matrix is personalized for each user.  
	//The first term is the bias, w[0]=a[0]*w[0]+b[0].
	public double[] getTransformedWeights(){
		double[] weights = new double[m_featureNo+1];
		//Use two for loops to access all features indexed by group indexes.
		for(int groupIndex: m_featureGroupIndex.keySet()){
			ArrayList<Integer> groupFeatures = m_featureGroupIndex.get(groupIndex);
			for(int featureIndex: groupFeatures)
				//m_A[groupIndex]=a, m_A[groupIndex+groupSize+1]=b, wi*a_{g_i}+b_{g_i}
				weights[featureIndex] = m_A[groupIndex]*m_weights[featureIndex] + m_A[m_dim + groupIndex]; 
		}
//		weights[0] = m_A[0]*m_weights[0] + m_A[m_dim];//the dimension of a and b is (featureGroupNo+1).
		return weights;
	}
	
	//Predict a new review.
	public int predict(_Review review){
		_SparseFeature[] fv = review.getSparse();
		int predL = 0;
		// Calculate each class's probability.P(yd=1|xd)=1/(1+e^{-(AW)^T*xd})
		double p1 = logit(fv, getTransformedWeights());
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
			Pi = logit(fv, getTransformedWeights());
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
	
	//P(y=1|x)=1/1+exp(-w^T*x);P(Y=0|x)=1-P(y=1|x).w' = A*w
	public double logit(_SparseFeature[] fv, double[] weights){
		double value = -weights[0];
		int index = 0;
		for(_SparseFeature f: fv){
			index = f.getIndex();
			value -= weights[index+1]*f.getValue();
		}
		return 1/(1+Math.exp(value));
	}
	
	//Calculate the gradients for the use in LBFGS.
	public void calculateGradients(ArrayList<_Review> trainSet){
		double sumA = 0, sumB = 0, value = 0;
		double Pi = 0;//Pi = P(yd=1|xd);
		int Yi;
		
		m_g = new double[m_dim*2];
		//Update gradients one review by one review.
		for(_Review review: trainSet){
			Yi = review.getYLabel();
			Pi = logit(review.getSparse(), getTransformedWeights());
			
//			//index by feature group indexes.
//			m_g[0] -= (Yi - Pi)*m_weights[0]; //a[0] = w0*x0; x0=1???
//			m_g[m_dim] -= (Yi - Pi);//b[0]
			
			for(int k: m_featureGroupIndex.keySet()){ //k starts from 0.
				ArrayList<Integer> featureIndexes = m_featureGroupIndex.get(k);
				for(int index: featureIndexes){
					value = findFeatureValue(review, index);
					sumA += m_weights[index]*value;//accumulate: \sum_{i, g(i)=k}{wi*xi}
					sumB += value;//accumulate: \sum_{i, g(i)=k}{xi}
				}
				m_g[k] -= (Yi - Pi)*sumA;//update a with (Yi-Pi)*\sum_{i}{wi*xi}
				m_g[k+m_dim] -= (Yi - Pi)*sumB;//update b with (Yi-Pi)*\sum_{i}{xi}
				sumA = 0;
				sumB = 0;
			}
		}
		//Add the regularization parts.
		for(int i=0; i<m_dim; i++){
			m_g[i] += 2*m_eta1*(m_A[i]-1);// add 2*eta1*(ak-1)
			m_g[i+m_dim] += 2*m_eta2*m_A[i+m_dim]; // add 2*eta2*bk
		}
		double mag = 0;
		for(int i=0; i<m_g.length; i++){
			mag += m_g[i]*m_g[i];
		}
		System.out.println("Gradient mag is " + mag);
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
	
	public void fillTrueLabels(int[] trueLs){
		m_predStat[0] = trueLs;
	}
	
	public void fillPredLabels(int[] predLs){
		m_predStat[1] = predLs;
	}
}
