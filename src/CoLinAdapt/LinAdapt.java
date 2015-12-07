package CoLinAdapt;

import java.util.ArrayList;
import java.util.Arrays;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._PerformanceStat;
import structures._Review;
import structures._SparseFeature;

public class LinAdapt {
	//The number of feature groups k, so the total number of dimensions of weights is 2k+2.
	int m_dim;//The dimension of scaling and shifting parameters (including the bias term)
	int m_vSize; // total number of optimizable variables
	int[] m_featureGroupIndexes; // bias term is at position 0
	double[] m_weights; //global model weight
	_PerformanceStat m_perfStat; //Performance of the user's prediction.
	
	//Trade-off parameters	
	double m_eta1; // weight for scaling.
	double m_eta2; // weight for shifting.
	
	double[] m_A; //The transformation matrix which is 2*k+2 dimension.
	double[] m_diag; //parameter used in lbfgs.
	double[] m_g;//optimized gradients. 
	
	/****Online mode: the true labels and predicted labels for each prediction
	 * true labels: [0, 1, 0, 1]
	 * pred labels: [1, 0, 1, 1]
	 *****Batch mode: TP table*/
//	protected int[][] m_predStat; 
	
	public LinAdapt(int fg, int fn, double[] globalWeights, int[] featureGroupIndexes){
		m_dim = fg + 1;//fg is the total number of feature groups.
		m_vSize = 2*m_dim;
		
		m_weights = globalWeights;//one term for bias
		m_featureGroupIndexes = featureGroupIndexes;
		m_A = new double[m_dim*2];//Two bias terms.
		
		m_eta1 = 0.5;
		m_eta2 = 0.5;
	}  
	
	//Initialize the weights of the transformation matrix.
	protected void initA(){		
		for(int i=0; i < m_dim; i++)
			m_A[i] = 1;//Initialize scaling to be 1 and shifting be 0.
	}
	
	protected void initLBFGS(){
		if(m_g == null)
			m_g = new double[m_vSize];
		if(m_diag == null)
			m_diag = new double[m_vSize];
		
		Arrays.fill(m_diag, 0);
		Arrays.fill(m_g, 0);
	}
	
//	//Create instance for the model with pred labels and true labels, used for online mode since we need the middle data.
//	public void setPerformanceStat(int[] predLs, int[] trueLs){
//		m_perfStat = new _PerformanceStat(predLs, trueLs);
//	}

	//Create instance for the model with existing TPTable, used for batch mode.
	protected void setPerformanceStat(int[][] TPTable){
		m_perfStat = new _PerformanceStat(TPTable);
	}
	
	//Predict a new review.
	public int predict(_Review review){
		// Calculate each class's probability.P(yd=1|xd)=1/(1+e^{-(AW)^T*xd})
		return logit(review.getSparse())>0.5 ? 1:0;
	}
	
	// We can do A*w*x at the same time to reduce computation.
	protected double logit(_SparseFeature[] fvs){
		double value = m_A[0]*m_weights[0] + m_A[m_dim];//Bias term: w0*a0+b0.
		int n = 0, k = 0; // feature index and feature group index
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupIndexes[n];
			value += (m_A[k]*m_weights[n] + m_A[k + m_dim])*fv.getValue();
		}
		return 1/(1+Math.exp(-value));
	}
	
	//Calculate the function value of the new added instance.
	protected double calculateFunctionValue(ArrayList<_Review> trainSet){
		double L = 0; //log likelihood.
		int Yi;
		_SparseFeature[] fv;
		double Pi = 0, R1 = 0;
		
		for(_Review review: trainSet){
			Yi = review.getYLabel();
			fv = review.getSparse();
			Pi = logit(fv);
			if(Yi == 1)
				L += Math.log(Pi);
			else 
				L += Math.log(1 - Pi);
		}
		
		//Add regularization parts.
		for(int i=0; i<m_dim; i++){
			R1 += m_eta1*(m_A[i]-1)*(m_A[i]-1);//(a[i]-1)^2
			R1 += m_eta2*(m_A[m_dim+i])*(m_A[m_dim+i]);//b[i]^2
		}
		
		return R1 - L;
	}
	
	protected void gradientByFunc(ArrayList<_Review> trainSet) {
		double delta;
		int n, k; // feature index and feature group index
		
		//Update gradients one review by one review.
		for(_Review review: trainSet){
			delta = review.getYLabel() - logit(review.getSparse());
			
			//Bias term.
			m_g[0] -= delta*m_weights[0]; //a[0] = w0*x0; x0=1
			m_g[m_dim] -= delta;//b[0]
			
			//Traverse all the feature dimension to calculate the gradient.
			for(_SparseFeature fv: review.getSparse()){
				n = fv.getIndex() + 1;
				k = m_featureGroupIndexes[n];
				m_g[k] -= delta * m_weights[n] * fv.getValue();
				m_g[m_dim + k] -= delta * fv.getValue();  
			}
		}
	}
	
	//Calculate the gradients for the use in LBFGS.
	protected void gradientByR1(){
		//R1 regularization part
		for(int k=0; k<m_dim; k++){
			m_g[k] += 2 * m_eta1 * (m_A[k]-1);// add 2*eta1*(a_k-1)
			m_g[k+m_dim] += 2 * m_eta2 * m_A[k+m_dim]; // add 2*eta2*b_k
		}
	}
		
	//Calculate the gradients for the use in LBFGS.
	protected void calculateGradients(ArrayList<_Review> trainSet){
		Arrays.fill(m_g, 0);
		gradientByFunc(trainSet);
		gradientByR1();
	}
	
	protected double gradientTest() {
		double magA = 0, magB = 0 ;
		for(int i=0; i<m_dim; i++){
			magA += m_g[i]*m_g[i];
			magB += m_g[i+m_dim]*m_g[i+m_dim];
		}
		System.out.format("Gradient magnitude for a: %.5f, b: %.5f\n", magA, magB);
		return magA + magB;
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
		
		initLBFGS();
		try{
			do{
				fValue = calculateFunctionValue(trainSet);
				System.out.println("Fvalue is " + fValue);
				
				calculateGradients(trainSet);
				gradientTest();
				
				LBFGS.lbfgs(m_vSize, 6, m_A, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-10, iflag);//In the training process, A is updated.
			} while(iflag[0] != 0);
		} catch(ExceptionWithIflag e) {
			e.printStackTrace();
		}
	}
	
	//Batch mode: given a set of reviews and accumulate the TP table.
	public void test(ArrayList<_Review> testSet){
		int trueL = 0, predL = 0;
		int[][] TPTable = new int[2][2];
		for(_Review r:testSet){
			trueL = r.getYLabel();
			predL = predict(r);
			TPTable[trueL][predL]++;
		}
		setPerformanceStat(TPTable);
	}
	
	public double[] getA(){
		return m_A;
	}
}
