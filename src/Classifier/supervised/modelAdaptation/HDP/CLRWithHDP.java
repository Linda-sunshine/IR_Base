package Classifier.supervised.modelAdaptation.HDP;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.apache.commons.math.special.Gamma;
//import cern.jet.random.Gamma;
import cern.jet.random.tfloat.FloatUniform;
import structures._Doc;
import structures._HDPThetaStar;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import structures._Review.rType;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess._DPAdaptStruct;
/***
 * This class implements the clustered logistic regression with HDP introduced.
 * Instead of assigning each user to one group, each user is currently a mixture of global user groups.
 * @author lin
 */
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

/**
 * Notes on Sep 30:
 * 1. We can randomly assign cluster to each review at the initial stage.
 * 2. In sampling z_ij, the k is fixed, thus \phi_k and \psi_k can be used directly
 * in the calculation of likelihood.
 * 3. In sampling \gamma, multiple sampling is to introduce more randomness.
 * 4. In M step, we need to optimize both \psi and \phi.
 * 5. Also, we would better construct different features for multinomial distribution.
 * 6. We don't consider the super user for the \psi.
 * 7. We can design a data structure which extends from thetastar to contain both \psi and \phi.
 */
public class CLRWithHDP extends CLRWithDP {

	protected int m_initK = 10;//assume we have 10 global groups at the beginning.
	//\alpha is the concentration parameter for the first layer.
	protected double m_eta = 6;//concentration parameter for second layer DP.  
	protected double m_beta = 6; //concentration parameter for \psi.
	protected double[] m_globalLM;//the global language model serving as the prior for psi.
	public static _HDPThetaStar[] m_hdpThetaStars = new _HDPThetaStar[1000];//phi+psi
	protected DirichletPrior m_D0; //Dirichlet prior.
	protected double[] m_gamma = new double[1000];//global mixture proportion.
	protected HashMap<String, Double> m_stirlings; //store the calculated stirling numbers.
	// We don't have psi for the super model.
	
	public CLRWithHDP(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel) {
		super(classNo, featureSize, featureMap, globalModel);
		m_kBar = m_initK;//init kBar.
		m_stirlings = new HashMap<String, Double>();
	}
	
	@Override
	public String toString() {
		return String.format("CLRWithHDP[dim:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:%.3f,#Iter:%d,N(%.3f,%.3f)]", m_dim, m_M, m_alpha, m_eta, m_beta, m_eta1, m_numberOfIterations, m_abNuA[0], m_abNuA[1]);
	}
	
	public void setGlobalLM(double[] lm){
		m_globalLM = lm;
	}
	@Override
	protected void init(){
		initPriorD0();
		initPriorG0();
		initAssignment();

		super.init();

		//Initialize the structures for multi-threading.
		if (m_multiThread) {
			int numberOfCores = Runtime.getRuntime().availableProcessors();
			m_fValues = new double[numberOfCores];
			m_gradients = new double[numberOfCores][]; 
		}
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		
		for(_User user:userList)
			m_userList.add(new _HDPAdaptStruct(user));
		m_pWeights = new double[m_gWeights.length];		
	}
	
	//Init the dirichlet prior.
	public void initPriorD0(){
		m_D0 = new DirichletPrior();//dirichlet distribution for psi and gamma.
		m_D0.sampling(m_gamma, m_kBar+1, m_alpha);
	}
	
	//Randomly assign user reviews to k user groups.
	public void initAssignment(){
		for(int k=0; k<m_kBar; k++){
			m_hdpThetaStars[k] = new _HDPThetaStar(m_dim);
			m_G0.sampling(m_hdpThetaStars[k].getModel()); //sample \phi for each group.
			//init psi with the global language model.
			if(m_hdpThetaStars[k].getPsiModel() == null)
				m_hdpThetaStars[k].initPsi(m_dim-1);
			System.arraycopy(m_globalLM, 0, m_hdpThetaStars[k].getPsiModel(), 0, m_globalLM.length);
		}
		int rndIndex = 0;
		double rnd = 0;
		_HDPThetaStar curTheta;
		_HDPAdaptStruct user;
		for(_AdaptStruct u: m_userList){
			user = (_HDPAdaptStruct) u;
			for(_Review r: user.getReviews()){
				rnd = Math.random() * m_kBar;
				rndIndex = findIndex(rnd);
				
				//find the random theta and update the setting.
				curTheta = m_hdpThetaStars[rndIndex];
				curTheta.updateMemCount(1);
				curTheta.addOneReview(r);
				r.setHDPThetaStar(curTheta);
				user.incHDPThetaStarMemSize(curTheta);
			} 
		}
	}
	
	public int findIndex(double rnd){
		int index = -1;
		if(rnd == 0) 
			index = 0;
		else if(rnd%1 == 0)//if rnd is marginal.
			index = (int)rnd-1;
		else
			index = (int)rnd;
		return index;
	}
	@Override
	//Initialize the phi for each review.
	protected void initThetaStars(){
		m_pNewCluster = Math.log(m_beta) - Math.log(m_M);//to avoid repeated computation
		_HDPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				if (r.getType() != rType.ADAPTATION)
					continue; // only touch the adaptation data
				sampleOneInstance(user, r);
			}
		}		
	}
	
	//Sample auxiliary \phis for further use, also sample one \psi in case we get the new cluster.
	public void sampleThetaStars(){
		for(int m=m_kBar; m<m_kBar+m_M; m++){
			if (m_hdpThetaStars[m] == null)
				m_hdpThetaStars[m] = new _HDPThetaStar(m_dim);
			m_G0.sampling(m_hdpThetaStars[m].getModel());//getModel-> get \phi.
		}
	}
	
	//Assign cluster to each review.
	public void sampleOneInstance(_HDPAdaptStruct user, _Review r){
		double likelihood, logSum = 0, gamma_k;
		int k;
		
		//Step 1: reset thetaStars for the auxiliary thetaStars.
		sampleThetaStars();
		
		//Step 2: sample thetaStar based on the loglikelihood of p(z=k|\gamma,\eta)p(y|x,\phi)p(x|\psi)
		for(k=0; k<m_kBar+m_M; k++){
			r.setHDPThetaStar(m_hdpThetaStars[k]);
			
			//loglikelihood of y, i.e., p(y|x,\phi)
			likelihood = calcLogLikelihood(r); 
			if(Double.isNaN(likelihood))
				System.out.println("NaN!!!");
			//p(z=k|\gamma,\eta)
			gamma_k = (k < m_kBar) ? m_gamma[k]:m_gamma[m_kBar];
			likelihood += Math.log(user.getHDPThetaMemSize(m_hdpThetaStars[k])+m_eta*gamma_k);
			if(Double.isNaN(likelihood)){
				System.out.println("NaN!!!");
				System.out.println(user.getHDPThetaMemSize(m_hdpThetaStars[k]));
				System.out.println(Math.log(user.getHDPThetaMemSize(m_hdpThetaStars[k])+m_eta*gamma_k));
			}
			//loglikelihood of x, i.e., p(x|\psi)	
			likelihood += calcLogLikelihood_x(r);
			if(Double.isNaN(likelihood))
				System.out.println("NaN!!!");
			m_hdpThetaStars[k].setProportion(likelihood);//this is in log space!
			if(k==0) logSum = likelihood;
			else logSum = Utils.logSum(logSum, likelihood);
			if(Double.isNaN(logSum))
				System.out.println("NaN!!!");
		};
		//Sample group k with likelihood.
		k = sampleInLogSpace(logSum);
		
		//Step 3: update the setting after sampling z_ij.
		m_hdpThetaStars[k].updateMemCount(1);//-->1
		m_hdpThetaStars[k].addOneReview(r);//-->2
		r.setHDPThetaStar(m_hdpThetaStars[k]);//-->3
		
		//If we get a new cluster, sample \psi for the new one.
		if(k >= m_kBar){
			m_hdpThetaStars[k].initPsi(m_dim-1);
			System.arraycopy(m_globalLM, 0, m_hdpThetaStars[k].getPsiModel(), 0, m_globalLM.length);
//			m_D0.sampling(m_hdpThetaStars[k].getPsiModel(), m_beta);
		}
		
		//Update the user info with the newly sampled hdpThetaStar.
		user.incHDPThetaStarMemSize(m_hdpThetaStars[k]);//-->4
		
		if(k >= m_kBar){
			swapTheta(m_kBar, k);
			appendOneDim4Gamma();
			m_kBar++;
		}
	}
	// Since we have a new group, we will assign a new weight to this dimension.
	protected void appendOneDim4Gamma(){
		double re = m_gamma[m_kBar];
		double rnd = cern.jet.random.Beta.staticNextDouble(1, m_alpha);
		m_gamma[m_kBar] = rnd*re;
		m_gamma[m_kBar+1] = (1-rnd)*re;
	}
	
	//Sample hdpThetaStar with likelihood.
	public int sampleInLogSpace(double logSum){
		logSum += Math.log(FloatUniform.staticNextFloat());//we might need a better random number generator
		int k = 0;
		double newLogSum = m_hdpThetaStars[0].getProportion();
		do {
			if (newLogSum>=logSum)
				break;
			k++;
			newLogSum = Utils.logSum(newLogSum, m_hdpThetaStars[k].getProportion());
		} while (k<m_kBar+m_M);
		if (k==m_kBar+m_M) {
			System.err.println("[Warning]Hit the very last element in theatStar!");
			k--; // we might hit the very last
		}
		return k;
	}
	@Override
	protected void swapTheta(int a, int b) {
		_HDPThetaStar cTheta = m_hdpThetaStars[a];
		m_hdpThetaStars[a] = m_hdpThetaStars[b];
		m_hdpThetaStars[b] = cTheta;// kBar starts from 0, the size decides how many are valid.
	}
	
	//Swap values at indexes a and b.
	protected void swapGamma(int a, int b){
		double val = m_gamma[a];
		m_gamma[a] = m_gamma[b];
		m_gamma[b] = val;
	}
	//Calculate the function value of the new added instance.
	protected double calcLogLikelihood(_Review r){
		double L = 0, Pi = 0, sum; //log likelihood.
		// log likelihood given by the logistic function.
		sum = Utils.dotProduct(r.getHDPThetaStar().getModel(), r.getSparse(), 0);
		Pi = Utils.logistic(sum);
		if(r.getYLabel() == 1) {
			if (Pi>0.0)
				L += Math.log(Pi);					
			else
				L -= Utils.MAX_VALUE;
		} else {
			if (Pi<1.0)
				L += Math.log(1 - Pi);					
			else
				L -= Utils.MAX_VALUE;
		}
		return L;
	}		

	public double calcLogLikelihood_x(_Review r){
		double L = 0, beta_v = m_beta/(m_dim-1), sum = 0;
		if(r.getHDPThetaStar().getPsiModel() == null){
			//construct the feature vector.
			double[] mij = new double[m_dim-1];
			for(_SparseFeature fv: r.getSparse()){
				mij[fv.getIndex()] = fv.getValue();
				sum += fv.getValue();
			}
			//Gamma:org.apache.commons.math.special.Gamma
			L += Gamma.logGamma(m_beta);
			L -= Gamma.logGamma(m_beta+sum);	
			//for those v with mij,v=0, frac = \gamma(beta_v)/\gamma(beta_v)=1, log frac = 0.
			for(_SparseFeature fv: r.getSparse())
				L += Gamma.logGamma(beta_v+fv.getValue()) - Gamma.logGamma(beta_v);
			return L;
		}
		double[] psi = r.getHDPThetaStar().getPsiModel();
		for(_SparseFeature fv: r.getSparse()){
			L += fv.getValue()*Math.log(psi[fv.getIndex()]);// do we need the coefficients? factorial?
		}
		//add the factorial part in front of likelihood???
		return L;
	}
	
	// The main MCMC algorithm, assign each review to clusters.
	protected void calculate_E_step(){
		_HDPThetaStar curThetaStar;
		_HDPAdaptStruct user;
		int index;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				curThetaStar = r.getHDPThetaStar();
				
				//Step 1: remove the current review from the thetaStar.
				curThetaStar.updateMemCount(-1);
				curThetaStar.rmReview(r);
				if(curThetaStar.getMemSize() == 0) {// No data associated with the cluster.
					index = findHDPThetaStar(curThetaStar);
					swapTheta(m_kBar-1, index); // move it back to \theta*
					swapGamma(m_kBar-1, index); // swap gammas for later use.
					// swap \gamma_index and \gamma_k(weight for last cluster), add \gamma_e to \gamma_k.
					m_gamma[m_kBar-1] += m_gamma[m_kBar];//recycle the weight of gamma[index].
					m_kBar --;
				}
				sampleOneInstance(user, r);
			}
		}
		//Update gamma based on the current assignment.
		updateGamma(); // why for loop BETA_K times?
	}
	
	//Sample how many local groups inside user reviews.
	protected int sampleH(_HDPAdaptStruct user, _HDPThetaStar s){
		int n = user.getHDPThetaMemSize(s);
		if(n==1) return 1;//s(1,1)=1
		double[] prob = new double[n];
		//Find corresponding gamma value.
		int index = findHDPThetaStar(s);
		double etaGammak = Math.log(m_eta) + Math.log(m_gamma[index]);
		//the number of local groups lies in the range [1, n];
		for(int h=1; h<=n; h++){
			double stir = stirling(n, h);
			prob[h-1] = h*etaGammak + Math.log(stir);
		}
		//h starts from 0, we want the number of tables here.		
		return Utils.sampleInLogArray(prob, n)+1;
	}
	
	// n is the total number of observation under group k for the user.
	// h is the number of tables in group k for the user.
	public double stirling(int n, int h){
		if(n==h) return 1;
		if(h==0 || h>n) return 0;
		String key = n+"@"+h;
		if(m_stirlings.containsKey(key))
			return m_stirlings.get(key);
		else
			return stirling(n-1, h-1)+(n-1)*stirling(n-1,h);
	}
	
	//Sample the global mixture proportion, \gamma~Dir(m1, m2,..,\alpha)
	protected void sampleGamma(){
		double alpha = cern.jet.random.Gamma.staticNextDouble(m_alpha, 1);
		double sum = alpha;
		for(int k=0; k<m_kBar; k++){
			m_gamma[k] = cern.jet.random.Gamma.staticNextDouble(m_hdpThetaStars[k].m_hSize, 1);
			sum += m_gamma[k];
		}
		for(int k=0; k<m_kBar; k++) m_gamma[k]/=sum;
		m_gamma[m_kBar] = alpha/sum;//\gamma_e.
	}
	
	protected void updateGamma(){
		for(int k=0; k<m_kBar; k++)
			m_hdpThetaStars[k].m_hSize = 0;
		int index;
		_HDPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_HDPThetaStar s: user.getHDPThetaMemSizeMap().keySet()){
				index = findHDPThetaStar(s);
				m_hdpThetaStars[index].m_hSize += sampleH(user, m_hdpThetaStars[index]);
			}
		}
		sampleGamma();
	}
	
	@Override
	// Assign index to each set of parameters.
	protected void assignClusterIndex(){
		for(int i=0; i<m_kBar; i++)
			m_hdpThetaStars[i].setIndex(i);
	}
	
	// Sample the weights given the cluster assignment.
	protected double calculate_M_step(){
		assignClusterIndex();
		
		//Optimize language model parameters with MLE.
		MLELanguageModels();
		
		//Optimize logistic regression parameters with lbfgs.
		MLELogisticModels();
		
		return 0;
	}
	
	//MLE to estimate language models.
	public void MLELanguageModels(){
		double[] prob = new double[m_dim];
		double sum = 0;
		for(_HDPThetaStar theta: m_hdpThetaStars){
			Arrays.fill(prob, 0);
			for(_Review r: theta.getReviews()){
				for(_SparseFeature fv: r.getSparse()){
					prob[fv.getIndex()] += fv.getValue();
					sum += fv.getValue();
				}
			}
			//Estimate the prob for language model.
			for(int i=0; i<prob.length; i++) prob[i]/=sum;
			theta.updatePsiModel(prob);
		}
	}
	protected double logLikelihood() {
		_HDPAdaptStruct user;
		double fValue = 0;
		
		// Use instances inside one cluster to update the thetastar.
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				fValue -= calcLogLikelihood(r);
				gradientByFunc(user, r, 1); // calculate the gradient by the review.
			}
		}
		return fValue;
	}
	
	protected void gradientByFunc(_AdaptStruct u, _Doc r, double weight, double[] g) {
		_DPAdaptStruct user = (_DPAdaptStruct)u;
		_Review review = (_Review) r;
		int n; // feature index
		int cIndex = user.getThetaStar().getIndex();
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error,cannot find the theta star!");
		
		int offset = m_dim*cIndex;
		double delta = weight * (review.getYLabel() - logit(review.getSparse(), review, user));
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
		//Bias term.
		g[offset] -= delta; //x0=1

		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			g[offset + n] -= delta * fv.getValue();
		}
	}
	
	protected double logit(_SparseFeature[] fvs, _Review r, _AdaptStruct u){
		double sum = Utils.dotProduct(r.getHDPThetaStar().getModel(), fvs, 0);
		return Utils.logistic(sum);
	}
	
	//Language model estimates.
	public void MLELogisticModels(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, oldFValue = Double.MAX_VALUE;
		int displayCount = 0;		

		initLBFGS();// init for lbfgs.
		try{
			do{
				Arrays.fill(m_g, 0); // initialize gradient
				
				//regularization part
				fValue = calculateR1();
				if (m_multiThread)
					fValue += logLikelihood_MultiThread();
				else
					fValue += logLikelihood();
				
				if (m_displayLv==2) {
					System.out.print("Fvalue is " + fValue + "\t");
					gradientTest();
				} else if (m_displayLv==1) {
					if (fValue<oldFValue)
						System.out.print("o");
					else
						System.out.print("x");
					
					if (++displayCount%100==0)
						System.out.println();
				} 
				
				LBFGS.lbfgs(m_g.length, 5, m_models, fValue, m_g, false, m_diag, iprint, 1e-2, 1e-16, iflag);//In the training process, A is updated.
				setThetaStars_phi();
				oldFValue = fValue;
			} while(iflag[0] != 0);
			System.out.println();
		} catch(ExceptionWithIflag e) {
			System.err.println("LBFGS FAILURE!");
			e.printStackTrace();
		}	
	}
	
	// Assign the optimized weights to the cluster.
	protected void setThetaStars_phi(){
		double[] beta;
		for(int i=0; i<m_kBar; i++){
			beta = m_hdpThetaStars[i].getModel();
			System.arraycopy(m_models, i*m_dim, beta, 0, m_dim);
		}
	}
	
	// The main EM algorithm to optimize cluster assignment and distribution parameters.
	public double train(){
		System.out.println(toString());
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		int count = 0;
		
		init(); // clear user performance and init cluster assignment		
		
		// Burn in period.
		while(count++ < m_burnIn){
			calculate_E_step();
			lastLikelihood = calculate_M_step();
		}
		
		// EM iteration.
		for(int i=0; i<m_numberOfIterations; i++){
			// Cluster assignment, thinning to reduce auto-correlation.
			calculate_E_step();
			
			// Optimize the parameters
			curLikelihood = calculate_M_step();

			delta = (lastLikelihood - curLikelihood)/curLikelihood;
			
			if (i%m_thinning==0)
				evaluateModel();
			
			printInfo();
			System.out.print(String.format("[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.3f\n", i, curLikelihood, delta));
			if(Math.abs(delta) < m_converge)
				break;
			lastLikelihood = curLikelihood;
		}

		evaluateModel(); // we do not want to miss the last sample?!
//		setPersonalizedModel();
		return curLikelihood;
	}
	
	protected int findHDPThetaStar(_HDPThetaStar theta) {
		for(int i=0; i<m_kBar; i++)
			if (theta == m_hdpThetaStars[i])
				return i;
		
		System.err.println("[Error]Hit unknown theta star when searching!");
		return -1;// impossible to hit here!
	}
}
