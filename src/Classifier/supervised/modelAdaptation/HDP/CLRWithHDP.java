package Classifier.supervised.modelAdaptation.HDP;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithDP;
import cern.jet.random.tdouble.Beta;
import cern.jet.random.tdouble.Gamma;
import cern.jet.random.tfloat.FloatUniform;
import structures.MyPriorityQueue;
import structures._Doc;
import structures._HDPThetaStar;
import structures._RankItem;
import structures._Review;
import structures._thetaStar;
import structures._Review.rType;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

public class CLRWithHDP extends CLRWithDP {

	//\alpha is the concentration parameter for the first layer.
	protected double m_eta = 1.0;//concentration parameter for second layer DP.  
	protected double m_beta = 1.0; //concentration parameter for \psi.
	protected double m_c = 1;//the constant in front of probabilities of language model.
	
	protected double[] m_betas;//concentration vector for the prior of psi.
	public static _HDPThetaStar[] m_hdpThetaStars = new _HDPThetaStar[1000];//phi+psi
	double[] m_cache = new double[1000]; // shared cache space to avoid repeatedly creating new space
	protected DirichletPrior m_D0; //generic Dirichlet prior.
	protected double m_gamma_e = 1.0;
	protected double m_nBetaDir = 0; // normalization constant for Dir(\psi)

	protected HashMap<String, Double> m_stirlings; //store the calculated stirling numbers.
	protected boolean m_newCluster = false; // whether to create new cluster for testing
	protected int m_lmDim = -1; // dimension for language model
	

	public CLRWithHDP(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel, 
			double[] betas, double alpha, double beta, double eta) {
		super(classNo, featureSize, featureMap, globalModel);
		m_D0 = new DirichletPrior();//dirichlet distribution for psi and gamma.
		m_stirlings = new HashMap<String, Double>();
		
		setConcentrationParams(alpha, beta, eta);
		setBetas(betas);
	}
	
	public CLRWithHDP(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel, 
			double[] betas) {
		super(classNo, featureSize, featureMap, globalModel);
		m_D0 = new DirichletPrior();//dirichlet distribution for psi and gamma.
		m_stirlings = new HashMap<String, Double>();
		setBetas(betas);
	}
	
	@Override
	public String toString() {
		return String.format("CLRWithHDP[dim:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:%.3f,#Iter:%d,N(%.3f,%.3f)]", m_dim, m_M, m_alpha, m_eta, m_beta, m_eta1, m_numberOfIterations, m_abNuA[0], m_abNuA[1]);
	}
	
	// Set the constant of \pi_v.
	public void setC(double c){
		m_c = c;
	}
	public void setBetas(double[] lm){
		m_betas = lm;// this is in real space!
		
		m_lmDim = lm.length;
		for(int i=0; i<m_lmDim; i++) {
			m_betas[i] = m_c * m_betas[i] + m_beta;
			m_nBetaDir -= Utils.lgamma(m_betas[i]);
		}
		m_nBetaDir += Utils.lgamma(Utils.sumOfArray(m_betas));
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		
		for(_User user:userList)
			m_userList.add(new _HDPAdaptStruct(user));
		m_pWeights = new double[m_gWeights.length];		
	}
	
	//Randomly assign user reviews to k user groups.
	@Override
	public void initThetaStars(){
		initPriorG0();
		_HDPAdaptStruct user;		
		double L = 0, beta_sum = Utils.sumOfArray(m_betas), betaSum_lgamma = Utils.lgamma(beta_sum), sum = 0;
		int index;
		
		for(_AdaptStruct u: m_userList){
			user = (_HDPAdaptStruct) u;
			for(_Review r: user.getReviews()){
				//for all reviews pre-compute the likelihood of being generated from a random language model				
				L = 0;
				sum = beta_sum;//sum = v*beta+\sum \pi_v(global language model)
				//for those v with mij,v=0, frac = \gamma(beta_v)/\gamma(beta_v)=1, log frac = 0.
				for(_SparseFeature fv: r.getSparse()) {
					index = fv.getIndex();
					sum += fv.getTF();	
					//log \gamma(m_v+\pi_v+beta)/\gamma(\pi_v+beta)
					L += Utils.lgamma(fv.getTF() + m_betas[index]) - Utils.lgamma(m_betas[index]);//logGamma(\beta_i) can be pre-computed for efficiency
				}
				L += betaSum_lgamma - Utils.lgamma(sum);
				r.setL4NewCluster(L);
				
				if (r.getType() == rType.TEST)
					continue;
				
				sampleOneInstance(user, r);
			} 
		}
	}

	//Sample auxiliary \phis for further use, also sample one \psi in case we get the new cluster.
	@Override
	public void sampleThetaStars(){
		double gamma_e = m_gamma_e/m_M;
		for(int m=m_kBar; m<m_kBar+m_M; m++){
			if (m_hdpThetaStars[m] == null){
				if (this instanceof CLinAdaptWithHDP)// this should include all the inherited classes for adaptation based models
					m_hdpThetaStars[m] = new _HDPThetaStar(2*m_dim, gamma_e);
				else
					m_hdpThetaStars[m] = new _HDPThetaStar(m_dim, gamma_e);
			} else
				m_hdpThetaStars[m].setGamma(gamma_e);//to unify the later operations
			
			//sample \phi from Normal distribution.
			m_G0.sampling(m_hdpThetaStars[m].getModel());//getModel-> get \phi.
			
			//we do not need to sample psi since we will integrate it out in likelihood calculation.
		}
	}
	
	//Assign cluster to each review.
	protected void sampleOneInstance(_HDPAdaptStruct user, _Review r){
		double likelihood, logSum = 0, gamma_k;
		int k;
		
		//Step 1: reset thetaStars for the auxiliary thetaStars.
		sampleThetaStars();
		
		//Step 2: sample thetaStar based on the loglikelihood of p(z=k|\gamma,\eta)p(y|x,\phi)p(x|\psi)
		for(k=0; k<m_kBar+m_M; k++){
			r.setHDPThetaStar(m_hdpThetaStars[k]);
			
			//log likelihood of y, i.e., p(y|x,\phi)
			likelihood = calcLogLikelihoodY(r); 
			
			//log likelihood of x, i.e., p(x|\psi)	
			likelihood += calcLogLikelihoodX(r);
			
			//p(z=k|\gamma,\eta)

			gamma_k = m_hdpThetaStars[k].getGamma();
			likelihood += Math.log(user.getHDPThetaMemSize(m_hdpThetaStars[k]) + m_eta*gamma_k);
			
			m_hdpThetaStars[k].setProportion(likelihood);//this is in log space!
			
			if(k==0) 
				logSum = likelihood;
			else 
				logSum = Utils.logSum(logSum, likelihood);
//			System.out.print("likehood:"+likelihood+"\t"+"logsum:"+logSum+"\n");
		}
		
		
		//Sample group k with likelihood.
		k = sampleInLogSpace(logSum);
//		System.out.println(k+"-----------------");
		
		//Step 3: update the setting after sampling z_ij.
		m_hdpThetaStars[k].updateMemCount(1);//-->1
		r.setHDPThetaStar(m_hdpThetaStars[k]);//-->2
		
		//Step 4: Update the user info with the newly sampled hdpThetaStar.
		user.incHDPThetaStarMemSize(m_hdpThetaStars[k], 1);//-->3		

		if(k >= m_kBar){//sampled a new cluster
			m_hdpThetaStars[k].initPsiModel(m_lmDim);
			m_D0.sampling(m_hdpThetaStars[k].getPsiModel(), m_betas, true);//we should sample from Dir(\beta)
			
			double rnd = Beta.staticNextDouble(1, m_alpha);
			m_hdpThetaStars[k].setGamma(rnd*m_gamma_e);
			m_gamma_e = (1-rnd)*m_gamma_e;
			
			swapTheta(m_kBar, k);
			m_kBar++;
		}
	}
	
	//Sample hdpThetaStar with likelihood.
	protected int sampleInLogSpace(double logSum){
		logSum += Math.log(FloatUniform.staticNextFloat());//we might need a better random number generator
		
		int k = 0;
		double newLogSum = m_hdpThetaStars[0].getProportion();
		do {
			if (newLogSum>=logSum)
				break;
			k++;
			newLogSum = Utils.logSum(newLogSum, m_hdpThetaStars[k].getProportion());
		} while (k<m_kBar+m_M);
		
		if (k==m_kBar+m_M)
			k--; // we might hit the very last
		return k;
	}
	
	@Override
	protected void swapTheta(int a, int b) {
		if(a == b) 
			return;//If they are the same, no need to swap.
		
		_HDPThetaStar cTheta = m_hdpThetaStars[a];
		m_hdpThetaStars[a] = m_hdpThetaStars[b];
		m_hdpThetaStars[b] = cTheta;// kBar starts from 0, the size decides how many are valid.
	}
	
	//Calculate the function value of the new added instance.
	protected double calcLogLikelihoodY(_Review r){
		double L = 0, Pi = 0; //log likelihood.
		// log likelihood given by the logistic function.
		Pi = logit(r.getSparse(), r);
		
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

	protected double calcLogLikelihoodX(_Review r){		
		double[] psi = r.getHDPThetaStar().getPsiModel();
		//we will integrate it out
		if(psi == null){
			return r.getL4NewCluster();
		} else {		
			double L = 0;
			for(_SparseFeature fv: r.getSparse())
				L += fv.getTF() * psi[fv.getIndex()];
			return L;
		}
	}
	
	// The main MCMC algorithm, assign each review to clusters.
	protected void calculate_E_step(){
		_HDPThetaStar curThetaStar;
		_HDPAdaptStruct user;
		int index, sampleSize=0;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				if (r.getType() == rType.TEST)
					continue;//do not touch testing reviews!
				
				curThetaStar = r.getHDPThetaStar();
				
				//Step 1: remove the current review from the thetaStar and user side.
				user.incHDPThetaStarMemSize(curThetaStar, -1);				
				curThetaStar.updateMemCount(-1);

				if(curThetaStar.getMemSize() == 0) {// No data associated with the cluster.
					m_gamma_e += curThetaStar.getGamma();
					index = findHDPThetaStar(curThetaStar);
					swapTheta(m_kBar-1, index); // move it back to \theta*
					m_kBar --;
				}
				
				//Step 2: sample new cluster assignment for this review
				sampleOneInstance(user, r);
				
				if (++sampleSize%2000==0) {
					System.out.print('.');
					sampleGamma();//will this help sampling?
					if (sampleSize%100000==0)
						System.out.println();
				}
			}
		}
//		sampleGamma();//will this help sampling?
		System.out.println(m_kBar);
	}
	
	//Sample how many local groups inside user reviews.
	protected int sampleH(_HDPAdaptStruct user, _HDPThetaStar s){
		int n = user.getHDPThetaMemSize(s);
		if(n==1)
			return 1;//s(1,1)=1		

		double etaGammak = Math.log(m_eta) + Math.log(s.getGamma());
		//the number of local groups lies in the range [1, n];
		for(int h=1; h<=n; h++){
			double stir = stirling(n, h);
			m_cache[h-1] = h*etaGammak + Math.log(stir);
		}
		
		//h starts from 0, we want the number of tables here.	
		return Utils.sampleInLogArray(m_cache, n) + 1;
	}
	
	// n is the total number of observation under group k for the user.
	// h is the number of tables in group k for the user.
	double stirling(int n, int h){
		if(n==h) return 1;
		if(h==0 || h>n) return 0;
		String key = n+"@"+h;
		if(m_stirlings.containsKey(key))
			return m_stirlings.get(key);
		else {
			double result = stirling(n-1, h-1) + (n-1)*stirling(n-1, h);
			m_stirlings.put(key, result);
			return result;
		}
	}
	
	//Sample the global mixture proportion, \gamma~Dir(m1, m2,..,\alpha)
	protected void sampleGamma(){

		for(int k=0; k<m_kBar; k++)
			m_hdpThetaStars[k].m_hSize = 0;
		
		_HDPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);			
			for(_HDPThetaStar s:user.getHDPTheta())
				s.m_hSize += sampleH(user, s);
		}		
		
		m_cache[m_kBar] = Gamma.staticNextDouble(m_alpha, 1);//for gamma_e
		
		double sum = m_cache[m_kBar];
		for(int k=0; k<m_kBar; k++){
			m_cache[k] = Gamma.staticNextDouble(m_hdpThetaStars[k].m_hSize+m_alpha, 1);
			sum += m_cache[k];
		}
		
		for(int k=0; k<m_kBar; k++) 
			m_hdpThetaStars[k].setGamma(m_cache[k]/sum);
		
		m_gamma_e = m_cache[m_kBar]/sum;//\gamma_e.
	}
	
	@Override
	// Assign index to each set of parameters.
	protected void assignClusterIndex(){
		for(int i=0; i<m_kBar; i++)
			m_hdpThetaStars[i].setIndex(i);
	}
	
	// Sample the weights given the cluster assignment.
	@Override
	protected double calculate_M_step(){
		assignClusterIndex();		
		
		//Step 1: sample gamma based on the current assignment.
		sampleGamma(); // why for loop BETA_K times?
		
		//Step 2: Optimize language model parameters with MLE.
		//Step 3: Optimize logistic regression parameters with lbfgs.
		return estPsi() + estPhi();
	}
	
	//We should use maximum a posterior to estimate language models.
	public double estPsi(){
		double sum = 0, lmProb[], logLikelihood = 0;
		_HDPThetaStar theta;
		
		//Step 1: reset psi in each cluster
		for(int k=0; k<m_kBar; k++){ 
			theta = m_hdpThetaStars[k];
			lmProb = theta.getPsiModel();
					
			System.arraycopy(m_betas, 0, lmProb, 0, m_lmDim);//start from prior mean vector
		}
		
		//Step 2: accumulate count for each psi accordingly
		_HDPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);		
			for(_Review r: user.getReviews()){
				if (r.getType() == rType.TEST)
					continue;
				
				lmProb = r.getHDPThetaStar().getPsiModel();
				for(_SparseFeature fv: r.getSparse())
					lmProb[fv.getIndex()] += fv.getTF();
			}
		}
		
		//Step 3: normalize the language models and compute the likelihood of Dir(\psi|\beta)
		for(int k=0; k<m_kBar; k++){ 
			theta = m_hdpThetaStars[k];
			lmProb = theta.getPsiModel();
					
			sum = Math.log(Utils.sumOfArray(lmProb));
			for(int v=0; v<m_lmDim; v++) {
				lmProb[v] = Math.log(lmProb[v]) - sum;
				logLikelihood += (m_betas[v]-1) * lmProb[v];//shall we compute the normalization constant in front of Dirichlet?
			}
			logLikelihood += m_nBetaDir;//Dirchlet normalization constant
		}
			
		//Step 4: compute the data likelihood 
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);		
			for(_Review r: user.getReviews()){
				if (r.getType() == rType.TEST)
					continue;
				
				lmProb = r.getHDPThetaStar().getPsiModel();
				for(_SparseFeature fv: r.getSparse())
					logLikelihood += fv.getTF() * lmProb[fv.getIndex()];
			}
		}
		return logLikelihood;
	}
	
	@Override
	protected double logLikelihood() {
		_HDPAdaptStruct user;
		double fValue = 0;
		
		// Use instances inside one cluster to update the thetastar.
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				if (r.getType() == rType.TEST)
					continue;
				fValue -= calcLogLikelihoodY(r);
				gradientByFunc(user, r, 1); // calculate the gradient by the review.
			}
		}
		return fValue;
	}

	protected double logLikelihood_MultiThread() {
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();		
		
		//init the shared structure		
		Arrays.fill(m_fValues, 0);
		for(int k=0; k<numberOfCores; ++k){
			Arrays.fill(m_gradients[k], 0);
			
			threads.add((new Thread() {
				int core, numOfCores;
				double[] m_gradient, m_fValue;

				@Override
				public void run() {
					_HDPAdaptStruct user;
					try {						
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							user = (_HDPAdaptStruct)m_userList.get(i+core);
							
							for(_Review review:user.getReviews()){
								if (review.getType() != rType.ADAPTATION )//&& review.getType() != rType.TEST)

									continue;	
								
								m_fValue[core] -= calcLogLikelihoodY(review);
								gradientByFunc(user, review, 1.0, this.m_gradient);//weight all the instances equally
							}			
						}
					} catch(Exception ex) {
						ex.printStackTrace(); 
					}
				}
				
				private Thread initialize(int core, int numOfCores, double[] gradient, double[] f) {
					this.core = core;
					this.numOfCores = numOfCores;
					this.m_gradient = gradient;
					this.m_fValue = f;
					
					return this;
				}
			}).initialize(k, numberOfCores, m_gradients[k], m_fValues));
			
			threads.get(k).start();
		}
		
		for(int k=0;k<numberOfCores;++k){
			try {
				threads.get(k).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for(int k=0;k<numberOfCores;++k)
			Utils.scaleArray(m_g, m_gradients[k], 1);
		return Utils.sumOfArray(m_fValues);
	}

	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc r, double weight, double[] g) {
		_Review review = (_Review) r;
		int n; // feature index
		int cIndex = review.getHDPThetaStar().getIndex();
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error,cannot find the HDP theta star!");
		
		int offset = m_dim*cIndex;
		double delta = weight * (review.getYLabel() - logit(review.getSparse(), review));
		
		//Bias term.
		g[offset] -= delta; //x0=1

		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			g[offset + n] -= delta * fv.getValue();
		}
	}
	
	protected double logit(_SparseFeature[] fvs, _Review r){
		double sum = Utils.dotProduct(r.getHDPThetaStar().getModel(), fvs, 0);
		return Utils.logistic(sum);
	}
	
	// Assign the optimized \phi to the cluster.
	@Override
	protected void setThetaStars(){
		double[] beta;
		for(int i=0; i<m_kBar; i++){
			beta = m_hdpThetaStars[i].getModel();
			System.arraycopy(m_models, i*m_dim, beta, 0, m_dim);
		}
	}
	
	// The main EM algorithm to optimize cluster assignment and distribution parameters.
	@Override
	public double train(){
		System.out.println(toString());
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		int count = 0;
		
		init(); // clear user performance and init cluster assignment	
//		calculate_M_step();
		
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
	
	@Override
	protected void accumulateClusterModels(){
		if (m_models==null || m_models.length!=getVSize())
			m_models = new double[getVSize()];
		
		for(int i=0; i<m_kBar; i++)
			System.arraycopy(m_hdpThetaStars[i].getModel(), 0, m_models, m_dim*i, m_dim);
	}
	
	@Override
	protected double calculateR1(){
		double R1 = 0;
		for(int i=0; i<m_kBar; i++)
			R1 += m_G0.logLikelihood(m_hdpThetaStars[i].getModel(), m_eta1, 0);//the last is dummy input
		
		// Gradient by the regularization.
		if (m_G0.hasVctMean()) {//we have specified the whole mean vector
			for(int i=0; i<m_kBar*m_dim; i++) 
				m_g[i] += m_eta1 * (m_models[i]-m_gWeights[i%m_dim]) / (m_abNuA[1]*m_abNuA[1]);
		} else {//we only have a simple prior
			for(int i=0; i<m_kBar*m_dim; i++)
				m_g[i] += m_eta1 * (m_models[i]-m_abNuA[0]) / (m_abNuA[1]*m_abNuA[1]);
		}
		return R1;
	}
	
	@Override
	// After we finish estimating the clusters, we calculate the probability of each testing review belongs to each cluster.
	// Indeed, it is for per review, for inheritance we don't change the function name.
	protected void calculateClusterProbPerUser(){
		double prob, logSum;
		double[] probs;
		if(m_newCluster) 
			probs = new double[m_kBar+1];
		else 
			probs = new double[m_kBar];
		
		_HDPAdaptStruct user;
		_HDPThetaStar curTheta;
		
		//sample a new cluster parameter first.
		if(m_newCluster) {
			m_hdpThetaStars[m_kBar].setGamma(m_gamma_e);//to make it consistent since we will only use one auxiliary variable
			m_G0.sampling(m_hdpThetaStars[m_kBar].getModel());
		}
		int count = 0;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				if (r.getType() != rType.TEST)
					continue;				
				
				for(int k=0; k<probs.length; k++){
					curTheta = m_hdpThetaStars[k];
					r.setHDPThetaStar(curTheta);
					prob = calcLogLikelihoodX(r) + Math.log(user.getHDPThetaMemSize(curTheta) + m_eta*curTheta.getGamma());//this proportion includes the user's current cluster assignment
					probs[k] = prob;
				}
			
				logSum = Utils.logSumOfExponentials(probs);
				for(int k=0; k<probs.length; k++)
					probs[k] -= logSum;
				if(Utils.max(probs)==0) count++;
				r.setClusterPosterior(probs);
			}
		}
		if(count > 0)
			System.err.println("Prob zero vector count:" + count);
	}
	
	@Override
	public void printInfo(){
		//clear the statistics
		for(int i=0; i<m_kBar; i++) 
			m_hdpThetaStars[i].resetCount();

		//collect statistics across users in adaptation data
		_HDPThetaStar theta = null;
		_HDPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++) {
			user = (_HDPAdaptStruct)m_userList.get(i);
			for(_Review r: user.getReviews()){
				if (r.getType() != rType.ADAPTATION)
					continue; // only touch the adaptation data
				else{
					theta = r.getHDPThetaStar();
					if(r.getYLabel() == 1) theta.incPosCount(); 
					else theta.incNegCount();
				}
			}
		}
		System.out.print("[Info]Clusters:");
		for(int i=0; i<m_kBar; i++)
			System.out.format("%s\t", m_hdpThetaStars[i].showStat());	
		System.out.print(String.format("\n[Info]%d Clusters are found in total!\n", m_kBar));
	}
	
	// Set the parameters.
	public void setConcentrationParams(double alpha, double eta, double beta){
		m_alpha = alpha;
		m_eta = eta;
		m_beta = beta;
	}
	
	public void setMultiTheadFlag(boolean b){
		m_multiThread = b;
	}
}
