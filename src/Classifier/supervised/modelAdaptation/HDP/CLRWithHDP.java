package Classifier.supervised.modelAdaptation.HDP;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

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
import cern.jet.random.tdouble.Beta;
import cern.jet.random.tdouble.Gamma;
import cern.jet.random.tfloat.FloatUniform;
import structures._Doc;
import structures._HDPThetaStar;
import structures._Review;
import structures._Review.rType;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

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
	protected double m_eta = 1.0;//concentration parameter for second layer DP.  
	protected double m_beta = 1.0; //concentration parameter for \psi.
	protected double[] m_globalLM;//the global language model serving as the prior for psi.
	public static _HDPThetaStar[] m_hdpThetaStars = new _HDPThetaStar[1000];//phi+psi
	protected DirichletPrior m_D0; //generic Dirichlet prior.
	protected double[] m_gamma = new double[1000];//global mixture proportion.
	protected HashMap<String, Double> m_stirlings; //store the calculated stirling numbers.

	protected boolean m_newCluster = false;
	protected int m_lmDim = -1; // dimension for language model
	
	public CLRWithHDP(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel) {
		super(classNo, featureSize, featureMap, globalModel);
		m_kBar = m_initK;//init kBar.
		m_D0 = new DirichletPrior();//dirichlet distribution for psi and gamma.
		m_stirlings = new HashMap<String, Double>();
	}
	
	@Override
	public String toString() {
		return String.format("CLRWithHDP[dim:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:%.3f,#Iter:%d,N(%.3f,%.3f)]", m_dim, m_M, m_alpha, m_eta, m_beta, m_eta1, m_numberOfIterations, m_abNuA[0], m_abNuA[1]);
	}
	
	public void setGlobalLM(double[] lm){
		m_globalLM = lm;
		m_lmDim = lm.length;
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
		
		m_D0.sampling(m_gamma, m_kBar+1, m_alpha, false);
		m_pNewCluster = Math.log(m_alpha) - Math.log(m_M);//to avoid repeated computation
		
		for(int k=0; k<m_kBar; k++){
			m_hdpThetaStars[k] = new _HDPThetaStar(m_dim, m_lmDim);
			
			//sample \phi from Normal distribution.
			m_G0.sampling(m_hdpThetaStars[k].getModel()); 
			
			//sample \psi from Dirichlet with the global language model.
			m_D0.sampling(m_hdpThetaStars[k].getPsiModel(), m_globalLM, true);
		}
		
		int rndIndex = 0;
		_HDPThetaStar curTheta;
		_HDPAdaptStruct user;
		for(_AdaptStruct u: m_userList){
			user = (_HDPAdaptStruct) u;
			for(_Review r: user.getReviews()){
				if (r.getType() == rType.TEST)
					continue;
				
				rndIndex = (int)(Math.random() * m_kBar);

				//find the random theta and update the setting.
				curTheta = m_hdpThetaStars[rndIndex];
				curTheta.updateMemCount(1);
				curTheta.addOneReview(r);
				r.setHDPThetaStar(curTheta);
				user.updateHDPThetaStarMemSize(curTheta, 1);
			} 
		}
	}

	//Sample auxiliary \phis for further use, also sample one \psi in case we get the new cluster.
	@Override
	public void sampleThetaStars(){
		for(int m=m_kBar; m<m_kBar+m_M; m++){
			if (m_hdpThetaStars[m] == null)
				m_hdpThetaStars[m] = new _HDPThetaStar(m_dim);
			
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
			
			//loglikelihood of y, i.e., p(y|x,\phi)
			likelihood = calcLogLikelihoodY(r); 
			
			//p(z=k|\gamma,\eta)
			gamma_k = (k < m_kBar) ? m_gamma[k]:m_gamma[m_kBar];
			likelihood += Math.log(user.getHDPThetaMemSize(m_hdpThetaStars[k])+m_eta*gamma_k);
			
			//loglikelihood of x, i.e., p(x|\psi)	
			likelihood += calcLogLikelihoodX(r);
			
			m_hdpThetaStars[k].setProportion(likelihood);//this is in log space!
			
			if(k==0) 
				logSum = likelihood;
			else 
				logSum = Utils.logSum(logSum, likelihood);
		};
		
		//Sample group k with likelihood.
		k = sampleInLogSpace(logSum);
		
		//Step 3: update the setting after sampling z_ij.
		m_hdpThetaStars[k].updateMemCount(1);//-->1
		m_hdpThetaStars[k].addOneReview(r);//-->2
		r.setHDPThetaStar(m_hdpThetaStars[k]);//-->3
		
		//Update the user info with the newly sampled hdpThetaStar.
		user.updateHDPThetaStarMemSize(m_hdpThetaStars[k], 1);//-->4
		
		if(k >= m_kBar){
			m_hdpThetaStars[k].initPsiModel(m_lmDim);
			m_D0.sampling(m_hdpThetaStars[k].getPsiModel(), m_globalLM, true);
			
			swapTheta(m_kBar, k);
			appendOneDim4Gamma();
			m_kBar++;
		}
	}
	// Since we have a new group, we will assign a new weight to this dimension.
	protected void appendOneDim4Gamma(){
		double re = m_gamma[m_kBar];
		double rnd = Beta.staticNextDouble(1, m_alpha);
		m_gamma[m_kBar] = rnd*re;
		m_gamma[m_kBar+1] = (1-rnd)*re;
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
		
		if (k==m_kBar+m_M) {
			//System.err.println("[Warning]Hit the very last element in theatStar!");
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
	protected double calcLogLikelihoodY(_Review r){
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

	public double calcLogLikelihoodX(_Review r){
		double L = 0, beta_lgamma = Utils.lgamma(m_beta), sum = 0;
		//we will integrate it out
		if(r.getHDPThetaStar().getPsiModel() == null){
			//for those v with mij,v=0, frac = \gamma(beta_v)/\gamma(beta_v)=1, log frac = 0.
			for(_SparseFeature fv: r.getSparse()) {
				sum += fv.getTF();
				L += Utils.lgamma(m_beta+fv.getTF()) - beta_lgamma;
			}
			
			return L + Utils.lgamma(m_beta*m_lmDim) - Utils.lgamma(m_beta*m_lmDim+sum);
		} else {		
			double[] psi = r.getHDPThetaStar().getPsiModel();
			for(_SparseFeature fv: r.getSparse())
				L += fv.getTF()*psi[fv.getIndex()];
	
			return L;
		}
	}
	
	// The main MCMC algorithm, assign each review to clusters.
	protected void calculate_E_step(){
		System.out.print("[Info]E step: Sample z_ij...");

		_HDPThetaStar curThetaStar;
		_HDPAdaptStruct user;
		int index;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				if (r.getType() == rType.TEST)
					continue;//do not touch testing reviews!
				
				curThetaStar = r.getHDPThetaStar();
				
				//Step 1: remove the current review from the thetaStar and user side.
				user.updateHDPThetaStarMemSize(curThetaStar, -1);
				if(user.getHDPThetaMemSize(curThetaStar)==0)
					user.rmThetaFromMemSizeMap(curThetaStar);
				
				curThetaStar.updateMemCount(-1);
				curThetaStar.rmReview(r);

				if(curThetaStar.getMemSize() == 0) {// No data associated with the cluster.
					index = findHDPThetaStar(curThetaStar);
					swapTheta(m_kBar-1, index); // move it back to \theta*
					swapGamma(m_kBar-1, index); // swap gammas for later use.
					// swap \gamma_index and \gamma_k(weight for last cluster), add \gamma_e to \gamma_k.
					m_gamma[m_kBar-1] += m_gamma[m_kBar];//recycle the weight of gamma[index].
					m_gamma[m_kBar] = 0;
					m_kBar --;
				}
				
				//Step 2: sample new cluster assignment for this review
				sampleOneInstance(user, r);
			}
		}
	}
	
	//Sample how many local groups inside user reviews.
	protected int sampleH(_HDPAdaptStruct user, _HDPThetaStar s){
		int n = user.getHDPThetaMemSize(s);
		if(n==1)
			return 1;//s(1,1)=1
		
		double[] prob = new double[n];
		//Find corresponding gamma value.
		int index = s.getIndex();
		
		double etaGammak = Math.log(m_eta) + Math.log(m_gamma[index]);
		//the number of local groups lies in the range [1, n];
		for(int h=1; h<=n; h++){
			double stir = stirling(n, h);
			prob[h-1] = h*etaGammak + Math.log(stir);
		}
		
		//h starts from 0, we want the number of tables here.	
		int h = Utils.sampleInLogArray(prob, n)+1;
		return h;
	}
	
	// n is the total number of observation under group k for the user.
	// h is the number of tables in group k for the user.
	public double stirling(int n, int h){
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
		System.out.print("[Info]E step: Sample gamma...\n");
		double alpha = Gamma.staticNextDouble(m_alpha, 1);
		
		double sum = alpha;
		for(int k=0; k<m_kBar; k++){
			m_gamma[k] = Gamma.staticNextDouble(m_hdpThetaStars[k].m_hSize, 1);
			sum += m_gamma[k];
		}
		for(int k=0; k<m_kBar; k++) 
			m_gamma[k]/=sum;
		
		m_gamma[m_kBar] = alpha/sum;//\gamma_e.
//		System.out.print(String.format("%d global groups.\n", m_kBar));
	}
	
	protected void updateGamma(){
		for(int k=0; k<m_kBar; k++)
			m_hdpThetaStars[k].m_hSize = 0;
		
		int index;
		_HDPAdaptStruct user;
		System.out.print("Sample h...");
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			
			for(_HDPThetaStar s: user.getHDPThetaMemSizeMap().keySet()){
				index = s.getIndex();
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
	@Override
	protected double calculate_M_step(){
		assignClusterIndex();
		
		
		//Step 1: sample gamma based on the current assignment.
		updateGamma(); // why for loop BETA_K times?
		
		//Step 2: Optimize language model parameters with MLE.
		//Step 3: Optimize logistic regression parameters with lbfgs.
		return estPsi() + estPhi();
	}
	
	//We should use maximum a posterior to estimate language models.
	public double estPsi(){
		double sum = 0, lmProb[], logLikelihood = 0;
		_HDPThetaStar theta;
		for(int k=0; k<m_kBar; k++){ 
			theta = m_hdpThetaStars[k];
			lmProb = theta.getPsiModel();
			
//			System.arraycopy(m_globalLM, 0, lmProb, 0, m_lmDim);//set the prior properly
			
			Arrays.fill(lmProb, m_beta/m_lmDim);
			for(_Review r: theta.getReviews()){
				for(_SparseFeature fv: r.getSparse()){
					lmProb[fv.getIndex()] += fv.getTF();
				}
			}
			sum = Math.log(Utils.sumOfArray(lmProb));
			
			//Estimate the prob for language model.
			for(int i=0; i<m_lmDim; i++) 
				lmProb[i] = Math.log(lmProb[i]) - sum;
			
			for(_Review r: theta.getReviews()){
				for(_SparseFeature fv: r.getSparse()){
					logLikelihood += fv.getTF() * lmProb[fv.getIndex()];
				}
			}
		}
		
		return logLikelihood;
	}
	
	//why do not we scan through the reviews associated with each cluster?
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
	
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc r, double weight, double[] g) {
		_DPAdaptStruct user = (_DPAdaptStruct)u;
		_Review review = (_Review) r;
		int n; // feature index
		int cIndex = review.getHDPThetaStar().getIndex();
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error,cannot find the HDP theta star!");
		
		int offset = m_dim*cIndex;
		double delta = weight * (review.getYLabel() - logit(review.getSparse(), review));
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
	
	protected double logit(_SparseFeature[] fvs, _Review r){
		double sum = Utils.dotProduct(r.getHDPThetaStar().getModel(), fvs, 0);
		return Utils.logistic(sum);
	}
	
	protected double estPhi(){
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
					fValue += logLikelihood_MultiThread();//this could be implemented at a per-review basis
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
//			e.printStackTrace();
		}	
		return oldFValue;
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
	@Override
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
		double prob;
		double[] probs;
		if(m_newCluster) 
			probs = new double[m_kBar+1];
		else 
			probs = new double[m_kBar];
		
		_HDPAdaptStruct user;
		_HDPThetaStar curTheta;
		
		//sample a new cluster parameter first.
		if(m_newCluster)
			m_G0.sampling(m_hdpThetaStars[m_kBar].getModel());
			
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				if (r.getType() != rType.TEST)
					continue;				
				
				for(int k=0; k<probs.length; k++){
					curTheta = m_hdpThetaStars[k];
					r.setHDPThetaStar(curTheta);
					prob = calcLogLikelihoodX(r) + Math.log(user.getHDPThetaMemSize(curTheta)+m_eta*m_gamma[k]);//this proportion includes the user's current cluster assignment
					probs[k] = Math.exp(prob);//this will be in real space!					
				}
				Utils.L1Normalization(probs);
				r.setClusterPosterior(probs);
			}
		}
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
