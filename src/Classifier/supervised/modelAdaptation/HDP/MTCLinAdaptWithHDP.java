package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;

import cern.jet.random.tfloat.FloatUniform;
import structures._Psi;
import structures._Review;
import structures._thetaStar;
import structures._Review.rType;
import structures._SparseFeature;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess._DPAdaptStruct;

public class MTCLinAdaptWithHDP extends MTCLinAdaptWithDP {
	protected double m_eta = 6;//concentration parameter for second layer DP.  
	protected double m_beta = 6; //concentration parameter
	protected _Psi[] m_psis = new _Psi[1000]; //language model parameters.
	protected DirichletPrior m_D0; //Dirichlet prior.
	protected double[] m_gamma;//global mixture proportion.
	protected double[] m_supPsiModel; //psi for the super model.
	
	public MTCLinAdaptWithHDP(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup);
	}
	
	@Override
	protected void init(){
		super.init();
		initPriors();
		initParams();
		
		//Initialize the structures for multi-threading.
		if (m_multiThread) {
			int numberOfCores = Runtime.getRuntime().availableProcessors();
			m_fValues = new double[numberOfCores];
			m_gradients = new double[numberOfCores][]; 
		}
	}
	
	public void initPriors(){
		initPriorD0(); //dirichlet distribution for psi and gamma.
		initPriorG0();//normal distribution for thetastar.
	}
	//Init dirichlet priors.
	public void initPriorD0(){
		DirichletPrior m_D0 = new DirichletPrior();
		m_D0.sample(m_supPsiModel, m_beta);
	}
	public void initParams(){
		m_D0.sample(m_gamma, m_alpha);
	}
	@Override
	//Initialize the phi for each review.
	protected void initThetaStars(){
		m_pNewCluster = Math.log(m_alpha) - Math.log(m_M);//to avoid repeated computation
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
	//Assign cluster to each review.
	public void sampleOneInstance(_HDPAdaptStruct user, _Review r){
		double likelihood, logSum = 0, gamma_k;
		int k;
		
		//Step 1: reset thetaStars for the auxiliary thetaStars.
		sampleThetaStars();
		
		//Step 2: sample thetaStar based on the loglikelihood of p(z=k|\gamma,\eta)p(y|x,\phi)p(x|\psi)
		for(k=0; k<m_kBar+m_M; k++){
			r.setThetaStar(m_thetaStars[k]);
			if(user.getThetaStarMemSize(m_thetaStars[k]) == 0)
				user.addThetaStar(m_thetaStars[k]);
			else
				user.incThetaStarMemSize(m_thetaStars[k]);
			
			//loglikelihood of y, i.e., p(y|x,\phi)
			likelihood = calcLogLikelihood(r); 
			
			//p(z=k|\gamma,\eta)
			gamma_k = (k < m_kBar) ? m_gamma[k]:m_gamma[m_gamma.length-1];
			likelihood += 2*Math.log(user.getThetaStarMemSize(m_thetaStars[k])+m_eta*m_gamma[k]);
			//???What is the prior for \phi?????////
			
			//loglikelihood of x, i.e., p(x|\psi)
			if(k<m_kBar) r.setPsi(m_psis[k]);
			else likelihood += m_pNewCluster;	
			likelihood += calcLogLikelihood_x(r);
			
			m_thetaStars[k].setProportion(likelihood);//this is in log space!
			
			if(k==0) logSum = likelihood;
			else logSum = Utils.logSum(logSum, likelihood);
		}

		logSum += Math.log(FloatUniform.staticNextFloat());//we might need a better random number generator
		
		k = 0;
		double newLogSum = m_thetaStars[0].getProportion();
		do {
			if (newLogSum>=logSum)
				break;
			k++;
			newLogSum = Utils.logSum(newLogSum, m_thetaStars[k].getProportion());
		} while (k<m_kBar+m_M);
	
		if (k==m_kBar+m_M) {
			System.err.println("[Warning]Hit the very last element in theatStar!");
			k--; // we might hit the very last
		}
		
		//Step 3: update the setting after sampling z_ij.
		m_thetaStars[k].updateMemCount(1);
		r.setThetaStar(m_thetaStars[k]);
		
		if(user.getThetaStarMemSize(m_thetaStars[k]) == 0)
			user.addThetaStar(m_thetaStars[k]);
		else
			user.incThetaStarMemSize(m_thetaStars[k]);
		
		if(k >= m_kBar){
			swapTheta(m_kBar, k);
			m_kBar++;
		}
	}
	//Calculate the function value of the new added instance.
	protected double calcLogLikelihood(_Review r){
		double L = 0, Pi = 0, sum; //log likelihood.
		// log likelihood given by the logistic function.
		sum = Utils.dotProduct(r.getThetaStar().getModel(), r.getSparse(), 0);
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
		double L = 0;
		if(r.getPsi() == null){
			// add the likelihood for the new group.
			return L;
		}
		double[] psi = r.getPsi().getModel();
		for(_SparseFeature fv: r.getSparse()){
			L += fv.getValue()*Math.log(psi[fv.getIndex()+1]);// do we need the coefficients? factorial?
		}
		return L;
	}
	
	// The main MCMC algorithm, assign each review to clusters.
	protected void calculate_E_step(){
		_thetaStar curThetaStar;
		_HDPAdaptStruct user;
		
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				curThetaStar = user.getThetaStar();
				
				//Step 1: remove the current review from the thetaStar.
				curThetaStar.updateMemCount(-1);
				
				if(curThetaStar.getMemSize() == 0) {// No data associated with the cluster.
					swapTheta(m_kBar-1, findThetaStar(curThetaStar)); // move it back to \theta*
					m_kBar --;
				}
				sampleOneInstance(user, r);
			}
		}
		//Update gamma based on the current assignment.
		updateGamma(); // why for loop BETA_K times?
	}
	
	//Sample how many local groups inside user reviews.
	protected int sampleH(_HDPAdaptStruct user, _thetaStar s){
		return 0;
		//need to be added.
	}
	
	//Sample the global mixture proportion.
	protected void sampleGamma(){
		//need to be added.
	}
	
	protected void updateGamma(){
		for(int k=0; k<m_kBar; k++)
			m_thetaStars[k].m_hSize = 0;
		int index;
		_HDPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_thetaStar s: user.getHMap().keySet()){
				index = findThetaStar(s);
				m_thetaStars[index].m_hSize += sampleH(user, m_thetaStars[index]);
			}
		}
		sampleGamma();
	}
}
