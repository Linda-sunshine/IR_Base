package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;
import cern.jet.random.tfloat.FloatUniform;
import structures._HDPThetaStar;
import structures._Review;
import structures._thetaStar;
import structures._Review.rType;
import structures._SparseFeature;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess._DPAdaptStruct;
/***
 * This class implements the CLinAdapt with HDP added.
 * Currently, each review is assigned to one group and each user is a mixture of the components.
 * @author lin
 *
 */
public class MTCLinAdaptWithHDP extends MTCLinAdaptWithDP {
	//\alpha is the concentration parameter for the first layer.
	protected double m_eta = 6;//concentration parameter for second layer DP.  
	protected double m_beta = 6; //concentration parameter for \psi.
	
	public static _HDPThetaStar[] m_hdpThetaStars = new _HDPThetaStar[1000];//phi+psi

	protected DirichletPrior m_D0; //Dirichlet prior.
	protected double[] m_gamma;//global mixture proportion.
	// We don't have psi for the super model.
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
		DirichletPrior m_D0 = new DirichletPrior();//dirichlet distribution for psi and gamma.
		initPriorG0();//normal distribution for \phi.
	}
	
	public void initParams(){
		m_D0.sampling(m_gamma, m_alpha);
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
				m_hdpThetaStars[m] = new _HDPThetaStar(2*m_dim);
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
			//p(z=k|\gamma,\eta)
			gamma_k = (k < m_kBar) ? m_gamma[k]:m_gamma[m_gamma.length-1];
			likelihood += Math.log(user.getHDPThetaMemSize(m_hdpThetaStars[k])+m_eta*gamma_k);
			//loglikelihood of x, i.e., p(x|\psi)	
			likelihood += calcLogLikelihood_x(r);
			m_hdpThetaStars[k].setProportion(likelihood);//this is in log space!
			if(k==0) logSum = likelihood;
			else logSum = Utils.logSum(logSum, likelihood);
		}
		logSum += Math.log(FloatUniform.staticNextFloat());//we might need a better random number generator
		k = 0;
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
		
		//Step 3: update the setting after sampling z_ij.
		m_hdpThetaStars[k].updateMemCount(1);
		r.setHDPThetaStar(m_hdpThetaStars[k]);
		//If we get a new cluster, sample \psi for the new one.
		if(k >= m_kBar) m_D0.sampling(m_hdpThetaStars[k].getPsiModel(), m_beta);
		//Update the user info with the newly sampled hdpThetaStar.
		if(user.getHDPThetaMemSize(m_hdpThetaStars[k]) == 0)
			user.addHDPThetaStar(m_hdpThetaStars[k]);
		else
			user.incHDPThetaStarMemSize(m_hdpThetaStars[k]);
		
		if(k >= m_kBar){
			swapTheta(m_kBar, k);
			m_kBar++;
		}
	}
	@Override
	protected void swapTheta(int a, int b) {
		_HDPThetaStar cTheta = m_hdpThetaStars[a];
		m_hdpThetaStars[a] = m_hdpThetaStars[b];
		m_hdpThetaStars[b] = cTheta;// kBar starts from 0, the size decides how many are valid.
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
		double L = 0;
		if(r.getHDPThetaStar().getPsiModel() == null){
			// add the likelihood for the new group.
			return L;
		}
		double[] psi = r.getHDPThetaStar().getPsiModel();
		for(_SparseFeature fv: r.getSparse()){
			L += fv.getValue()*Math.log(psi[fv.getIndex()+1]);// do we need the coefficients? factorial?
		}
		//add the factorial part in front of likelihood.
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
			m_hdpThetaStars[k].m_hSize = 0;
		int index;
		_HDPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_thetaStar s: user.getHMap().keySet()){
				index = findThetaStar(s);
				m_hdpThetaStars[index].m_hSize += sampleH(user, m_hdpThetaStars[index]);
			}
		}
		sampleGamma();
	}
}
