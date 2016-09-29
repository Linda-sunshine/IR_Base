package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;

import cern.jet.random.tfloat.FloatUniform;
import structures._Psi;
import structures._Review;
import structures._Review.rType;
import structures._SparseFeature;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess._DPAdaptStruct;

public class MTCLinAdaptWithHDP extends MTCLinAdaptWithDP {
	protected double m_eta = 6;// concentration parameter for second layer DP.  
	protected _Psi[] m_psis = new _Psi[1000]; // language model parameters.
	protected DirichletPrior m_D0; // Dirichlet prior.
	protected double[] m_gamma;// global mixture proportion.
	
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
		
		//init the structures for multi-threading
		if (m_multiThread) {
			int numberOfCores = Runtime.getRuntime().availableProcessors();
			m_fValues = new double[numberOfCores];
			m_gradients = new double[numberOfCores][]; 
		}
	}
	
	public void initPriors(){
		initPriorD0(); // dirichlet distribution for psi and gamma.
		initPriorG0();// normal distribution for thetastar.
	}
	public void initPriorD0(){
		DirichletPrior m_D0 = new DirichletPrior();
	}
	public void initParams(){
		m_D0.sample(m_gamma, m_alpha);
	}
	@Override
	protected void initThetaStars(){
		m_pNewCluster = Math.log(m_alpha) - Math.log(m_M);//to avoid repeated computation
		_HDPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_HDPAdaptStruct) m_userList.get(i);
			for(_Review r: user.getReviews()){
				if (r.getType() != rType.ADAPTATION)
					continue; // only touch the adaptation data
				sampleOneInstance(r);
			}
		}		
	}

	// Assign cluster assignment to one review.
	public void sampleOneInstance(_Review r){
		double likelihood, logSum = 0;
		int k;
		
		//reset thetaStars for the auxiliary thetaStars.
		sampleThetaStars();
		for(k=0; k<m_kBar+m_M; k++){
			r.setThetaStar(m_thetaStars[k]);
			//for existing groups, we have the psi.
			if(k < m_kBar) r.setPsi(m_psis[k]);
			likelihood = calcLogLikelihood(r);
		}		
		if (k<m_kBar)
			likelihood += Math.log(m_thetaStars[k].getMemSize() + m_eta*m_gamma[k]);
		else
			likelihood += m_pNewCluster ;
					 
		m_thetaStars[k].setProportion(likelihood);//this is in log space!

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
		// log likelihood given by the multinomial distribution.
		L += calcMtnmlLikelihood(r);
		return L;
	}		

	public double calcMtnmlLikelihood(_Review r){
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
}
