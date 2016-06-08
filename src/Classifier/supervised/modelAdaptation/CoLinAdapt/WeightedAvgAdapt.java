package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import structures._User;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

public class WeightedAvgAdapt extends WeightedAvgTransAdapt {
	
	public class Neighbor {
		double m_sim;
		double[] m_weights;
	
		public Neighbor(double s, double[] ws){
			m_sim = s;
			m_weights = ws;
		}
		
		public double getSimilarity(){
			return m_sim;
		}
		
		public double[] getWeights(){
			return m_weights;
		}
	}
	
	public WeightedAvgAdapt(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
		m_dim = m_featureSize + 1; // We use all features to do average.
	}

	@Override
	public String toString() {
		return String.format("WeightedAvgAdapt[dim:%d,eta1:%.3f,k:%d,NB:%s]", m_dim, m_eta1, m_topK, m_sType);
	}
	
	@Override
	protected int getVSize() {
		return m_dim*m_userList.size();
	}
	
	@Override
	void constructUserList(ArrayList<_User> userList) {
		int vSize = m_dim;
		
		//step 1: create space
		m_userList = new ArrayList<_AdaptStruct>();		
		for(int i=0; i<userList.size(); i++) {
			_User user = userList.get(i);
			m_userList.add(new _CoLinAdaptStruct(user, m_dim, i, m_topK));
			user.setModel(m_gWeights); // Init user weights with global weights.
		}
		m_pWeights = new double[m_gWeights.length];			
		
		//huge space consumption
		_CoLinAdaptStruct.sharedA = new double[getVSize()];
		
		//step 2: copy each user's weights to shared A(weights) in _CoLinAdaptStruct		
		for(int i=0; i<m_userList.size(); i++)
			System.arraycopy(m_gWeights, 0, _CoLinAdaptStruct.sharedA, vSize*i, vSize);
	}
	@Override
//	// In this logit function, we need to sum over all the neighbors of the current user.
//	protected double logit(_SparseFeature[] fvs, _AdaptStruct user){
//		
//		_CoLinAdaptStruct ui = (_CoLinAdaptStruct) user;
//		// The user itself.
//		double sum = 0;
//		double subSum = ui.getPWeight(0); // bias term
//		for(_SparseFeature f:fvs) 
//			subSum += ui.getPWeight(f.getIndex()+1) * f.getValue();		
//		sum += ui.getSelfSim() * subSum;
//		
//		// Traverse all neighbors of the current user.
//		for(_RankItem nit: ui.getNeighbors()){
//			_CoLinAdaptStruct uj = (_CoLinAdaptStruct) m_userList.get(nit.m_index);
//			subSum = uj.getPWeight(0);
//			for(_SparseFeature f: fvs) 
//				subSum += uj.getPWeight(f.getIndex()+1) * f.getValue();		
//			sum += nit.m_value * subSum;
//		}
//		return Utils.logistic(sum);
//	}
	
	// In this logit function, we need to sum over all the neighbors of the current user.
	protected double logit(_SparseFeature[] fvs, _AdaptStruct user){
		
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct) user;
		// The user itself.
		double sum = 0;
		double subSum = ui.getPWeight(0); // bias term
		for(_SparseFeature f:fvs) 
			subSum += ui.getPWeight(f.getIndex()+1) * f.getValue();		
		sum += 1.0/m_topK * subSum;
		
		// Traverse all neighbors of the current user.
		for(_RankItem nit: ui.getNeighbors()){
			_CoLinAdaptStruct uj = (_CoLinAdaptStruct) m_userList.get(nit.m_index);
			subSum = uj.getPWeight(0);
			for(_SparseFeature f: fvs) 
				subSum += uj.getPWeight(f.getIndex()+1) * f.getValue();		
			sum += 1.0/m_topK * subSum;
		}
		return Utils.logistic(sum);
	}
	
	
	@Override
	protected double calculateFuncValue(_AdaptStruct u) {		
		_CoLinAdaptStruct user = (_CoLinAdaptStruct)u;
		// Likelihood of the user.
		double L = calcLogLikelihood(user); //log likelihood.
		// regularization between the personal weighs and global weights.
		double R1 = m_eta1 * Utils.EuclideanDistance(user.getPWeights(), m_gWeights);// 0.5*(a[i]-1)^2
		return R1 - L;
	}
	
	//shared gradient calculation by batch and online updating
	@Override
//	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
//		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u, uj;
//		
//		int n, offsetj;
//		int offset = m_dim*ui.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
//		double delta = (review.getYLabel() - logit(review.getSparse(), ui));
//		if(m_LNormFlag)
//			delta /= getAdaptationSize(ui);
//		
//		// Current user's info: Bias term + other features.
//		m_g[offset] -= weight*delta*ui.getSelfSim(); // \theta_{ii}*x_0 and x_0=1
//		for(_SparseFeature fv: review.getSparse()){
//			n = fv.getIndex() + 1;
//			m_g[offset + n] -= weight * delta * ui.getSelfSim() * fv.getValue();//\theta_{ii}*x_d
//		}
//		
//		// Neighbors' info.
//		for(_RankItem nit: ui.getNeighbors()) {
//			offsetj = m_dim*nit.m_index;
//			m_g[offsetj] -= weight * delta*nit.m_value; // neighbors' bias term.
//			for(_SparseFeature fv: review.getSparse()){
//				n = fv.getIndex() + 1;
//				m_g[offsetj + n] -= weight * delta * nit.m_value * fv.getValue(); // neighbors' other features.
//			}
//		}
//	}
	
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u, uj;
		
		int n, offsetj;
		int offset = m_dim*ui.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		double delta = (review.getYLabel() - logit(review.getSparse(), ui));
		if(m_LNormFlag)
			delta /= getAdaptationSize(ui);
		
		// Current user's info: Bias term + other features.
		m_g[offset] -= weight*delta*1.0/m_topK; // \theta_{ii}*x_0 and x_0=1
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			m_g[offset + n] -= weight * delta * 1.0/m_topK * fv.getValue();//\theta_{ii}*x_d
		}
		
		// Neighbors' info.
		for(_RankItem nit: ui.getNeighbors()) {
			offsetj = m_dim*nit.m_index;
			m_g[offsetj] -= weight * delta*1.0/m_topK; // neighbors' bias term.
			for(_SparseFeature fv: review.getSparse()){
				n = fv.getIndex() + 1;
				m_g[offsetj + n] -= weight * delta * 1.0/m_topK * fv.getValue(); // neighbors' other features.
			}
		}
	}
	//Calculate the gradients for the use in LBFGS.
	@Override
	protected void gradientByR1(_AdaptStruct u){
		_CoLinAdaptStruct user = (_CoLinAdaptStruct)u;
		int offset = m_dim*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		//R1 regularization part
		for(int k=0; k<m_dim; k++)
			m_g[offset + k] += 2 * m_eta1 * (user.getPWeight(k)-m_gWeights[k]);// (w_i-w_g)
	}
	
	//this is batch training in each individual user
	@Override
	public double train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, oldFValue = Double.MAX_VALUE;;
		int vSize = getVSize(), displayCount = 0;
		_CoLinAdaptStruct user;
		initLBFGS();
		init();
		try{
			do{
				fValue = 0;
				Arrays.fill(m_g, 0); // initialize gradient				
				// accumulate function values and gradients from each user
				for(int i=0; i<m_userList.size(); i++) {
					user = (_CoLinAdaptStruct)m_userList.get(i);
					fValue += calculateFuncValue(user);
					calculateGradients(user);
				}
				if (m_displayLv==2) {
					gradientTest();
					System.out.println("Fvalue is " + fValue);
				} else if (m_displayLv==1) {
					if (fValue<oldFValue)
						System.out.print("o");
					else
						System.out.print("x");
						
					if (++displayCount%100==0)
						System.out.println();
				} 
					
				LBFGS.lbfgs(vSize, 5, _CoLinAdaptStruct.getSharedA(), fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);//In the training process, A is updated.
				setPersonalizedModel();
			} while(iflag[0] != 0);
			System.out.println();
		} catch(ExceptionWithIflag e) {
			System.out.println("LBFGS fails!!!!");
			e.printStackTrace();
		}		
			
		setPersonalizedModel();
		setNeighbors();
		return oldFValue;
	}	
	
	// Collect all users' weight for lbfgs optimization.
	public double[] getAllUserWeights(){
		double[] weights = new double[m_userList.size()*m_dim];
		for(int i=0; i<m_userList.size(); i++){
			_CoLinAdaptStruct u = (_CoLinAdaptStruct) m_userList.get(i);
			System.arraycopy(u.getPWeights(), 0, weights, i*m_dim, m_dim);
		}
		return weights;
	}
	
	public void setPersonalizedModel(){
		_CoLinAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_CoLinAdaptStruct)m_userList.get(i);
			System.arraycopy(_CoLinAdaptStruct.sharedA, i*m_dim, m_pWeights, 0, m_dim);
			user.setPersonalizedModel(m_pWeights);
		}
	}
	
	public void setNeighbors(){
		_CoLinAdaptStruct ui, uj;
		Neighbor[] neighbors = new Neighbor[m_topK];
		Neighbor nj;
		int count = 0;
		for(int i=0; i<m_userList.size(); i++){
			count = 0;
			ui = (_CoLinAdaptStruct)m_userList.get(i);
			for(_RankItem nit: ui.getNeighbors()){
				uj = (_CoLinAdaptStruct) m_userList.get(nit.m_index);
				nj = new Neighbor(nit.m_value, uj.getPWeights());
				neighbors[count++] = nj;
			}
			ui.getUser().setNeighbors(neighbors);
		}
	}
	
	@Override
	protected double gradientTest() {
		int offset, uid;
		double mag = 0;
		for(int n=0; n<m_userList.size(); n++) {
			uid = n*m_dim;
			for(int i=0; i<m_dim; i++){
				offset = uid + i;
				mag += m_g[offset]*m_g[offset];
			}
		}
		if (m_displayLv==2)
			System.out.format("Gradient magnitude: %.5f\n", mag);
		return mag;
	}
}
