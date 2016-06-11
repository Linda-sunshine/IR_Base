package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.ArrayList;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;

import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

public class WeightedAvgTransAdapt extends CoLinAdapt {

	public WeightedAvgTransAdapt(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList){		
		super.loadUsers(userList);
		_CoLinAdaptStruct ui;
		
		double sum; // the user's own similarity.
		// Normalize the similarity of neighbors.
		for(int i=0; i<userList.size(); i++){
			ui = (_CoLinAdaptStruct) m_userList.get(i);
			sum = 1;
			// Collect the sum of similarity.
			for(_RankItem nit: ui.getNeighbors())
				sum += nit.m_value;
			
			// Update the user's similarity.
			ui.setSelfSim(1/sum);
			ui.getUser().setSimilarity(1/sum);
			for(_RankItem nit: ui.getNeighbors())
				nit.m_value /= sum;
		}
	}
	
	@Override
	// In this logit function, we need to sum over all the neighbors of the current user.
	protected double logit(_SparseFeature[] fvs, _AdaptStruct user){
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct) user;
		
		int n, k; // group index		
		double sum = 0, subSum = ui.getScaling(0) * m_gWeights[0] + ui.getShifting(0); // bias term
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			subSum += (ui.getScaling(k)*m_gWeights[n] + ui.getShifting(k)) * fv.getValue();
		}
		sum += ui.getSelfSim() * subSum;
		
		// Traverse all neighbors of the current user.
		for(_RankItem nit: ui.getNeighbors()){
			_CoLinAdaptStruct uj = (_CoLinAdaptStruct) m_userList.get(nit.m_index);
			subSum = uj.getScaling(0) * m_gWeights[0] + uj.getShifting(0);// bias term for the neighbor.
			for(_SparseFeature fv: fvs){
				n = fv.getIndex() + 1;
				k = m_featureGroupMap[n];
				subSum += (uj.getScaling(k)*m_gWeights[n] + uj.getShifting(k)) * fv.getValue();		
			}
			sum += nit.m_value * subSum;
		}
		return Utils.logistic(sum);
	}
	
	@Override
	protected double calculateFuncValue(_AdaptStruct u){
		_LinAdaptStruct user = (_LinAdaptStruct)u;
		
		double L = calcLogLikelihood(user); //log likelihood.
		double R1 = 0;
		
		//Add regularization parts.
		for(int i=0; i<m_dim; i++){
			R1 += m_eta1 * (user.getScaling(i)-1) * (user.getScaling(i)-1);//(a[i]-1)^2
			R1 += m_eta2 * user.getShifting(i) * user.getShifting(i);//b[i]^2
		}
		return R1 - L;
	}
	
	@Override
	// We want to user RegLR's calculateGradients while we cannot inherit from far than parent class.
	protected void calculateGradients(_AdaptStruct u){
		gradientByFunc(u);
		gradientByR1(u); // inherit from LinAdapt
	}
	
	//shared gradient calculation by batch and online updating
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
			
		int n, k, offsetj;
		int offset = m_dim*ui.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		double delta = (review.getYLabel() - logit(review.getSparse(), ui));
		if(m_LNormFlag)
			delta /= getAdaptationSize(ui);
			
		// Current user's info: Bias term + other features.
		m_g[offset] -= weight*delta*ui.getSelfSim()*m_gWeights[0]; // \theta_{ii}*w_g[0]*x_0 and x_0=1
		m_g[offset + m_dim] -= weight*delta*ui.getSelfSim(); // \theta_{ii}*x_0
		
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			m_g[offset + k] -= weight * delta * ui.getSelfSim() * m_gWeights[n] * fv.getValue();//\theta_{ii}*x_d
			m_g[offset + k + m_dim] -= weight * delta * ui.getSelfSim() * fv.getValue();
		}
			
		// Neighbors' info: Bias term + other features.
		for(_RankItem nit: ui.getNeighbors()) {
			offsetj = 2*m_dim*nit.m_index;
			// Bias term.
			m_g[offsetj] -= weight * delta * nit.m_value * m_gWeights[0]; // neighbors' bias term.
			m_g[offsetj + m_dim] -= weight * delta * nit.m_value;
			
			for(_SparseFeature fv: review.getSparse()){
				n = fv.getIndex() + 1;
				k = m_featureGroupMap[n];
				m_g[offsetj + k] -= weight * delta * nit.m_value * m_gWeights[n] * fv.getValue(); // neighbors' other features.
				m_g[offsetj + m_dim + k] -= weight * delta * nit.m_value * fv.getValue();
			}
		}
	}

}
