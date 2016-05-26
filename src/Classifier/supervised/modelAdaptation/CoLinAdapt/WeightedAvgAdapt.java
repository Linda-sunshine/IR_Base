package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;
import Classifier.supervised.modelAdaptation.ModelAdaptation;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

public class WeightedAvgAdapt extends CoLinAdapt {
	
	public WeightedAvgAdapt(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
	}

	@Override
	// In this logit function, we need to sum over all the neighbors of the current user.
	protected double logit(_SparseFeature[] fvs, _AdaptStruct user){
		
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct) user;
		// The user itself.
		double sum = 0;
		double subSum = ui.getPWeight(0); // bias term
		for(_SparseFeature f:fvs) 
			subSum += ui.getPWeight(f.getIndex()+1) * f.getValue();		
		sum += ui.getSelfSim() * subSum;
		
		// Traverse all neighbors of the current user.
		for(_RankItem nit: ui.getNeighbors()){
			_CoLinAdaptStruct uj = (_CoLinAdaptStruct) m_userList.get(nit.m_index);
			subSum = uj.getPWeight(0);
			for(_SparseFeature f: fvs) 
				subSum += uj.getPWeight(f.getIndex()+1) * f.getValue();		
			sum += nit.m_value * subSum;
		}
		return Utils.logistic(sum);
	}
	
	@Override
	protected double calculateFuncValue(_AdaptStruct u) {		
		_CoLinAdaptStruct user = (_CoLinAdaptStruct)u;
		// Likelihood of the user.
		double L = calcLogLikelihood(user); //log likelihood.
		// regularization between the personal weighs and global weights.
		double R1 = m_eta1 * Utils.EuclideanDistance(user.getPWeights(), m_gWeights);//(a[i]-1)^2
		return R1 - L;
	}
	
	//this is batch training in each individual user
	@Override
	public double train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, oldFValue = Double.MAX_VALUE;;
		int vSize = getVSize(), displayCount = 0;
		_LinAdaptStruct user;
		m_diffs = new ArrayList<Double>();
		initLBFGS();
		init();
		try{
			do{
				fValue = 0;
				Arrays.fill(m_g, 0); // initialize gradient				
				// accumulate function values and gradients from each user
				for(int i=0; i<m_userList.size(); i++) {
					user = (_LinAdaptStruct)m_userList.get(i);
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
//				m_diffs.add(calculateDifference());
			} while(iflag[0] != 0);
			System.out.println();
		} catch(ExceptionWithIflag e) {
			System.out.println("LBFGS fails!!!!");
			e.printStackTrace();
		}		
			
		setPersonalizedModel();
		return oldFValue;
	}	
	
}
