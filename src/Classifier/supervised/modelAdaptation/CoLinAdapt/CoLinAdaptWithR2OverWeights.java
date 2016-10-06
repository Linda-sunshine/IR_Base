package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.Arrays;
import java.util.HashMap;

import structures._RankItem;
import utils.Utils;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

public class CoLinAdaptWithR2OverWeights extends CoLinAdapt {

	// The only difference between this method and CoLinAdapt is the R2 regularization is over weights.
	public CoLinAdaptWithR2OverWeights(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
	}

	@Override
	public String toString() {
		return String.format("CoLinAdaptWithR2OverWeighs[dim:%d,eta1:%.3f,eta2:%.3f,eta3:%.3f,k:%d,NB:%s]", m_dim, m_eta1, m_eta2, m_eta3, m_topK, m_sType);
	}
	public double calculateFuncValue(_AdaptStruct u){
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u, uj;
		
		double L = calcLogLikelihood(ui); //log likelihood.
		double R1 = 0, R2 = 0, diff = 0;
		
		//Add regularization parts.
		for(int i=0; i<m_dim; i++){
			R1 += m_eta1 * (ui.getScaling(i)-1) * (ui.getScaling(i)-1);//(a[i]-1)^2
			R1 += m_eta2 * ui.getShifting(i) * ui.getShifting(i);//b[i]^2
		}
		
		// R2 regularization over the weights of users.
		for(_RankItem nit: ui.getNeighbors()){
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			diff = Utils.EuclideanDistance(ui.getPWeights(), uj.getPWeights());
			R2 += nit.m_value * m_eta3 * diff;
		}
		return -L + R1 + R2;
	}
	
	// Since we have different definition of R2, we have different gradients.
	public void gradientByR2(_AdaptStruct user){
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)user, uj;
		int offseti = m_dim*2*ui.getId(), offsetj, k;
		double coef, dA, dB;
		for(_RankItem nit: ui.getNeighbors()){
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			offsetj = m_dim*2*uj.getId();
			coef = 2 * nit.m_value * m_eta3;
			
			for(int n=0; n<m_featureSize+1; n++){
				k = m_featureGroupMap[n];
				dA = coef * (ui.getPWeight(n) - uj.getPWeight(n))*m_gWeights[n]; // (w_{i,v} - w_{j,v})*w_{g,v}
				dB = coef * (ui.getPWeight(n) - uj.getPWeight(n)); //
				// update ui's gradient
				m_g[offseti + k] += dA;
				m_g[offseti + k + m_dim] += dB;
				
				// update uj's gradient.
				m_g[offsetj + k] -= dA;
				m_g[offsetj + k + m_dim] -= dB;
			}
		}
	}
	@Override
	protected void calculateGradients(_AdaptStruct u){
		gradientByFunc(u);
		gradientByR1(u);
		gradientByR2(u);
	}
	//this is batch training in each individual user
	@Override
	public double train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, oldFValue = Double.MAX_VALUE;;
		int vSize = getVSize(), displayCount = 0;
		_LinAdaptStruct user;
			
		initLBFGS();
		init();
		try{
			do{
				fValue = 0;
				Arrays.fill(m_g, 0); // initialize gradient				
				setPersonalizedModel();

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
				oldFValue = fValue;
					
				LBFGS.lbfgs(vSize, 5, _CoLinAdaptStruct.getSharedA(), fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);//In the training process, A is updated.
			} while(iflag[0] != 0);
			System.out.println();
		} catch(ExceptionWithIflag e) {
			e.printStackTrace();
		}		
			
		setPersonalizedModel();
		return oldFValue;
	}
}
