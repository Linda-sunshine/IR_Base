package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.Arrays;
import java.util.HashMap;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.CoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt._CoLinAdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt._LinAdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

// In this class, we do R1 over the global weights, ||wi-wg||.
public class CoLinAdaptWithR1OverWeights extends CoLinAdapt{

	public CoLinAdaptWithR1OverWeights(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
	}

	@Override
	protected double calculateFuncValue(_AdaptStruct u){
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
		
		// log likelihood.
		double L = calcLogLikelihood(ui);
		double R1 = 0, R2;
		
		// R1 regularization.
		for(int i=0; i<ui.getPWeights().length; i++){
			R1 += m_eta1 * Utils.EuclideanDistance(ui.getPWeights(), m_gWeights);
		}
		
		// R2 regularization.
		R2 = super.calculateR2(u);
		
		return R1 + R2 - L;
	}
	
	// Calculate the gradients for the use in LBFGS.
	@Override
	protected void gradientByR1(_AdaptStruct u){
		_CoLinAdaptStruct user = (_CoLinAdaptStruct)u;
		double dA, dB;
		int k, offset = 2*m_dim*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		
		//R1 regularization part
		for(int n=0; n<m_featureSize+1; n++){
			k = m_featureGroupMap[n];
			dA = m_eta1 * (user.getPWeights()[n] - m_gWeights[n]) * m_gWeights[n];
			dB = m_eta1 * (user.getPWeights()[n] - m_gWeights[n]);
			
			m_g[offset + k] += dA;
			m_g[offset + k] += dB;
		}
	}
	
	//this is batch training in each individual user
	@Override
	public double train() {
		int[] iflag = { 0 }, iprint = { -1, 3 };
		double fValue, oldFValue = Double.MAX_VALUE;
		int vSize = super.getVSize(), displayCount = 0;
		_LinAdaptStruct user;

		initLBFGS();
		init();
		try {
			do {
				fValue = 0;
				Arrays.fill(m_g, 0); // initialize gradient
				setPersonalizedModel();
				// accumulate function values and gradients from each user
				for (int i = 0; i < m_userList.size(); i++) {
					user = (_LinAdaptStruct) m_userList.get(i);
					fValue += calculateFuncValue(user);
					calculateGradients(user);
				}
				if (m_displayLv == 2) {
					gradientTest();
					System.out.println("Fvalue is " + fValue);
				} else if (m_displayLv == 1) {
					if (fValue < oldFValue)
						System.out.print("o");
					else
						System.out.print("x");

					if (++displayCount % 100 == 0)
						System.out.println();
				}

				LBFGS.lbfgs(vSize, 5, _CoLinAdaptStruct.getSharedA(), fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);
				setPersonalizedModel();
			} while (iflag[0] != 0);
			System.out.println();
		} catch (ExceptionWithIflag e) {
			System.out.println("LBFGS fails!!!!");
			e.printStackTrace();
		}
		setPersonalizedModel();
		return oldFValue;
	}
}
