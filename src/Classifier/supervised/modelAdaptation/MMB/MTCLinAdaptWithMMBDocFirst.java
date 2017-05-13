package Classifier.supervised.modelAdaptation.MMB;

import java.util.HashMap;

/**
 * In this class, we want to sample documents first, after documents get stable clusters, 
 * we do cluster assignment for documents and edges then.
 * @author lin
 */
public class MTCLinAdaptWithMMBDocFirst extends MTCLinAdaptWithMMB {

	public MTCLinAdaptWithMMBDocFirst(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] betas) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup, betas);
	}

	@Override
	public String toString() {
		return String.format("MTCLinAdaptWithMMBWithDocFirst[dim:%d,dimSup:%d,lmDim:%d,M:%d,rho:%.5f,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:(%.3f,%.3f),#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]",m_dim,m_dimSup,m_lmDim,m_M,m_rho,m_alpha,m_eta,m_beta,m_eta1,m_eta2,m_numberOfIterations,m_abNuA[0],m_abNuA[1],m_abNuB[0],m_abNuB[1]);
	}
	
	/***
	 * In the training process, we sample documents first.
	 * Then we sample documents and edges together.*/
	@Override
	public double train(){
		System.out.println(toString());
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		int count = 0;
		
		// init: assign each review to one cluster.
		init(); // clear user performance and init cluster assignment	
		
		// Burn in period for doc (no edge sampling involved)
		while(count++ < m_burnIn){
			super.calculate_E_step();
			lastLikelihood = calculate_M_step();
		}
		
		// init: assign each edge to one cluster.
		initThetaStars4Edges();
		
		count = 0;
		while(count++ < m_burnIn){
			calculate_E_step();

			// Optimize the parameters
			curLikelihood = calculate_M_step();
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
			
			printInfo(i%5==0);//no need to print out the details very often
			System.out.print(String.format("\n[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.3f\n", i, curLikelihood, delta));
			if(Math.abs(delta) < m_converge)
				break;
			lastLikelihood = curLikelihood;
		}

		evaluateModel(); // we do not want to miss the last sample?!
//		setPersonalizedModel();
		return curLikelihood;
	}
}
