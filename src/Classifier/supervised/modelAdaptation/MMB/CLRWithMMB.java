package Classifier.supervised.modelAdaptation.MMB;

import java.util.HashMap;

import org.apache.commons.math3.distribution.BinomialDistribution;

import Classifier.supervised.modelAdaptation.HDP.CLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP._HDPAdaptStruct;
import cern.jet.random.tdouble.Beta;
import structures._HDPThetaStar;
import utils.Utils; 

public class CLRWithMMB extends CLRWithHDP {
	double[] m_ab = new double[]{0.1, 0.1}; // parameters used in the gamma function in mmb model.
	double m_rho = 0.1;
	BinomialDistribution m_bernoulli = new BinomialDistribution(1, m_rho);
	
	public CLRWithMMB(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel,
			double[] betas) {
		super(classNo, featureSize, featureMap, globalModel, betas);
	}
	
	@Override
	// The function is used in "sampleOneInstance".
	public double calcGroupPopularity(_HDPAdaptStruct user, int k, double gamma_k){
		return user.getHDPThetaMemSize(m_hdpThetaStars[k]) + m_eta*gamma_k + user.getHDPThetaEdgeSize(m_hdpThetaStars[k]);
	}
	
	// Sample one edge between (ui, uj)
	public void sampleOneEdge(_HDPAdaptStruct ui, _HDPAdaptStruct uj){
		int k;
		double likelihood, gamma_k, logSum = 0;
		for(k=0; k<m_kBar+m_M; k++){
			
			ui.setThetaStar(m_hdpThetaStars[k]);
			
			//log likelihood of the edge p(e_{ij}, z, B)
			// p(eij|z_{i->j}, z_{j->i}, B)*p(z_{i->j}|\pi_i)*p(z_{j->i|\pi_j})
			likelihood = calcLogLikelihoodE(ui, uj);
						
			//p(z=k|\gamma,\eta)
			gamma_k = m_hdpThetaStars[k].getGamma();
			likelihood += Math.log(calcGroupPopularity(ui, k, gamma_k));
			
			m_hdpThetaStars[k].setProportion(likelihood);//this is in log space!
						
			if(k==0) 
				logSum = likelihood;
			else 
				logSum = Utils.logSum(logSum, likelihood);
		}
		//Sample group k with likelihood.
		k = sampleInLogSpace(logSum);
		
		//Step 3: update the setting after sampling z_ij.
		m_hdpThetaStars[k].updateEdgeCount(1);//-->1
		ui.updateNeighbors(uj, m_hdpThetaStars[k]);
		
		//Step 4: Update the user info with the newly sampled hdpThetaStar.
		ui.incHDPThetaStarEdgeSize(m_hdpThetaStars[k], 1);//-->3		

		if(k >= m_kBar){//sampled a new cluster
			
			m_hdpThetaStars[k].initPsiModel(m_lmDim);
			m_hdpThetaStars[k].initB();
			m_D0.sampling(m_hdpThetaStars[k].getPsiModel(), m_betas, true);//we should sample from Dir(\beta)
			
			double rnd = Beta.staticNextDouble(1, m_alpha);
			m_hdpThetaStars[k].setGamma(rnd*m_gamma_e);
			m_gamma_e = (1-rnd)*m_gamma_e;
			
			swapTheta(m_kBar, k);
			m_kBar++;
		}
	}
	protected double calcLogLikelihoodE(_HDPAdaptStruct ui, _HDPAdaptStruct uj){
		int eij = ui.hasEdge(uj) ? 1 : 0;
		double[] B = ui.getThetaStar().getB();
		if(B == null){
			return Utils.lgamma(m_ab[0] + eij) + Utils.lgamma(1- eij + m_ab[1])
					- Math.log(m_ab[0] + m_ab[1] + 1) - Utils.lgamma(m_ab[0]) - Utils.lgamma(m_ab[1]);
		}
		else{
			// probability for Bernoulli distribution: p(e_ij|z_{i->j}, z_{j->i},B)
			double p = ui.getThetaStar().getB()[uj.getThetaStar().getIndex()];
			double loglikelihood = 0;
			//int eij = 0; // get eij from user perspective.
			loglikelihood = eij == 0 ? (1 - p) : p;
			loglikelihood = Math.log(loglikelihood);
			return loglikelihood;
		}
	}
	
	protected void calculate_E_step(){
		// sample z_{i,d}
		super.calculate_E_step();
		
		// sample z_{i->j}
		_HDPThetaStar curThetaStar;
		_HDPAdaptStruct ui, uj;
		int index, sampleSize=0;
		for(int i=0; i<m_userList.size(); i++){
			ui = (_HDPAdaptStruct) m_userList.get(i);
			for(int j=0; j<m_userList.size() && i!=j; j++){
				uj = (_HDPAdaptStruct) m_userList.get(j);

				// There are three cases.
				// case 1: eij = 1
				if(ui.hasEdge(uj) && ui.getEdge(uj) == 1){
					//Step 1: remove the edge from the thetaStar and user side.
					curThetaStar = ui.getThetaStar(uj);
					ui.incHDPThetaStarEdgeSize(curThetaStar, -1);	
					ui.rmNeighbor(uj); // remove the neighbor
					curThetaStar.updateEdgeCount(-1);
					
					if(curThetaStar.getMemSize() == 0 && curThetaStar.getEdgeSize() == 0){// No data associated with the cluster.
						curThetaStar.resetB();// Clear the probability vector.
						curThetaStar.resetPsiModel();// Clear the language model parameter.
						m_gamma_e += curThetaStar.getGamma();
						index = findHDPThetaStar(curThetaStar);
						swapTheta(m_kBar-1, index); // move it back to \theta*
						m_kBar --;
					}
				// case 2: eij = 0 (the connection between ui and uj is from MMB while it is 0)
				}else if(ui.hasEdge(uj) && ui.getEdge(uj) == 0){
					//Step 2: sample new cluster assignment for this pair (ui, uj).
					sampleOneEdge(ui, uj);
				
					if (++sampleSize%2000==0) {
						System.out.print('.');
						sampleGamma();//will this help sampling?
						if (sampleSize%100000==0)
							System.out.println();
					}
				// case 3: there is no connection between ui and uj.
				} else{
					
				}
			}
		}
	}
	@Override
	protected double calculate_M_step(){
		// sample gamma + estPsi + estPhi
		double likelihood = super.calculate_M_step();
		return likelihood + estB() + estRho();
	}
	
	// Estimate the Bernoulli rates matrix.
	public double estB(){
		return 0;
	}
	
	// Estimate the sparsity parameter.
	public double estRho(){
		return 0;
	}
	
}
