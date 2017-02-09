package Classifier.supervised.modelAdaptation.MMB;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.distribution.BinomialDistribution;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.HDP.CLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP._HDPAdaptStruct;
import cern.jet.random.tdouble.Beta;
import structures._HDPThetaStar;
import structures._MMBNeighbor;
import structures._User;
import utils.Utils; 

public class CLRWithMMB extends CLRWithHDP {
	double[] m_ab = new double[]{0.1, 0.1}; // parameters used in the gamma function in mmb model.
	double m_rho = 0.1;
	double m_rhoCount = 0; // count how many 1 edges we have.
	double[][] m_Bs; // the matrix of all probability.  
	BinomialDistribution m_bernoulli = new BinomialDistribution(1, m_rho);
	BetaDistribution m_Beta = new BetaDistribution(m_ab[0], m_ab[1]);
	
	HashMap<String, _HDPAdaptStruct> m_userMap; // key: userID, value: _AdaptStruct
	
	public CLRWithMMB(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel,
			double[] betas) {
		super(classNo, featureSize, featureMap, globalModel, betas);
	}
	
	
	@Override
	public String toString() {
		return String.format("CLRWithMMB[dim:%d,lmDim:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:%.3f,#Iter:%d,N(%.3f,%.3f)]", m_dim,m_lmDim,m_M, m_alpha, m_eta, m_beta, m_eta1, m_numberOfIterations, m_abNuA[0], m_abNuA[1]);
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		
		for(_User user:userList)
			m_userList.add(new _HDPAdaptStruct(user));
		m_pWeights = new double[m_gWeights.length];		
	}
	
	@Override
	public void initThetaStars(){
		// assign each review to one cluster.
		super.initThetaStars();
		_HDPAdaptStruct uj;
		
		m_userMap = new HashMap<String, _HDPAdaptStruct>();
		// construct the map.
		for(_AdaptStruct ui: m_userList)
			m_userMap.put(ui.getUserID(), (_HDPAdaptStruct) ui);
		
		// add the friends one by one.
		for(_AdaptStruct ui: m_userList){
			for(String nei: ui.getUser().getFriends()){
				if(m_userMap.containsKey(nei)){
					uj = m_userMap.get(nei);
					sampleOneEdge((_HDPAdaptStruct) ui, uj, 1);
					System.out.print("o");
				} else{
					System.out.print("x");
				}
			}
			System.out.println();
		}
	}

	@Override
	// The function is used in "sampleOneInstance".
	public double calcGroupPopularity(_HDPAdaptStruct user, int k, double gamma_k){
		return user.getHDPThetaMemSize(m_hdpThetaStars[k]) + m_eta*gamma_k + user.getHDPThetaEdgeSize(m_hdpThetaStars[k]);
	}
	
	// Sample one edge between (ui, uj)
	public void sampleOneEdge(_HDPAdaptStruct ui, _HDPAdaptStruct uj, int e){
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
		ui.addNeighbor(uj, m_hdpThetaStars[k], e);
		
		//Step 4: Update the user info with the newly sampled hdpThetaStar.
		ui.incHDPThetaStarEdgeSize(m_hdpThetaStars[k], 1);//-->3		

		if(k >= m_kBar) 
			sampleNewCluster(k);
		
//		if(k >= m_kBar){//sampled a new cluster
//
//			m_hdpThetaStars[k].initPsiModel(m_lmDim);
//			m_hdpThetaStars[k].initB(m_kBar+1);
//			m_D0.sampling(m_hdpThetaStars[k].getPsiModel(), m_betas, true);//we should sample from Dir(\beta)
//			m_D0.sampling(m_hdpThetaStars[k].getB(), m_ab, false);// use the true space.
//			double rnd = Beta.staticNextDouble(1, m_alpha);
//			m_hdpThetaStars[k].setGamma(rnd*m_gamma_e);
//			m_gamma_e = (1-rnd)*m_gamma_e;
//			
//			swapTheta(m_kBar, k);
//			m_kBar++;
//		}
	}
	
	public void sampleNewCluster(int k){
		m_hdpThetaStars[k].initPsiModel(m_lmDim);
		m_D0.sampling(m_hdpThetaStars[k].getPsiModel(), m_betas, true);//we should sample from Dir(\beta)
		m_hdpThetaStars[k].initB();
		sampleB(m_hdpThetaStars[k]);
		double rnd = Beta.staticNextDouble(1, m_alpha);
		m_hdpThetaStars[k].setGamma(rnd*m_gamma_e);
		m_gamma_e = (1-rnd)*m_gamma_e;
			
		swapTheta(m_kBar, k);
		m_kBar++;
	}
	
	// sample each element of the vector 
	public void sampleB(_HDPThetaStar theta){
		for(int k=0; k<m_kBar; k++){
			// add the prob between B_{existing theta, new theta} to the hashmap. 
			m_hdpThetaStars[k].addOneB(theta, m_Beta.sample());
			// add the prob between B_{new theta, existing theta} to the hashmap. 
			theta.addOneB(m_hdpThetaStars[k], m_Beta.sample());
		}
	}
	protected double calcLogLikelihoodE(_HDPAdaptStruct ui, _HDPAdaptStruct uj){
		int eij = ui.hasEdge(uj) ? 1 : 0;
		if(!ui.getThetaStar().hasB(uj.getThetaStar())){
			return Utils.lgamma(m_ab[0] + eij) + Utils.lgamma(1- eij + m_ab[1])
					- Math.log(m_ab[0] + m_ab[1] + 1) - Utils.lgamma(m_ab[0]) - Utils.lgamma(m_ab[1]);
		} else{
			// probability for Bernoulli distribution: p(e_ij|z_{i->j}, z_{j->i},B)
			double p = ui.getThetaStar().getOneB(uj.getThetaStar());
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
		int index, sampleSize = 0, eij = 0;
		for(int i=0; i<m_userList.size(); i++){
			ui = (_HDPAdaptStruct) m_userList.get(i);
			for(int j=0; j<m_userList.size() && i!=j; j++){
				uj = (_HDPAdaptStruct) m_userList.get(j);

				// There are three cases.
				// Case 1&2: eij = 1 or (eij = 0 && eij is from mmb)
				if(ui.hasEdge(uj)){
					eij = ui.getEdge(uj);
					// remove the edge from the theta.
					curThetaStar = ui.getThetaStar(uj);
					curThetaStar.updateEdgeCount(-1);
					
					// remove the neighbor from user.
					ui.rmNeighbor(uj);
					
					if(curThetaStar.getMemSize() == 0 && curThetaStar.getEdgeSize() == 0){// No data associated with the cluster.
						m_gamma_e += curThetaStar.getGamma();
						index = findHDPThetaStar(curThetaStar);
						swapTheta(m_kBar-1, index); // move it back to \theta*
						curThetaStar = null;// Clear the probability vector.
						m_kBar --;
					}
					// if eij == 1, sample z_{i->j}
					if(eij == 1){
						sampleOneEdge(ui, uj, 1);
						sampleSize++;
					}
					// else if eij == 0, we need another variable to decide whether sample it or not.
					else{
						if(m_bernoulli.sample() == 1){
							sampleOneEdge(ui, uj, 0);
							sampleSize++;
						}
					}
				// Case 3: eij = 0 && eij is from background model.
				// We don't record the edge, thus no need to remove it.
				} else{
					if(m_bernoulli.sample() == 1)
						sampleOneEdge(ui, uj, 0);
						sampleSize++;
					}
				}				
			if (sampleSize%2000==0) {
				System.out.print('.');
				if (sampleSize%100000==0)
					System.out.println();
			}
		}
	}
	@Override
	protected double calculate_M_step(){
		assignClusterIndex();
		
		// sample gamma + estPsi + estPhi
		double likelihood = super.calculate_M_step();
		return likelihood + estB() + estRho();
	}
	
	// Estimate the Bernoulli rates matrix using Newton-Raphson method.
	// We maintain two matrixes to calculate the B. 
	public double estB(){
		double phi_gh = 0;
		_HDPAdaptStruct ui;
		_HDPThetaStar thetai, thetaj;
		_MMBNeighbor mui, muj;
		double[][] B_n = new double[m_kBar][m_kBar];
		double[][] B_d = new double[m_kBar][m_kBar];
		HashMap<_HDPAdaptStruct, _MMBNeighbor> neighborsMap;
		// Iterate through all the user pairs.
		for(int i=0; i<m_userList.size(); i++){
			int g = 0, h = 0;
			ui = (_HDPAdaptStruct) m_userList.get(i);
			neighborsMap = ui.getNeighbors();
			thetai = ui.getThetaStar();
			for(_HDPAdaptStruct uj: neighborsMap.keySet()){
				muj = neighborsMap.get(uj);
				mui = uj.getOneNeighbor(ui);
				g = muj.getHDPThetaStar().getIndex();
				h = mui.getHDPThetaStar().getIndex();
				thetaj = neighborsMap.get(uj).getHDPThetaStar();
				phi_gh = thetai.getOneB(thetaj) * thetaj.getOneB(thetai);
				// B(g, h) = B_n/B_d
				// B_n = (\sum_{p,q}Y(p,q)phi_{p->q,g}phi_{q->p, h})
				// B_d = (1-\rho)(\sum_{p,q}phi_{p->q,g}phi_{q->p, h})
				if(muj.getEdge() == 1){
					B_n[g][h] += phi_gh;// numerator
					m_rhoCount++; // used in the estimation of \rho
				}
				B_d[g][h] += phi_gh; // denominator
			}
		}
		for(int g=0; g<m_kBar; g++){
			for(int h=0; h<m_kBar; h++){
				m_Bs[g][h] = B_n[g][h]/(1-m_rho)*B_d[g][h];
			}
		}
		assignB();
		return 0;// how to calculate the likelihood.
	}
	// Assign the newly estimated Bs to each group parameter.
	public void assignB(){
		int h = 0;
		HashMap<_HDPThetaStar, Double> B;
		for(int g=0; g<m_kBar; g++){
			B = m_hdpThetaStars[g].getB();
			for(_HDPThetaStar thetaj: B.keySet()){
				h = thetaj.getIndex();
				B.put(thetaj, m_Bs[g][h]);
			}
		}
	}
	
	// Estimate the sparsity parameter.
	// \rho = (1 - \sum_{p,q}Y(p,q)/N^2
	public double estRho(){
		m_rho = 1 - m_rhoCount/(m_userList.size()*m_userList.size());
		return 0;
	}
}
