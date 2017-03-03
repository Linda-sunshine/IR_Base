package Classifier.supervised.modelAdaptation.MMB;
import java.util.HashMap;
import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.distribution.BinomialDistribution;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.HDP.CLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP._HDPAdaptStruct;
import cern.jet.random.Binomial;
import cern.jet.random.tdouble.Beta;
import cern.jet.random.tfloat.FloatUniform;
import structures._HDPThetaStar;
import structures._MMBNeighbor;
import structures._SparseFeature;
import utils.Utils; 

public class CLRWithMMB extends CLRWithHDP {

	double m_rho = 0.1; // sparsity parameter
	double[] m_pNew = new double[2]; // prob for the new cluster in sampling mmb edges.
	// parameters used in the gamma function in mmb model, prior of B~beta(a, b), prior of \rho~Beta(c, d)
	double[] m_abcd = new double[]{0.1, 0.1, 0.1, 0.1}; 
	// The prob
	double m_pNewJoint = 2*(Math.log(m_rho) + Math.log(m_abcd[1]+1) - Math.log(m_abcd[0]+m_abcd[1]+1));

	// Me: total number of edges eij=1;Ne: total number of edges eij=0 from mmb; Le: total number of edges eij=0 from background model.
	double m_Me = 0, m_Ne = 0, m_Le = 0;
	
	double[][] m_Bs;
	double[][] m_cache; // two-dim array for storing probs used in sampling zero edge.
	BetaDistribution m_Beta = new BetaDistribution(m_abcd[0], m_abcd[1]);
	BinomialDistribution m_bernoulli;
	HashMap<String, _HDPAdaptStruct> m_userMap; // key: userID, value: _AdaptStruct
	
	// Because we have to store all the indicators for all the edges(even edges from background model).
	// Thus, we can maintain a matrix for indexing.
	_HDPThetaStar[][] m_indicator = new _HDPThetaStar[m_userList.size()][m_userList.size()];
		
	public CLRWithMMB(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel,
			double[] betas) {
		super(classNo, featureSize, featureMap, globalModel, betas);
	} 
	
	public void calcProbNew(){
		for(int e=0; e<2; e++){
			m_pNew[e] = Utils.lgamma(m_abcd[0] + e) + Utils.lgamma(m_abcd[1])
					- Math.log(m_abcd[0] + m_abcd[1] + e) - Utils.lgamma(m_abcd[0]) - Utils.lgamma(m_abcd[1]);
		}
	}
	
	@Override
	public String toString() {
		return String.format("CLRWithMMB[dim:%d,lmDim:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:%.3f,#Iter:%d,N(%.3f,%.3f)]", m_dim,m_lmDim,m_M, m_alpha, m_eta, m_beta, m_eta1, m_numberOfIterations, m_abNuA[0], m_abNuA[1]);
	}
	
	public void check(){
		m_userMap = new HashMap<String, _HDPAdaptStruct>();
		// construct the map.
		for(_AdaptStruct ui: m_userList)
			m_userMap.put(ui.getUserID(), (_HDPAdaptStruct) ui);
		
		// add the friends one by one.
		for(_AdaptStruct ui: m_userList){
			for(String nei: ui.getUser().getFriends()){
				if(m_userMap.containsKey(nei)){
					String[] frds = m_userMap.get(nei).getUser().getFriends();
					if(hasFriend(frds, ui.getUserID()))
						System.out.print("");
					else
						System.out.print("x");
				}
			}
		}
	}
	
	public boolean hasFriend(String[] arr, String str){
		for(String a: arr){
			if(str.equals(a))
				return true;
		}
		return false;
	}
	
	public void sanityCheck(){
		_HDPAdaptStruct uj;
		for(_AdaptStruct ui: m_userList){
			for(String nei: ui.getUser().getFriends()){
				if(m_userMap.containsKey(nei)){	
					uj = m_userMap.get(nei);
					if(!isContain(uj.getUser().getFriends(), ui.getUserID())){
						System.out.print("(o,x)");
					}
				}
			}
		}
	}
	
	public boolean isContain(String[] strs, String s){
		for(String str: strs){
			if(str.equals(s))
				return true;
		}
		return false;
	}
	@Override
	public void initThetaStars(){
		// assign each review to one cluster.
		super.initThetaStars();
		_HDPAdaptStruct ui, uj;
		
		m_userMap = new HashMap<String, _HDPAdaptStruct>();
		
		// add the friends one by one.
		for(int i=0; i< m_userList.size(); i++){
			ui = (_HDPAdaptStruct) m_userList.get(i);
			m_userMap.put(ui.getUserID(), (_HDPAdaptStruct) ui);

			for(int j=i+1; j<m_userList.size(); j++){
				uj = (_HDPAdaptStruct) m_userList.get(j);
				// if ui and uj are friends, sample one edge.
				if(isContain(ui.getUser().getFriends(), uj.getUserID())){
					sampleEdge(i, j, 1);
					sampleEdge(j, i, 1);
				} else{
				// else sample indicators for zero edge, we treat all zero edges sampled from mmb at beginning.
					sampleEdge(i, j, 0);
					sampleEdge(j, i, 0); 
				}
			}
		}
	}

	@Override
	// The function is used in "sampleOneInstance".
	public double calcGroupPopularity(_HDPAdaptStruct user, int k, double gamma_k){
		return user.getHDPThetaMemSize(m_hdpThetaStars[k]) + m_eta*gamma_k + user.getHDPThetaEdgeSize(m_hdpThetaStars[k]);
	}
	
	// Sample one edge from mmb and e represents the edge value.
	public void sampleEdge(int i,int j, int e){
		int k;
		_HDPAdaptStruct ui = (_HDPAdaptStruct) m_userList.get(i);
		_HDPAdaptStruct uj = (_HDPAdaptStruct) m_userList.get(j);

		double likelihood, gamma_k, logSum = 0;
		for(k=0; k<m_kBar; k++){
			
			ui.setThetaStar(m_hdpThetaStars[k]);
			
			//log likelihood of the edge p(e_{ij}, z, B)
			// p(eij|z_{i->j}, z_{j->i}, B)*p(z_{i->j}|\pi_i)*p(z_{j->i|\pj_j})
			likelihood = calcLogLikelihoodE(ui, uj, e);
						
			//p(z=k|\gamma,\eta)
			gamma_k = m_hdpThetaStars[k].getGamma();
			likelihood += Math.log(calcGroupPopularity(ui, k, gamma_k));
			
			m_hdpThetaStars[k].setProportion(likelihood);//this is in log space!
						
			if(k==0) 
				logSum = likelihood;
			else 
				logSum = Utils.logSum(logSum, likelihood);
		}
		logSum = Utils.logSum(logSum, m_pNew[e]);
		
		//Sample group k with likelihood.
		k = sampleEdgeInLogSpace(logSum, e);
		
		if(k == -1) 
			sampleNewCluster4Edge();// shall we consider the current edge?? posterior sampling??
		
		//Step 3: update the setting after sampling z_ij.
		m_hdpThetaStars[m_kBar-1].updateEdgeCount(1, 1);//first 1 means edge 1, the second one mean increase by 1.
		ui.addNeighbor(uj, m_hdpThetaStars[m_kBar-1], 1);
		
		//Step 4: Update the user info with the newly sampled hdpThetaStar.
		ui.incHDPThetaStarEdgeSize(m_hdpThetaStars[m_kBar-1], 1);//-->3	
		
		//Step 5: Put the reference to the matrix for later usage.
		//Since we have all the info, we don't need to put the theta info in the _MMBNeighbor structure.
		m_indicator[i][j] = m_hdpThetaStars[m_kBar-1];
	}
	
	//Sample hdpThetaStar with likelihood.
	protected int sampleEdgeInLogSpace(double logSum, int e){
		logSum += Math.log(FloatUniform.staticNextFloat());//we might need a better random number generator
			
		int k = -1;
		double newLogSum = m_pNew[e];
		do {
			if (newLogSum>=logSum)
				break;
			k++;
			newLogSum = Utils.logSum(newLogSum, m_hdpThetaStars[k].getProportion());
		} while (k<m_kBar);
			
		if (k==m_kBar)
			k--; // we might hit the very last
		return k;
	}
	
	// sample eij = 0 from the joint probabilities of cij, zij and zji.
	public void sampleZeroEdgeJoint(int i, int j){
		/***we will consider all possible combinations of different memberships.
		 * 1.cij=0, cji=0, prob: (1-\rho)(1-\rho), 1 case
		 * 2.cij=1, cji=1, known (Bgh, Bhg), prob: \rho\rho(1-Bgh)(1-Bhg), k^2 possible cases
		 * 3.cij=1, cji=1, unknows (Bgh Bhg), prob: \Gamma(a)\Gamma(b+1)/\Gamma(a+b+1), 2k+1 possible cases */
		
		double bij = 0, bji = 0, logSum = 0, prob = 0;
		_HDPAdaptStruct ui = (_HDPAdaptStruct) m_userList.get(i);
		_HDPAdaptStruct uj = (_HDPAdaptStruct) m_userList.get(j);

		// If the matrix does not change (no new clusters added.)
		m_cache = new double[m_kBar+1][m_kBar+1];
		
		// Step 1: calc prob for different cases of cij, cji.
		_HDPThetaStar theta_g, theta_h;
		// case 1: existing thetas.
		for(int g=0; g<m_kBar; g++){
			theta_g = m_hdpThetaStars[g];
			for(int h=0; h<m_kBar; h++){
				theta_h = m_hdpThetaStars[h];
				bij = theta_g.getOneB(theta_h);
				bji = theta_h.getOneB(theta_g);
				// m_rho * m_rho * (1 - bij) * (1 - bji)
				prob = Math.log(m_rho)+Math.log(m_rho)+Math.log(1-bij)+Math.log(1-bji);
				m_cache[g][h] = prob;
			}
		}
		// case 2: either one is from new cluster.
		for(int k=0; k<=m_kBar; k++){
			m_cache[k][m_kBar] = m_pNew_0;
			m_cache[m_kBar][k] = m_pNew_0;
		}
		// Accumulate the log sum
		for(int g=0; g<m_cache.length; g++){
			for(int h=0; h<m_cache[0].length; h++){
				logSum = Utils.logSum(logSum, m_cache[g][h]);
			}
		}
		// case 3: background model while the prob is not stored in the two-dim array.
		prob = Math.log(1 - m_rho) + Math.log(1 - m_rho);
		logSum = Utils.logSum(logSum, prob);
		
		// Step 2: sample one pair from the prob matrix./*-
		int k = sampleIn2DimArrayLogSpace(logSum, prob);
		
		// Step 3: Analyze the sampled cluster results.
		// case 1: k == -1, sample from the background model;
		// case 2: k != -1, sample from mmb model.
		int g = 0, h = 0;
		if(k != -1){
			g = k / (m_kBar+1);
			h = k % (m_kBar+1);
			if(g == m_kBar || h == m_kBar){
				// we need to sample the new cluster
				sampleNewCluster4Edge();// shall we consider the current edge?? posterior sampling??
			}
			
			// Update the thetaStar and user info after getting z_ij.
			m_hdpThetaStars[g].updateEdgeCount(0, 1);//-->1
			ui.addNeighbor(uj, m_hdpThetaStars[g], 0);
			ui.incHDPThetaStarEdgeSize(m_hdpThetaStars[g], 1);	
			m_indicator[i][j] = m_hdpThetaStars[g];
			m_Ne++;
			
			// Update the thetaStar and user info after getting z_ji.
			m_hdpThetaStars[h].updateEdgeCount(0, 1);
			uj.addNeighbor(ui, m_hdpThetaStars[h], 0);
			uj.incHDPThetaStarEdgeSize(m_hdpThetaStars[h], 1);
			m_indicator[i][j] = m_hdpThetaStars[h];
			m_Ne++; 
		} else{
			m_Le++;// else it belongs to background model.
		}
	}
	
	
	//Sample hdpThetaStar with likelihood.
	protected int sampleIn2DimArrayLogSpace(double logSum, double back_prob){
		double sum = back_prob;
		for(int i=0; i<m_kBar+1; i++){
			for(int j=0; j<m_kBar+1; j++){
				sum = Utils.logSum(sum, m_cache[i][j]);
			}
		}
		logSum += Math.log(FloatUniform.staticNextFloat());//we might need a better random number generator
		
		int k = -1;
		// we start from the background model.
		double newLogSum = back_prob;
		do {
			if (newLogSum>=logSum)
				break;
			k++;
			if (k==(m_kBar+1)*(m_kBar+1)){
				k--; // we might hit the very last
				return k;
			}
			newLogSum = Utils.logSum(newLogSum, m_cache[k/(m_kBar+1)][k%(m_kBar+1)]);
			
		} while (k<(m_kBar+1)*(m_kBar+1));
		return k;
	}
	@Override
	// Sample new cluster based on sampling of z_{i,d}, thus, the cluster will not have edges.
	public void sampleNewCluster(int k, _SparseFeature[] fvs){		
		super.sampleNewCluster(k, fvs);
		// we need to sample the values for B and the new one is in index kBar-1 since kBar has increased.
		m_hdpThetaStars[m_kBar-1].initB();
		sampleB(m_hdpThetaStars[m_kBar-1]);
	}
	
	// Sample new cluster based on sampling of z_{i->j}, thus, the cluster will have edges info.
	public void sampleNewCluster4Edge(){
		// use the first available one as the new cluster.
		if(m_hdpThetaStars[m_kBar] == null){			
			m_hdpThetaStars[m_kBar] = new _HDPThetaStar(m_dim, 0);
		}
		m_hdpThetaStars[m_kBar].initPsiModel(m_lmDim);
		
		// we don't have fvs for sampling of language model parameters
		m_D0.sampling(m_hdpThetaStars[m_kBar].getPsiModel(), m_betas, true);//we should sample from Dir(\beta)
		
		// we have edge info for sampling of B
		m_hdpThetaStars[m_kBar].initB();
		sampleB(m_hdpThetaStars[m_kBar]);
		double rnd = Beta.staticNextDouble(1, m_alpha);
		m_hdpThetaStars[m_kBar].setGamma(rnd*m_gamma_e);
		m_gamma_e = (1-rnd)*m_gamma_e;
		m_kBar++;
	}
	// sample each element of the vector, including the new cluster.
	public void sampleB(_HDPThetaStar theta){
		// Add itself.
		theta.addOneB(theta, m_Beta.sample());
		for(int k=0; k<m_kBar; k++){
			// add the prob between B_{existing theta, new theta} to the hashmap. 
			m_hdpThetaStars[k].addOneB(theta, m_Beta.sample());
			// add the prob between B_{new theta, existing theta} to the hashmap. 
			theta.addOneB(m_hdpThetaStars[k], m_Beta.sample());
		}
	}
	
	// The function calculates the likelihood given by one edge from mmb model.
	protected double calcLogLikelihoodE(_HDPAdaptStruct ui, _HDPAdaptStruct uj, int e){
		if(ui.hasEdge(uj)){
			// probability for Bernoulli distribution: p(e_ij|z_{i->j}, z_{j->i},B)
			double bij = ui.getOneNeighbor(uj).getHDPThetaStar().getOneB(uj.getOneNeighbor(uj).getHDPThetaStar());
			double loglikelihood = 0;
			loglikelihood = e == 0 ? (1 - bij) : bij;
			loglikelihood = Math.log(loglikelihood);
			return loglikelihood;
		} else{
			return m_pNew[e];
		}
	}
	// Init the counters for different edges in E steps.
	private void initE(){
		m_Me = 0;
		m_Ne = 0;
		m_Le = 0;
	}
	
	/***
	 * The current sampling scheme for zero edges (e_0) is as follows:
	 * We assign group indicator to all zero edges (e_0) in the init stage.
	 * In E step: we decide whether e_0 is from background model or mmb model base on probs:
	 * p(background model)=1-\rho, p(e_0 from mmb)=\rho(1-z_{i->j}Bz_{j->i})
	 */
	protected void calculate_E_step(){
		initE();
		// sample z_{i,d}
		super.calculate_E_step();

		// sample z_{i->j}
		_HDPAdaptStruct ui, uj;
		double m_p_bk = 1-m_rho, m_p_mmb_0 = 0;
		for(int i=0; i<m_userList.size(); i++){
			ui = (_HDPAdaptStruct) m_userList.get(i);
			for(int j=i+1; j<m_userList.size(); j++){
				uj = (_HDPAdaptStruct) m_userList.get(j);

				// eij = 1
				if(ui.hasEdge(uj) && ui.getEdge(uj) == 1){
					updateOneMembership(ui, uj, 1);// update membership from ui->uj					
					updateOneMembership(uj, ui, 1);// update membership from uj->ui
					
					sampleEdge(i, j, 1);
					sampleEdge(j, i, 1);
					m_Me += 2;

				}else{
					// eij = 0 from mmb， remove the membership first.
					if(ui.hasEdge(uj) && ui.getEdge(uj) == 0){
						updateOneMembership(ui, uj, 0);
						updateOneMembership(uj, ui, 0);
					}
					// Decide whether it belongs to mmb or background model.
					// If Bij and Bji exist
					if(isBijValid(i, j)){
						m_p_mmb_0 = m_rho*(1-getBij(i, j));
						// sample i->j
						m_bernoulli = new BinomialDistribution(1, m_p_bk/(m_p_bk+m_p_mmb_0));
						// put the edge(two nodes) in the background model.
						if(m_bernoulli.sample() == 1){
							m_Le+=2;// the two nodes of the edges are assigned to background model
							continue; // their cluster assignment keep the same.
						}
						// sample from mmb for zero edges.
						sampleEdge(i, j, 0);
						sampleEdge(j, i, 0);
						m_Ne += 2;
					} else{
						sampleZeroEdgeJoint(i, j);
					}
				}
				checkSampleSize((int)(m_Me+m_Ne+m_Le));
			}
		}
		System.out.print(String.format("[Info]eij=1: %d, eij=0(mmb):%d, eij=0(background):%d\n",m_Me, m_Ne,m_Le));
	}
	
	// If both of them exist, then it is valid.
	// In case the previously assigned cluster is removed, we use the joint sampling to get it.*/
	public boolean isBijValid(int i, int j){
		return (m_indicator[i][j] != null) && (m_indicator[j][i] != null);
	}
	
	// z_{i->j}Bz_{j->i}
	// Get the probability for later sampling for both zero edge from mmb or background.
	public double getBij(int i, int j){
		_HDPThetaStar z_ij = m_indicator[i][j];
		_HDPThetaStar z_ji = m_indicator[j][i];
		return z_ij.getOneB(z_ji);
	}
	// Check the sample size and print out hint information.
	public void checkSampleSize(int sampleSize){
		if (sampleSize%2000==0) {
			System.out.print('.');
			if (sampleSize%100000==0)
				System.out.println();
		}
	}
	public void updateOneMembership(_HDPAdaptStruct ui, _HDPAdaptStruct uj, int e){
		int index = 0;
		_HDPThetaStar thetai = ui.getThetaStar(uj);
		thetai.updateEdgeCount(e, -1);
		
		// remove the neighbor from user.
		ui.rmNeighbor(uj);
		
		if(thetai.getMemSize() == 0 && thetai.getTotalEdgeSize() == 0){// No data associated with the cluster.
			m_gamma_e += thetai.getGamma();
			index = findHDPThetaStar(thetai);
			swapTheta(m_kBar-1, index); // move it back to \theta*
			thetai = null;// Clear the probability vector.
			m_kBar --;
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
		_HDPAdaptStruct ui;
		_MMBNeighbor mui, muj;
		double[][][] B = new double[m_kBar][m_kBar][2];
		HashMap<_HDPAdaptStruct, _MMBNeighbor> neighborsMap;
		// Iterate through all the user pairs.
		for(int i=0; i<m_userList.size(); i++){
			int g = 0, h = 0;
			ui = (_HDPAdaptStruct) m_userList.get(i);
			neighborsMap = ui.getNeighbors();
			for(_HDPAdaptStruct uj: neighborsMap.keySet()){
				if(uj == null) 
					System.out.print("u");
				muj = neighborsMap.get(uj);
				mui = uj.getOneNeighbor(ui);
				
				if(mui == null || muj == null)
					System.out.print("m");
				
				if(muj.getHDPThetaStar() == null || mui.getHDPThetaStar() == null)
					System.out.print("mt");
				
				g = muj.getHDPThetaStar().getIndex();
				h = mui.getHDPThetaStar().getIndex();
				/*** B(g, h) = sum_{m+a-1/m+n+a+b-2}
				 * m is the total number of edges eij=1 among the edges {i->j \in g, j->i \in h}
				 * n is the total number of edges eij=0 among the edges {i->j \in g, j->i \in h}*/
				B[g][h][muj.getEdge()]++;
				B[h][g][mui.getEdge()]++;
			}
		}
		m_Bs = new double[m_kBar][m_kBar];
		for(int g=0; g<m_kBar; g++){
			for(int h=0; h<m_kBar; h++){
				m_Bs[g][h] = (B[g][h][1]+m_abcd[0]-1)/(B[g][h][0]+B[g][h][1]+m_abcd[0]+m_abcd[1]-2);
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
	// \rho = (M+N+c-1)/(M+N+L+c+d-2)
	public double estRho(){
		m_rho = (m_Me+m_Ne+m_abcd[2]-1)/(m_Me+m_Ne+m_Le+m_abcd[2]+m_abcd[3]-2);
		return 0;
	}
}
