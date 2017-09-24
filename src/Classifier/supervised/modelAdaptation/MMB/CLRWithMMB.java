package Classifier.supervised.modelAdaptation.MMB;

import cern.jet.random.tdouble.Beta;
import cern.jet.random.tfloat.FloatUniform;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.HDP.CLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP._HDPAdaptStruct;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import org.apache.commons.math3.distribution.BinomialDistribution;
import structures._HDPThetaStar;
import structures._MMBNeighbor;
import structures._Review;
import structures._User;
import utils.Utils; 

public class CLRWithMMB extends CLRWithHDP {
	// sparsity parameter
	protected double m_rho = 0.1; 
	// As we store all the indicators for all the edges(even edges from background model), we maintain a matrix for indexing.
	protected _HDPThetaStar[][] m_indicator;
	
	// prob for the new cluster in sampling mmb edges.
	private double[] m_pNew = new double[2]; 
	// parameters used in the gamma function in mmb model, prior of B~beta(a, b), prior of \rho~Beta(c, d)
	private double[] m_abcd = new double[]{2, 2, 2, 2}; 
	// The probability for the new cluster in the joint probabilities.
	private double m_pNewJoint = 2*(Math.log(m_rho) + Math.log(m_abcd[1]) - Math.log(m_abcd[0]+m_abcd[1]));
	// Me: total number of edges eij=0;Ne: total number of edges eij=1 from mmb; Le: total number of edges eij=0 from background model.
	private double[] m_MNL = new double[3];
	// two-dim array for storing probs used in sampling zero edge.
	private double[][] m_cache; 
	// Bernoulli distribution used in deciding whether the edge belongs to mmb or background model.
	private BinomialDistribution m_bernoulli;
	// key: userID, value: _AdaptStruct
	private HashMap<String, _MMBAdaptStruct> m_userMap;
	// for debug purpose
	private HashMap<String, ArrayList<Integer>> stat = new HashMap<>();

	public CLRWithMMB(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel,
			double[] betas) {
		super(classNo, featureSize, featureMap, globalModel, betas);
		calcProbNew();
		initStat();
	} 
	
	public CLRWithMMB(int classNo, int featureSize, String globalModel,
			double[] betas) {
		super(classNo, featureSize, globalModel, betas);
		calcProbNew();
		initStat();
	} 
	
	// add the edge assignment to corresponding cluster
	public void addConnection(_MMBAdaptStruct ui, _MMBAdaptStruct uj, int e){
		_HDPThetaStar theta_g, theta_h;
		theta_g = ui.getOneNeighbor(uj).getHDPThetaStar();
		theta_h = uj.getOneNeighbor(ui).getHDPThetaStar();
		theta_g.addConnection(theta_h, e);
		theta_h.addConnection(theta_g, e);
	}
	
	// calculate the probability for generating new clusters in sampling edges.
	private void calcProbNew(){
		// if e_ij = 0, p = \rho*b/(a+b)
		m_pNew[0] = Math.log(m_rho) + Math.log(m_abcd[1]) - Math.log(m_abcd[0] + m_abcd[1]);
		// if e_ij = 1, p = \rho*a/(a+b)
		m_pNew[1] = Math.log(m_rho) + Math.log(m_abcd[0]) - Math.log(m_abcd[0] + m_abcd[1]);
	}

	@Override
	// calculate the group popularity in sampling documents and edges.
	protected double calcGroupPopularity(_HDPAdaptStruct u, int k, double gamma_k){
		_MMBAdaptStruct user= (_MMBAdaptStruct) u;
//		System.out.print(String.format("%d/%d, %.5f, %d/%d\n", user.getHDPThetaMemSize(m_hdpThetaStars[k]), m_hdpThetaStars[k].getMemSize(), m_eta*gamma_k, user.getHDPThetaEdgeSize(m_hdpThetaStars[k]), m_hdpThetaStars[k].getTotalEdgeSize()));
		return user.getHDPThetaMemSize(m_hdpThetaStars[k]) + m_eta*gamma_k + user.getHDPThetaEdgeSize(m_hdpThetaStars[k]);
	}
	
	// The function calculates the likelihood given by one edge from mmb model.
	protected double calcLogLikelihoodE(int k, _MMBAdaptStruct ui, _MMBAdaptStruct uj, int e){
		double likelihood = 0, e_1 = 0, e_0 = 0;
		_HDPThetaStar theta_s, theta_h;
		// Likelihood for e=0: \rho*(b+e_1)/(a+b+e_0+e_1)
		// Likelihood for e=1: \rho*(a+e_0)/(a+b+e_0+e_1)
		if(ui.hasEdge(uj)){
			theta_s = m_hdpThetaStars[k];
			theta_h = uj.getOneNeighbor(ui).getHDPThetaStar();
			e_0 = theta_s.getConnectionSize(theta_h, 0);
			e_1 = theta_s.getConnectionSize(theta_h, 1);
			likelihood = m_rho /(m_abcd[0] + m_abcd[1] + e_0 + e_1);
			if(e == 0){
				likelihood *= (m_abcd[1] + e_1);
			} else if(e == 1){
				likelihood *= (m_abcd[0] + e_0);
			}
			return Math.log(likelihood);
		} else{
			return m_pNew[e];
		}
	}
	
	// posterior predictive distribution: \int_{B_gh} (1-B_{gh}*prior d_{B_{gh}}
	// prior is Beta(a+e_1, b+e_0)
	// \gamma(a+b)\gamma(a+e_1)\gamma(b+e_0+1) / \gamma(a)\gamma(b)\gamma(a+b+e_0+e_1+1)
	public double calcLogPostPredictiveBgh(_HDPThetaStar theta_g, _HDPThetaStar theta_h){
		double prob = 0;
		double e_0 = 0, e_1 = 0;
		e_0 = theta_g.getConnectionSize(theta_h, 0);
		e_1 = theta_g.getConnectionSize(theta_h, 1);
		prob = Utils.lgamma(m_abcd[0]+m_abcd[1])+Utils.lgamma(m_abcd[0]+e_1)+Utils.lgamma(m_abcd[1]+e_0+1)
				-Utils.lgamma(m_abcd[0])-Utils.lgamma(m_abcd[1])-Utils.lgamma(m_abcd[0]+m_abcd[1]+e_0+e_1+1);
		return prob;
	}
	
	protected void calculate_E_step_Edge(){
		Arrays.fill(m_MNL, 0);
		// sample z_{i->j}
		_MMBAdaptStruct ui, uj;
		double m_p_bk = 1-m_rho, m_p_mmb_0 = 0;
		for(int i=0; i<m_userList.size(); i++){
			ui = (_MMBAdaptStruct) m_userList.get(i);
			for(int j=i+1; j<m_userList.size(); j++){
				uj = (_MMBAdaptStruct) m_userList.get(j);

				// eij = 1
				if(ui.hasEdge(uj) && ui.getEdge(uj) == 1){
					// remove the connection for B_gh, i->j \in g, j->i \in h.
					rmConnection(ui, uj, 1);
					// update membership from ui->uj, remove the edge
					updateEdgeMembership(ui, uj, 1);					
					// update membership from uj->ui, remove the edge
					updateEdgeMembership(uj, ui, 1);
					// sample new clusters for the two edges
					sampleEdges(i, j, 1);
					// add the new connection for B_g'h', i->j \in g', j->i \in h'
					addConnection(ui, uj, 1);
					// update the sample size with the specified index and value
					updateSampleSize(1, 2);

				}else{
					// eij = 0 from mmb， remove the membership first.
					if(ui.hasEdge(uj) && ui.getEdge(uj) == 0){
						rmConnection(ui, uj, 0);
						updateEdgeMembership(ui, uj, 0);
						updateEdgeMembership(uj, ui, 0);
					}
					// Decide edge eij = 0 with indicator z_i->j belongs to mmb or background model.
					// Calculate the prob first: p_mmb_0 = \rho*(1-B_gh)
					if(isBijValid(i, j)){
						m_p_mmb_0 = m_rho * Math.exp(calcLogPostPredictiveBgh(m_indicator[i][j], m_indicator[j][i]));
						// init the bernoulli distribution first with normalized parameter to sample the edge indicator i->j
						m_bernoulli = new BinomialDistribution(1, m_p_bk/(m_p_bk+m_p_mmb_0));
						// put the edge(two nodes) in the background model.
						if(m_bernoulli.sample() == 1){
							updateSampleSize(2, 2);
							continue; // their cluster assignment keep the same.
						}
						// sample from mmb for zero edges.
						sampleEdges(i, j, 0);
						// add the connection information to thetas.
						addConnection(ui, uj, 0);
						updateSampleSize(0, 2);
					} else{
						sampleZeroEdgeJoint(i, j);
					}
				}
			}
		}
		System.out.print(String.format("\n[Info]kBar: %d, eij=0(mmb): %.1f, eij=1:%.1f, eij=0(background):%.1f\n", m_kBar, m_MNL[0], m_MNL[1],m_MNL[2]));
		System.out.print(String.format("[Info]mmb_0 prob: %.5f, background prob: %.5f\n",m_p_mmb_0, m_p_bk));
		checkClusters();
	}

	private void check(){
		double graphEdgeCount = 0, edgeCount = 0;
		m_userMap = new HashMap<String, _MMBAdaptStruct>();
		// construct the map.
		for(_AdaptStruct ui: m_userList)
			m_userMap.put(ui.getUserID(), (_MMBAdaptStruct) ui);
		
		// add the friends one by one.
		for(_AdaptStruct ui: m_userList){
			graphEdgeCount += ui.getUser().getFriends().length;
			for(String nei: ui.getUser().getFriends()){
				if(!m_userMap.containsKey(nei)){
					System.out.print("o");
				} else{
					String[] frds = m_userMap.get(nei).getUser().getFriends();
					if(hasFriend(frds, ui.getUserID())){
						System.out.print("y");
						edgeCount++;
					}
					else
						System.out.print("x");
				}
			}
			System.out.println();
		}
		System.out.print(String.format("[Info]Graph avg edge size: %.4f, avg edge size: %.4f, user size: %d\n", graphEdgeCount/m_userList.size(), edgeCount/m_userList.size(), m_userList.size()));
	}
	
	private void checkClusters(){
		int index = 0;
		int zeroDoc = 0, zeroEdge = 0;
		while(m_hdpThetaStars[index] != null){
			if(index < m_kBar && m_hdpThetaStars[index].getTotalEdgeSize() == 0)
				zeroEdge++;
			if(index < m_kBar && m_hdpThetaStars[index].getMemSize() == 0)
				zeroDoc++;
			index++;
		}
		stat.get("onlyedges").add(zeroDoc);
		stat.get("onlydocs").add(zeroEdge);
		stat.get("mixture").add(m_kBar-zeroDoc-zeroEdge);
		
		System.out.print(String.format("[Info]Clusters with only edges: %d, Clusters with only docs: %d, kBar:%d, non_null hdp: %d\n", zeroDoc, zeroEdge, m_kBar, index));
	}
	
	// check if the sum(m_MNL) == sum(edges of all clusters)
	protected void checkEdges(){
		int mmb_0 = 0, mmb_1 = 0;
		_HDPThetaStar theta;
		for(int i=0; i<m_kBar; i++){
			theta = m_hdpThetaStars[i];
			mmb_0 += theta.getEdgeSize(0);
			mmb_1 += theta.getEdgeSize(1);
		}
		if(mmb_0 != m_MNL[0])
			System.out.println("Zero edges sampled from mmb is not correct!");
		if(mmb_1 != m_MNL[1])
			System.out.println("One edges sampled from mmb is not correct!");
	}

	// Estimate the sparsity parameter.
	// \rho = (M+N+c-1)/(M+N+L+c+d-2)
	public double estRho(){
		m_rho = (m_MNL[0]+m_MNL[1]+m_abcd[2]-1)/(m_MNL[0]+m_MNL[1]+m_MNL[2]+m_abcd[2]+m_abcd[3]-2);
		return 0;
	}
	
	public boolean hasFriend(String[] arr, String str){
		for(String a: arr){
			if(str.equals(a))
				return true;
		}
		return false;
	}
	
	private void initStat(){
		stat.put("onlyedges", new ArrayList<Integer>());
		stat.put("onlydocs", new ArrayList<Integer>());
		stat.put("mixture", new ArrayList<Integer>());
	}
	
	// init thetas for edges at the beginning
	public void initThetaStars4Edges(){
		_MMBAdaptStruct ui, uj;
		
		m_userMap = new HashMap<String, _MMBAdaptStruct>();
		
		// add the friends one by one.
		for(int i=0; i< m_userList.size(); i++){
			ui = (_MMBAdaptStruct) m_userList.get(i);
			m_userMap.put(ui.getUserID(), (_MMBAdaptStruct) ui);

			for(int j=i+1; j<m_userList.size(); j++){
				uj = (_MMBAdaptStruct) m_userList.get(j);
				// if ui and uj are friends, random sample clusters for the two connections
				// e_ij = 1, z_{i->j}, e_ji = 1, z_{j -> i} = 1
				if(hasFriend(ui.getUser().getFriends(), uj.getUserID())){
					// sample two edges between i and j
					randomSampleEdges(i, j, 1);
					// add the edge assignment to corresponding cluster
					// we have to add connections after we know the two edge assignment (the clusters for i->j and j->i)
					addConnection(ui, uj, 1);
					// update the sample size with the specified index and value
					// index 0 : e_ij = 0 from mmb; index 1 : e_ij = 1 from mmb; index 2 : 0 from background model
					updateSampleSize(1, 2);
				} else{
				// else sample indicators for zero edge, we treat all zero edges sampled from mmb at beginning.
					randomSampleEdges(i, j, 0);
					addConnection(ui, uj, 0);
					updateSampleSize(0, 2);
				}
			}
		}
	}
	
	// If both of them exist, then it is valid.
	// In case the previously assigned cluster is removed, we use the joint sampling to get it.*/
	public boolean isBijValid(int i, int j){
		return m_indicator[i][j].isValid() && m_indicator[j][i].isValid();
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		for(_User user:userList)
			m_userList.add(new _MMBAdaptStruct(user));
		m_pWeights = new double[m_gWeights.length];			
		m_indicator = new _HDPThetaStar[m_userList.size()][m_userList.size()];
	}
	
	// if ui and uj are friends, random sample clusters for the two connections
	// e_ij = 1, z_{i->j}, e_ji = 1, z_{j -> i} = 1
	private void randomSampleEdges(int i, int j, int e){
		randomSampleEdge(i, j, e);
		randomSampleEdge(j, i, e);
	}
	
	/*** In order to avoid creating too many thetas, we randomly assign nodes to thetas at beginning.
	 *   and this sampling function is only used for initial states.***/
	private void randomSampleEdge(int i,int j, int e){
		_MMBAdaptStruct ui = (_MMBAdaptStruct) m_userList.get(i);
		_MMBAdaptStruct uj = (_MMBAdaptStruct) m_userList.get(j);
		
		// Random sample one cluster.
		int k = (int) (Math.random() * m_kBar);
		
		// Step 3: update the setting after sampling z_ij
		// update the edge count for the cluster: first param means edge (0 or 1), the second one mean increase by 1.
		m_hdpThetaStars[k].updateEdgeCount(e, 1);
		// update the neighbor information to the neighbor hashmap 
		ui.addNeighbor(uj, m_hdpThetaStars[k], e);
		// update the user info with the newly sampled hdpThetaStar
		ui.incHDPThetaStarEdgeSize(m_hdpThetaStars[k], 1);//-->3	
			
		// Step 5: Put the cluster info in the matrix for later use
		// Since we have all the info, we don't need to put the theta info in the _MMBNeighbor structure.
		m_indicator[i][j] = m_hdpThetaStars[k];
	}

	// remove the connection between ui and uj, where i->j \in g, j->i \in h.
	public void rmConnection(_MMBAdaptStruct ui, _MMBAdaptStruct uj, int e){
		_HDPThetaStar theta_g, theta_h;
		theta_g = ui.getOneNeighbor(uj).getHDPThetaStar();
		theta_h = uj.getOneNeighbor(ui).getHDPThetaStar();
		theta_g.rmConnection(theta_h, e);
		theta_h.rmConnection(theta_g, e);
	}
	
	private void sampleEdges(int i, int j, int e){
		sampleEdge(i, j, e);
		sampleEdge(j, i, e);
	}
	
	// Sample one edge from mmb and e represents the edge value.
	public void sampleEdge(int i,int j, int e){
		int k;
		_MMBAdaptStruct ui = (_MMBAdaptStruct) m_userList.get(i);
		_MMBAdaptStruct uj = (_MMBAdaptStruct) m_userList.get(j);

		double likelihood, logNew, gamma_k, logSum = 0;
		for(k=0; k<m_kBar; k++){
			// In the initial states, we have to create new clusters.
			ui.setThetaStar(m_hdpThetaStars[k]);
			
			//log likelihood of the edge p(e_{ij}, z, B)
			// p(eij|z_{i->j}, z_{j->i}, B)*p(z_{i->j}|\pi_i)*p(z_{j->i|\pj_j})
			likelihood = calcLogLikelihoodE(k, ui, uj, e);
						
			//p(z=k|\gamma,\eta)
			gamma_k = m_hdpThetaStars[k].getGamma();
			likelihood += Math.log(calcGroupPopularity(ui, k, gamma_k));
			
			m_hdpThetaStars[k].setProportion(likelihood);//this is in log space!
						
			if(k==0) 
				logSum = likelihood;
			else 
				logSum = Utils.logSum(logSum, likelihood);
		}
		// fix1: the probability for new cluster
		logNew = Math.log(m_eta*m_gamma_e) + m_pNew[e];
		logSum = Utils.logSum(logSum, logNew);
		
		//Sample group k with likelihood.
		k = sampleEdgeInLogSpace(logSum, e);
		
		if(k == -1){
			sampleNewCluster4Edge();// shall we consider the current edge?? posterior sampling??
			k = m_kBar - 1;
		}
		// update the setting after sampling z_ij.
		m_hdpThetaStars[k].updateEdgeCount(e, 1);//first 1 means edge 1, the second one mean increase by 1.
		
		// update the user info with the newly sampled hdpThetaStar.
		ui.addNeighbor(uj, m_hdpThetaStars[k], e);
		
		ui.incHDPThetaStarEdgeSize(m_hdpThetaStars[k], 1);//-->3	
		
		// Put the reference to the matrix for later usage.
		// Since we have all the info, we don't need to put the theta info in the _MMBNeighbor structure.
		m_indicator[i][j] = m_hdpThetaStars[k];
	}
	
	//Sample hdpThetaStar with likelihood.
	protected int sampleEdgeInLogSpace(double logSum, int e){
		logSum += Math.log(FloatUniform.staticNextFloat());//we might need a better random number generator
			
		int k = -1;
		// [fixed bug], the prob for new cluster should consider the gamma too.
//		double newLogSum = m_pNew[e];
		double newLogSum = Math.log(m_eta*m_gamma_e) + m_pNew[e];
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
	
	// Sample new cluster based on sampling of z_{i->j}, thus, the cluster will have edges info.
	public void sampleNewCluster4Edge(){
		// use the first available one as the new cluster.	
		if (m_hdpThetaStars[m_kBar] == null){
			if (this instanceof CLinAdaptWithMMB)// this should include all the inherited classes for adaptation based models
				m_hdpThetaStars[m_kBar] = new _HDPThetaStar(2*m_dim);
			else
				m_hdpThetaStars[m_kBar] = new _HDPThetaStar(m_dim);
		}
		
		m_hdpThetaStars[m_kBar].enable();
		m_G0.sampling(m_hdpThetaStars[m_kBar].getModel());
		m_hdpThetaStars[m_kBar].initLMStat(m_lmDim);

		double rnd = Beta.staticNextDouble(1, m_alpha);
		m_hdpThetaStars[m_kBar].setGamma(rnd*m_gamma_e);
		m_gamma_e = (1-rnd)*m_gamma_e;
		m_kBar++;
	}
	
	@Override
	public void sampleThetaStars(){
		double gamma_e = m_gamma_e/m_M;
		for(int m=m_kBar; m<m_kBar+m_M; m++){
			if (m_hdpThetaStars[m] == null){
				if (this instanceof CLinAdaptWithMMB)// this should include all the inherited classes for adaptation based models
					m_hdpThetaStars[m] = new _HDPThetaStar(2*m_dim, gamma_e);
				else
					m_hdpThetaStars[m] = new _HDPThetaStar(m_dim, gamma_e);
			} else
				m_hdpThetaStars[m].setGamma(gamma_e);//to unify the later operations
			
			//sample \phi from Normal distribution.
			m_G0.sampling(m_hdpThetaStars[m].getModel());//getModel-> get \phi.
		}
	}

	// sample eij = 0 from the joint probabilities of cij, zij and zji.
	public void sampleZeroEdgeJoint(int i, int j){
		/***we will consider all possible combinations of different memberships.
		 * 1.cij=0, cji=0, prob: (1-\rho)(1-\rho), 1 case
		 * 2.cij=1, cji=1, known (Bgh, Bhg), prob: \rho\rho(1-Bgh)(1-Bhg), k^2 possible cases
		 * 3.cij=1, cji=1, unknows (Bgh Bhg), prob: \Gamma(a)\Gamma(b+1)/\Gamma(a+b+1), 2k+1 possible cases */
		
		double logSum = 0, prob = 0;
		_MMBAdaptStruct ui = (_MMBAdaptStruct) m_userList.get(i);
		_MMBAdaptStruct uj = (_MMBAdaptStruct) m_userList.get(j);

		// If the matrix does not change (no new clusters added.)
		m_cache = new double[m_kBar+1][m_kBar+1];
		
		// Step 1: calc prob for different cases of cij, cji.
		_HDPThetaStar theta_g, theta_h;
		// case 1: existing thetas.
		for(int g=0; g<m_kBar; g++){
			theta_g = m_hdpThetaStars[g];
			for(int h=0; h<m_kBar; h++){
				theta_h = m_hdpThetaStars[h];
				// \int_{B_gh}\int_{B_hg} m_rho * m_rho * (1 - b_gh) * (1 - b_hg)d_{B_gh}d_{B_hg}
//				prob = Math.log(m_rho)+Math.log(m_rho)+Math.log(1-b_gh)+Math.log(1-b_hg);
				m_cache[g][h] = 2 * Math.log(m_rho) + calcLogPostPredictiveBgh(theta_g, theta_h) + calcLogPostPredictiveBgh(theta_h, theta_g);
			}
		}
		// case 2: either one is from new cluster.
		for(int k=0; k<=m_kBar; k++){
			m_cache[k][m_kBar] = m_pNewJoint;
			m_cache[m_kBar][k] = m_pNewJoint;
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
			updateSampleSize(0, 1);
			
			// Update the thetaStar and user info after getting z_ji.
			m_hdpThetaStars[h].updateEdgeCount(0, 1);
			uj.addNeighbor(ui, m_hdpThetaStars[h], 0);
			uj.incHDPThetaStarEdgeSize(m_hdpThetaStars[h], 1);
			m_indicator[j][i] = m_hdpThetaStars[h];
			updateSampleSize(0, 1);
			addConnection(ui, uj, 0);
		} else{
			updateSampleSize(2, 2);
		}
	}

	// Set the sparsity parameter
	public void setRho(double v){
		m_rho = v;
	}
	
	@Override
	public String toString() {
		return String.format("CLRWithMMB[dim:%d,lmDim:%d,M:%d,rho:%.5f,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:%.3f,#Iter:%d,N(%.3f,%.3f)]", m_dim,m_lmDim,m_M, m_rho, m_alpha, m_eta, m_beta, m_eta1, m_numberOfIterations, m_abNuA[0], m_abNuA[1]);
	}
	
	// In the training process, we sample documents first, then sample edges.
	@Override
	public double train(){
		System.out.println(toString());
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		int count = 0;
		
		/**We want to sample documents first without knowing edges,
		 * So we have to rewrite the init function to split init thetastar for docs and edges.**/
		// clear user performance, init cluster assignment, assign each review to one cluster
		init();	
		// assign each edge to one cluster
		initThetaStars4Edges();
		
		checkEdges();
		// Burn in period for doc.
		while(count++ < m_burnIn){
			super.calculate_E_step();
			calculate_E_step_Edge();
			checkEdges();
			lastLikelihood = calculate_M_step();
			lastLikelihood += estRho();
		}
		
		// EM iteration.
		for(int i=0; i<m_numberOfIterations; i++){
			// Cluster assignment, thinning to reduce auto-correlation.
			calculate_E_step();
			calculate_E_step_Edge();

			// Optimize the parameters
			curLikelihood = calculate_M_step();
			curLikelihood += estRho();

			delta = (lastLikelihood - curLikelihood)/curLikelihood;
			
			if (i%m_thinning==0)
				evaluateModel();
			
//			printInfo(i%5==0);//no need to print out the details very often
			System.out.print(String.format("\n[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.3f\n", i, curLikelihood, delta));
			if(Math.abs(delta) < m_converge)
				break;
			lastLikelihood = curLikelihood;
		}

		printClusterInfo();
		evaluateModel(); // we do not want to miss the last sample?!
//		setPersonalizedModel();
		return curLikelihood;
	}
		
	
	private void updateSampleSize(int index, int val){
		if(index <0 || index > m_MNL.length)
			System.err.println("[Error]Wrong index!");
		m_MNL[index] += val;
		if (Utils.sumOfArray(m_MNL) % 1000000==0) {
			System.out.print('.');
			if (Utils.sumOfArray(m_MNL) % 50000000==0)
				System.out.println();
		}
	}
	@Override
	// Override this function since we have different conditions for removing clusters.
	public void updateDocMembership(_HDPAdaptStruct user, _Review r){
		int index = -1;
		_HDPThetaStar curThetaStar = r.getHDPThetaStar();

		// remove the current review from the user side.
		user.incHDPThetaStarMemSize(r.getHDPThetaStar(), -1);
				
		// remove the current review from the theta side.
		// remove the lm stat first before decrease the document count
		curThetaStar.rmLMStat(r.getLMSparse());
		curThetaStar.updateMemCount(-1);
		
		// No data associated with the cluster
		if(curThetaStar.getMemSize() == 0 && curThetaStar.getTotalEdgeSize() == 0) {
			// check if every dim gets 0 count in language model
			LMStatSanityCheck(curThetaStar);
			
			// recycle the gamma
			m_gamma_e += curThetaStar.getGamma();
			curThetaStar.resetGamma();	
			
			// swap the disabled theta to the last for later use
			index = findHDPThetaStar(curThetaStar);
			swapTheta(m_kBar-1, index); // move it back to \theta*
			
			// in case we forget to init some variable, we set it to null
			curThetaStar = null;
//			curThetaStar.disable();
			m_kBar --;
		}
	}
	
	public void updateEdgeMembership(_MMBAdaptStruct ui, _MMBAdaptStruct uj, int e){
		int index = -1;
		_HDPThetaStar thetai = ui.getThetaStar(uj);
		
		// remove the neighbor from user
		ui.rmNeighbor(uj);
		
		// update the edge information inside the user
		ui.incHDPThetaStarEdgeSize(thetai, -1);
		
		// update the edge count for the thetastar
		thetai.updateEdgeCount(e, -1);
		
		// No data associated with the cluster
		if(thetai.getMemSize() == 0 && thetai.getTotalEdgeSize() == 0){			
			// recycle the gamma
			m_gamma_e += thetai.getGamma();
			thetai.resetGamma();
			
			// swap the disabled theta to the last for later use
			index = findHDPThetaStar(thetai);
			if(index == -1)
				System.out.println("Bug");
			swapTheta(m_kBar-1, index); // move it back to \theta*
			
			// in case we forget to init some variable, we set it to null
			thetai = null;
//			thetai.disable();
			m_kBar --;
		}
	}
	
	public void printBMatrix(String filename){
		// Get the B matrix
		int[] eij;
		int[][] B = new int[m_kBar][m_kBar];
		_HDPThetaStar theta1;
		int index1 = 0, index2 = 0;
		for(int i=0; i<m_kBar; i++){
			theta1 = m_hdpThetaStars[i];
			index1 = theta1.getIndex();
			HashMap<_HDPThetaStar, int[]> connectionMap = theta1.getConnectionMap();
			for(_HDPThetaStar theta2: connectionMap.keySet()){
				index2 = theta2.getIndex();
				eij = connectionMap.get(theta2);
				B[index1][index2] = eij[1];
			}
		}
		// print out the B matrix
		try{
			PrintWriter writer = new PrintWriter(new File(filename), "UTF-8");
			for(int i=0; i<B.length; i++){
				int[] row = B[i];
				for(int j=0; j<row.length; j++){
					writer.write(String.format("%d", B[i][j]));
					if(j != row.length - 1){
						writer.write(",");
					}
				}
				writer.write("\n");
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	private void printClusterInfo(){
		try {
			_HDPThetaStar theta;
			PrintWriter writer = new PrintWriter(new File("cluster.txt"));
			for(int k=0; k<m_kBar; k++){
				theta = m_hdpThetaStars[k];
				writer.write(String.format("%d,%d,%d\n", theta.getMemSize(), theta.getEdgeSize(0), theta.getEdgeSize(1)));
			}
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void printEdgeAssignment(String filename){
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			_MMBAdaptStruct u1;
			int eij = 0, index1 = 0, index2 = 0;
			HashMap<_HDPAdaptStruct, _MMBNeighbor> neighbors;
			for(int i=0; i<m_userList.size(); i++){
				u1 = (_MMBAdaptStruct) m_userList.get(i);
				neighbors = u1.getNeighbors();
				for(_HDPAdaptStruct nei: neighbors.keySet()){
					eij = u1.getNeighbors().get(nei).getEdge();
					if(eij == 1){
						index1 = neighbors.get(nei).getHDPThetaStarIndex();
						index2 = ((_MMBAdaptStruct) nei).getNeighbors().get(u1).getHDPThetaStarIndex();
						writer.write(String.format("%s,%s,%d,%d\n", u1.getUserID(), nei.getUserID(), index1, index2));
					}
				}
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void printStat(String filename){
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			for(String key: stat.keySet()){
				writer.write(key+"\n");
				for(int v: stat.get(key)){
					writer.write(v+",");
				}
				writer.write("\n");
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
}
