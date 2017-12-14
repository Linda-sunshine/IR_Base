package Classifier.supervised.modelAdaptation.MMB;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.math3.distribution.BinomialDistribution;

import structures._HDPThetaStar;
import structures._HDPThetaStar._Connection;
import structures._MMBNeighbor;
import structures._Review;
import structures._Review.rType;
import structures._User;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.HDP.CLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP._HDPAdaptStruct;
import cern.jet.random.tdouble.Beta;
import cern.jet.random.tfloat.FloatUniform;
public class CLRWithMMB extends CLRWithHDP {
	// sparsity parameter
	protected double m_rho = 0.001; 
	// As we store all the indicators for all the edges(even edges from background model), we maintain a matrix for indexing.
	protected _HDPThetaStar[][] m_indicator;
	
	// prob for the new cluster in sampling mmb edges.
	protected double[] m_pNew = new double[2]; 
	// parameters used in the gamma function in mmb model, prior of B~beta(a, b), prior of \rho~Beta(c, d)
	protected double[] m_abcd = new double[]{0.1, 0.01, 2, 2}; 
	// Me: total number of edges eij=0;Ne: total number of edges eij=1 from mmb; Le: total number of edges eij=0 from background model.
	protected double[] m_MNL = new double[3];
	// Bernoulli distribution used in deciding whether the edge belongs to mmb or background model.
	protected BinomialDistribution m_bernoulli;
	
	// for debug purpose
	protected HashMap<String, ArrayList<Integer>> stat = new HashMap<>();

	public CLRWithMMB(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel,
			double[] betas) {
		super(classNo, featureSize, featureMap, globalModel, betas);
		initStat();
	} 
	
	public CLRWithMMB(int classNo, int featureSize, String globalModel,
			double[] betas) {
		super(classNo, featureSize, globalModel, betas);
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
	protected void calcProbNew(){
		// if e_ij = 0, p = \rho*b/(a+b)
		m_pNew[0] = Math.log(m_rho) + Math.log(m_abcd[1]) - Math.log(m_abcd[0] + m_abcd[1]);
		// if e_ij = 1, p = (1-\rho*a/(a+b))
		m_pNew[1] = Math.log(1 - m_rho * m_abcd[0] /(m_abcd[0] + m_abcd[1]));
	}

	@Override
	// calculate the group popularity in sampling documents and edges.
	protected double calcGroupPopularity(_HDPAdaptStruct u, int k, double gamma_k){
		_MMBAdaptStruct user= (_MMBAdaptStruct) u;
		return user.getHDPThetaMemSize(m_hdpThetaStars[k]) + m_eta*gamma_k + user.getHDPThetaEdgeSize(m_hdpThetaStars[k]);
	}
	
	/*** p(e=1)=\rho*B_gh, p(e=0)=1-\rho*B_gh
	/* corresponding predictive posterior distribution is:
	/* e=1: \rho*(a+e_1)/(a+b+e_0+e_1)
	/* e=0: 1-\rho*(a+e_1)/(a+b+e_0+e_1) **/
	public double calcLogLikelihoodEMarginal(_HDPThetaStar theta_g, _HDPThetaStar theta_h, int e){
		
		double prob = 0;
		double e_0 = 0, e_1 = 0;
		// some background model may have non-existing cluster
		if(theta_g.isValid() && theta_h.isValid()){
			e_0 = theta_g.getConnectionEdgeCount(theta_h, 0);
			e_1 = theta_g.getConnectionEdgeCount(theta_h, 1);
		} else{
			System.err.println("[Bug]Invalid thetas!");
		}
		prob = Math.log(m_rho) + Math.log(m_abcd[0] + e_1) - Math.log(m_abcd[0] + m_abcd[1] + e_0 + e_1);
		return e == 1 ? prob : Math.log(1 - Math.exp(prob));
	}
	
	/**e=0: \int_{B_gh} \rho*(1-B_{gh})*prior d_{B_{gh}} 
	/* e=1: \int_{B_gh} \rho*B_{gh}*prior d_{B_{gh}} 
	/* prior is Beta(a+e_1, b+e_0)***/
	public double calcLogLikelihoodE(_HDPThetaStar theta_g, _HDPThetaStar theta_h, int e){
		
		double prob = 0;
		double e_0 = 0, e_1 = 0;
		// some background model may have non-existing cluster
		if(theta_g.isValid() && theta_h.isValid()){
			e_0 = theta_g.getConnectionEdgeCount(theta_h, 0);
			e_1 = theta_g.getConnectionEdgeCount(theta_h, 1);
		}
		prob = e == 0 ? Math.log(m_abcd[1] + e_0) : Math.log(m_abcd[0] + e_1);
		prob += Math.log(m_rho) - Math.log(m_abcd[0] + m_abcd[1] + e_0 + e_1);
		return prob;
	}

//	protected void calculate_E_step_Edge(){
//		calcProbNew();
//		// sample z_{i->j}
//		_MMBAdaptStruct ui, uj;
//		int sampleSize = 0, eij = 0;
//
//		for(int i=0; i<m_userList.size(); i++){
//			ui = (_MMBAdaptStruct) m_userList.get(i);
//			for(int j=i+1; j<m_userList.size(); j++){
//				uj = (_MMBAdaptStruct) m_userList.get(j);
//				// print out the process of sampling edges
//				if (++sampleSize%100000==0) {
//					System.out.print('.');
//					if (sampleSize%50000000==0)
//						System.out.println();
//				}
//				// eij from mmb
//				if(ui.hasEdge(uj)){
//					eij = ui.getEdge(uj);
//					// remove the connection for B_gh, i->j \in g, j->i \in h.
//					rmConnection(ui, uj, eij);
//					// update membership from ui->uj, remove the edge
//					updateEdgeMembership(i, j, eij);	
//					// sample new cluster for the edge
//					sampleEdge(i, j, eij);
//					// update membership from uj->ui, remove the edge
//					updateEdgeMembership(j, i, eij);
//					// sample new clusters for the two edges
//					sampleEdge(j, i, eij);
//					// add the new connection for B_g'h', i->j \in g', j->i \in h'
//					addConnection(ui, uj, eij);
//				// edges from background
//				}else{
//					// remove the two edges from background model
//					updateSampleSize(2, -2);
//					sampleEdge(i, j, 0);
//					sampleEdge(j, i, 0);
//					addConnection(ui, uj, 0);
//				}
//			}
//		}
//		sampleC();
//	}
	
	protected void calculate_E_step_Edge(){
		calcProbNew();
		// sample z_{i->j}
		_MMBAdaptStruct ui, uj;
		int sampleSize = 0, eij = 0;

		for(int i=0; i<m_userList.size(); i++){
			ui = (_MMBAdaptStruct) m_userList.get(i);
			for(int j=i+1; j<m_userList.size(); j++){
				uj = (_MMBAdaptStruct) m_userList.get(j);
				// print out the process of sampling edges
				if (++sampleSize%100000==0) {
					System.out.print('.');
					if (sampleSize%50000000==0)
						System.out.println();
				}
				// eij=1
				if(ui.hasEdge(uj) && ui.getEdge(uj) == 1){
					eij = 1;
					// remove the connection for B_gh, i->j \in g, j->i \in h.
					rmConnection(ui, uj, eij);
					// update membership from ui->uj, remove the edge
					updateEdgeMembership(i, j, eij);	
					// sample new cluster for the edge
					sampleEdge(i, j, eij);
					// update membership from uj->ui, remove the edge
					updateEdgeMembership(j, i, eij);
					// sample new clusters for the two edges
					sampleEdge(j, i, eij);
					// add the new connection for B_g'h', i->j \in g', j->i \in h'
					addConnection(ui, uj, eij);
				// edges from background
				}else if(ui.hasEdge(uj) && ui.getEdge(uj) == 0){
					eij = 0;
					// remove the connection for B_gh, i->j \in g, j->i \in h.
					rmConnection(ui, uj, eij);
					// update membership from ui->uj, remove the edge
					updateEdgeMembership(i, j, eij);	
					// update membership from uj->ui, remove the edge
					updateEdgeMembership(j, i, eij);
					// sample cij and cluster assignmetn for the edges
					sampleZeroEdgeJoint(i, j);
				} else{
					// remove the two edges from background model
					updateSampleSize(2, -2);
					sampleZeroEdgeJoint(i, j);
				}
			}
		}
		mmb_0.add((int) m_MNL[0]); mmb_1.add((int) m_MNL[1]);bk_0.add((int) m_MNL[2]);
		System.out.print(String.format("\n[Info]kBar: %d, background prob: %.5f, eij=0(mmb): %.1f, eij=1:%.1f, eij=0(background):%.1f\n", m_kBar, 1-m_rho, m_MNL[0], m_MNL[1],m_MNL[2]));
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
	
	protected void checkMMBEdges(){
		int mmb = 0;
		for(_AdaptStruct u: m_userList){
			_MMBAdaptStruct user = (_MMBAdaptStruct) u;
			for(_HDPThetaStar th: user.getHDPTheta4Edge()){
				mmb += user.getHDPThetaEdgeSize(th);
			}
		}
		if(mmb != m_MNL[0] + m_MNL[1])
			System.out.println("mmb edges is not correct!");
	}

	// Estimate the sparsity parameter.
	// \rho = (M+N+c-1)/(M+N+L+c+d-2)
	public double estRho(){
		m_rho = (m_MNL[0] + m_MNL[1] + m_abcd[2] - 1) / (m_MNL[0] + m_MNL[1] + m_MNL[2] + m_abcd[2] + m_abcd[3] - 2);
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
	// assign all zero edges to mmb
	public void initThetaStars4EdgesMMB(){
		calcProbNew();
		_MMBAdaptStruct ui, uj;
		int sampleSize = 0;
		// add the friends one by one.
		for(int i=0; i< m_userList.size(); i++){
			ui = (_MMBAdaptStruct) m_userList.get(i);
			for(int j=i+1; j<m_userList.size(); j++){
				// print out the process of sampling edges
				if (++sampleSize%100000==0) {
					System.out.print('.');
					if (sampleSize%50000000==0)
						System.out.println();
				}
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
		// assign part of the zero edges to background model
		sampleC();
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
	protected void randomSampleEdges(int i, int j, int e){
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
	ArrayList<Integer> mmb_0 = new ArrayList<Integer>();
	ArrayList<Integer> mmb_1 = new ArrayList<Integer>();
	ArrayList<Integer> bk_0 = new ArrayList<Integer>();
	// we assume all zero edges are from mmb first
	// then utilize bernoulli to sample edges from background model
	protected void sampleC(){
		_MMBAdaptStruct ui, uj;
		double p_mmb_0 = 0, p_bk = 1-m_rho;
		for(int i=0; i<m_userList.size(); i++){
			ui = (_MMBAdaptStruct) m_userList.get(i);
			for(int j=i+1; j<m_userList.size(); j++){
				uj = (_MMBAdaptStruct) m_userList.get(j);
				// eij = 0 from mmb ( should be all zero edges)
				if(ui.hasEdge(uj) && ui.getEdge(uj) == 0){
					// bernoulli distribution to decide whether it is background or mmb
					p_mmb_0 = Math.exp(calcLogLikelihoodE(m_indicator[i][j], m_indicator[j][i], 0));
					m_bernoulli = new BinomialDistribution(1, p_mmb_0/(p_bk + p_mmb_0));
					// the edge belongs to bk
					if(m_bernoulli.sample() == 0){
						rmConnection(ui, uj, 0);
						updateEdgeMembership(i, j, 0);
						updateEdgeMembership(j, i, 0);
						updateSampleSize(2, 2);
					}
					// if the edge belongs to mmb, we just keep it
				}
			}
		}
		mmb_0.add((int) m_MNL[0]); mmb_1.add((int) m_MNL[1]);bk_0.add((int) m_MNL[2]);
		System.out.print(String.format("\n[Info]kBar: %d, background prob: %.5f, eij=0(mmb): %.1f, eij=1:%.1f, eij=0(background):%.1f\n", m_kBar, 1-m_rho, m_MNL[0], m_MNL[1],m_MNL[2]));
	}
	
	// Sample one edge from mmb and e represents the edge value.
	public void sampleEdge(int i,int j, int e){
		_HDPThetaStar theta_h = m_indicator[j][i];
		// if the theta is no longer valid (removed during the sampling of c)
		if(!theta_h.isValid()){
			sampleZeroEdgeJoint(i, j);
		} else{
			sampleMMBEdge(i, j, e);
		}
	}
	
	protected void sampleMMBEdge(int i, int j, int e){
		int k = 0;
		_HDPThetaStar theta_s, theta_h = m_indicator[j][i];
		double likelihood, logNew, gamma_k, logSum = 0;
		_MMBAdaptStruct ui = (_MMBAdaptStruct) m_userList.get(i);
		_MMBAdaptStruct uj = (_MMBAdaptStruct) m_userList.get(j);

		for(k=0; k<m_kBar; k++){			
			//log likelihood of the edge p(e_{ij}, z, B)
			// p(eij|z_{i->j}, z_{j->i}, B)*p(z_{i->j}|\pi_i)*p(z_{j->i|\pj_j})
			theta_s = m_hdpThetaStars[k];
			// we record all the 
			if(!theta_h.isValid())
				System.err.println("Invalid theta!!");
		
			likelihood = calcLogLikelihoodEMarginal(theta_s, theta_h, e);
			if(Double.isInfinite(likelihood))
				System.out.println("Infinite!");
		
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
	
		m_MNL[e]++;
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
	 protected int sampleIn2DimArrayLogSpace(double logSum, double back_prob, double[][] cacheB){
	 
	 	double rnd = FloatUniform.staticNextFloat();
	 	logSum += Math.log(rnd);//we might need a better random number generator
	 		
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
	 		newLogSum = Utils.logSum(newLogSum, cacheB[k/(m_kBar+1)][k%(m_kBar+1)]);
	 			
	 	} while (k<(m_kBar+1)*(m_kBar+1));
	 	return k;
	 }

	// sample eij = 0 from the joint probabilities of cij, zij and zji.
 	public void sampleZeroEdgeJoint(int i, int j){
 		/**we will consider all possible combinations of different memberships.
 		 * 1.cij=0, cji=0, prob: (1-\rho), 1 case
 		 * 2.cij=1, cji=1, known (Bgh, Bhg), prob: \rho(1-Bgh), k(k+1)/2 possible cases
 		 * posterior prob: \rho*(b+e_0)/(a+b+e_0+e_1)
 		 * 3.cij=1, cji=1, unknows (Bgh, Bhg), prob: \rho*b/(a+b), k+1 possible cases 
 		 * In total, we have (k+1)*(k+2)/2+1 possible cases. **/
 		// Step 1: calc prob for different cases of cij, cji.
 		// case 0: background model while the prob is not stored in the two-dim array.
 		double logSum = Math.log(1-m_rho);
 		/**We maintain a matrix for storing probability. As the matrix is 
 		 * symmetric, we only calculate upper-triangle. **/
 		double[][] cacheB = new double[m_kBar+1][m_kBar+1];
 		_MMBAdaptStruct ui = (_MMBAdaptStruct) m_userList.get(i);
 		_MMBAdaptStruct uj = (_MMBAdaptStruct) m_userList.get(j);
 
 		_HDPThetaStar theta_g, theta_h;
 		// case 1: existing thetas.
 		for(int g=0; g<m_kBar; g++){
 			theta_g = m_hdpThetaStars[g];
 			for(int h=g; h<m_kBar; h++){
 				theta_h = m_hdpThetaStars[h];
 				cacheB[g][h] = calcLogLikelihoodE(theta_g, theta_h, 0);
 				cacheB[g][h] += Math.log(theta_g.getGamma()) + Math.log(theta_h.getGamma());
 				logSum = Utils.logSum(logSum, cacheB[g][h]);
 			}
 		}
 		// case 2: either one is from new cluster.
 		// pre-calculate \rho*(b/(a+b))*\gamma_e
 		double pNew = Math.log(m_rho) + Math.log(m_abcd[1]) - Math.log(m_abcd[0] + m_abcd[1]) + Math.log(m_gamma_e);
 		for(int k=0; k<=m_kBar; k++){
 			cacheB[k][m_kBar] = pNew;
 			cacheB[k][m_kBar] += (k == m_kBar) ? Math.log(m_gamma_e) : Math.log(m_hdpThetaStars[k].getGamma());
  			logSum = Utils.logSum(logSum, cacheB[k][m_kBar]);
  		}
 		
  		// Step 2: sample one pair from the prob matrix./*-
 		int k = sampleIn2DimArrayLogSpace(logSum, Math.log(1-m_rho), cacheB);
  		
  		// Step 3: Analyze the sampled cluster results.
  		// case 1: k == -1, sample from the background model;
 		// case 2: k!= 1, sample from mmb model.
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

	// Save the language models of thetaStars
	public void saveClusterLanguageModels(String model){
		PrintWriter writer;
		String filename;
		File dir = new File(model);
		_HDPThetaStar theta;
		double[] lm;
		try{
			if(!dir.exists())
				dir.mkdirs();
			for(int i=0; i<m_kBar; i++){
				theta = m_hdpThetaStars[i];
				filename = String.format("%s/%d.lm", model, theta.getIndex());
				writer = new PrintWriter(new File(filename));
				lm = theta.getLMStat();
				for(int v=0; v<lm.length; v++){
					if(v == lm.length-1)
						writer.write(Double.toString(lm[v]));
					else
						writer.write(lm[v]+",");
				}
				writer.close();
			}
		} catch (IOException e){
			e.printStackTrace();
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

	protected void sanityCheck(){
		checkClusters();
		checkEdges();
		checkMMBEdges();
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
		initThetaStars4EdgesMMB();
		sanityCheck();
		
		// Burn in period for doc.
		while(count++ < m_burnIn){
			super.calculate_E_step();
			calculate_E_step_Edge();
			sanityCheck();

			lastLikelihood = calculate_M_step();
		}
		
		// EM iteration.
		for(int i=0; i<m_numberOfIterations; i++){
			// Cluster assignment, thinning to reduce auto-correlation.
			calculate_E_step();
			calculate_E_step_Edge();

			// Optimize the parameters
			curLikelihood = calculate_M_step();
			delta = (lastLikelihood - curLikelihood)/curLikelihood;
			
			if (i%m_thinning==0)
				evaluateModel();
			
//			printInfo(i%10==0);//no need to print out the details very often
			System.out.print(String.format("\n[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.3f\n", i, curLikelihood, delta));
			if(Math.abs(delta) < m_converge)
				break;
			lastLikelihood = curLikelihood;
		}
		
		evaluateModel(); // we do not want to miss the last sample?!
		return curLikelihood;
	}

	@Override
	public double trainTrace(String tracefile){
		m_numberOfIterations = 50;
		m_burnIn = 10;
		m_thinning = 1;
			
		System.out.println(toString());
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		double likelihoodY = 0, likelihoodX = 0, likelihoodE = 0;
		int count = 0;
			
		// clear user performance, init cluster assignment, assign each review to one cluster
		init();	
		initThetaStars4EdgesMMB();
		sanityCheck();
		
		// Burn in period for doc.
		while(count++ < m_burnIn){
			long start = System.currentTimeMillis()/1000;
			super.calculate_E_step();
			calculate_E_step_Edge();
			sanityCheck();
			long end = System.currentTimeMillis()/1000;
			System.out.println("[Time]The sampling iteration took " + (end-start) + " secs.");
			
			calculate_M_step();
//			estRho();
		}
		
		try{
			PrintWriter writer = new PrintWriter(new File(tracefile));
			// EM iteration.
			for(int i=0; i<m_numberOfIterations; i++){
				
				// Cluster assignment, thinning to reduce auto-correlation.
				long start = System.currentTimeMillis()/1000;
				super.calculate_E_step();
				calculate_E_step_Edge();
				sanityCheck();
				long end = System.currentTimeMillis()/1000;
				System.out.println("[Time]The sampling iteration took " + (end-start) + " secs.");

				likelihoodY = calculate_M_step();
//				estRho();
				likelihoodX = accumulateLikelihoodX();
				likelihoodE = accumulateLikelihoodEMMB();
//				likelihoodE = (m_MNL[0] + m_MNL[1])*Math.log(m_rho) + m_MNL[2]*Math.log(1-m_rho);
				likelihoodE += (m_MNL[2]/2)*Math.log(1-m_rho);
				curLikelihood = likelihoodE + likelihoodX + likelihoodY;
				delta = (lastLikelihood - curLikelihood)/curLikelihood;
				
				if (i%m_thinning==0){
					evaluateModel();
					test();
					for(_AdaptStruct u: m_userList)
						u.getPerfStat().clear();
				}
				
				writer.write(String.format("%.5f\t%.5f\t%.5f\t%.5f\t%d\t%.5f\t%.5f\n", likelihoodY, likelihoodX, likelihoodE, delta, m_kBar, m_perf[0], m_perf[1]));
				System.out.print(String.format("\n[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.3f\n", i, curLikelihood, delta));
				if(Math.abs(delta) < m_converge)
					break;
				lastLikelihood = curLikelihood;
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
		evaluateModel(); // we do not want to miss the last sample?!
		return curLikelihood;
	}
	
	// accumulate the likelihood given by review content
	protected double accumulateLikelihoodX(){
		_MMBAdaptStruct user;
		double likelihoodX = 0;
		for(int i=0; i<m_userList.size(); i++){
			user = (_MMBAdaptStruct) m_userList.get(i);
			if(user.getAdaptationSize() == 0)
				continue;
			for(_Review r: user.getReviews()){
				if (r.getType() == rType.TEST)
					continue;//do not touch testing reviews!
				likelihoodX += calcLogLikelihoodX(r);
			}
		}
		return likelihoodX;
	}
	
	// traverse all the clusters to get the likelihood given by mmb edges
	protected double accumulateLikelihoodEMMB(){
		double likelihoodE = 0;
		_Connection connection;
		int e_0, e_1;
		_HDPThetaStar theta_g, theta_h;
		for(int g=0; g<m_kBar; g++){
			theta_g = m_hdpThetaStars[g];
			for(int h=g; h<m_kBar; h++){
				theta_h = m_hdpThetaStars[h];
				if(!theta_g.hasConnection(theta_h)) continue;
				connection = theta_g.getConnection(theta_h);
				e_1 = connection.getEdge()[1];
				e_0 = connection.getEdge()[0];
				//likelihoodE: m*log(rho*(a+e_1))/(a+b+e_0+e_1))+n*log(rho*(b+e_0))/(a+b+e_0+e_1))
				likelihoodE += (e_0+e_1)*Math.log(m_rho)+e_1*Math.log(m_abcd[0]+e_1)+
						e_0*Math.log(m_abcd[1]+e_0)-(e_0+e_1)*Math.log(m_abcd[0]+m_abcd[1]+e_0+e_1);
//				likelihoodE += e_1*Math.log(m_abcd[0]+e_1)+e_0*Math.log(m_abcd[1]+e_0)
//						-(e_0+e_1)*Math.log(m_abcd[0]+m_abcd[1]+e_0+e_1);
			}
		}
		return likelihoodE;
	}
	
	protected void updateSampleSize(int index, int val){
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
			System.err.println("[Info]Zero cluster detected in updating doc!");
			// check if every dim gets 0 count in language mode
			LMStatSanityCheck(curThetaStar);
			
			// recycle the gamma
			m_gamma_e += curThetaStar.getGamma();
//			curThetaStar.resetGamma();	
			
			// swap the disabled theta to the last for later use
			index = findHDPThetaStar(curThetaStar);
			swapTheta(m_kBar-1, index); // move it back to \theta*
			
			curThetaStar.reset();
			m_kBar --;
		}
	}
	
	public void updateEdgeMembership(int i, int j, int e){
		_MMBAdaptStruct ui = (_MMBAdaptStruct) m_userList.get(i);
		_MMBAdaptStruct uj = (_MMBAdaptStruct) m_userList.get(j);
		
		int index = -1;
		_HDPThetaStar thetai = ui.getThetaStar(uj);
		
		// remove the neighbor from user
		ui.rmNeighbor(uj);
		
		// update the edge information inside the user
		ui.incHDPThetaStarEdgeSize(thetai, -1);
		
		// update the edge count for the thetastar
		thetai.updateEdgeCount(e, -1);
		
		m_MNL[e]--;
		// No data associated with the cluster
		if(thetai.getMemSize() == 0 && thetai.getTotalEdgeSize() == 0){		
			System.err.println("[Info]Zero cluster detected in updating doc!");
			// recycle the gamma
			m_gamma_e += thetai.getGamma();
			
			// swap the disabled theta to the last for later use
			index = findHDPThetaStar(thetai);
			if(index == -1)
				System.out.println("Bug");
			swapTheta(m_kBar-1, index); // move it back to \theta*
			
			thetai.reset();
			m_kBar --;
		}
	}
	
	public void printBMatrix(String filename){
		// Get the B matrix
		int idx = filename.indexOf("txt");
		String zerofile = filename.substring(0, idx-1)+"_0.txt";
		String onefile = filename.substring(0, idx-1)+"_1.txt";

		int[] eij;
		int[][][] B = new int[m_kBar][m_kBar][2];
		_HDPThetaStar theta1;
		int index1 = 0, index2 = 0;
		for(int i=0; i<m_kBar; i++){
			theta1 = m_hdpThetaStars[i];
			index1 = theta1.getIndex();
			HashMap<_HDPThetaStar, _Connection> connectionMap = theta1.getConnectionMap();
			for(_HDPThetaStar theta2: connectionMap.keySet()){
				index2 = theta2.getIndex();
				eij = connectionMap.get(theta2).getEdge();
				B[index1][index2][0] = eij[0];
				B[index1][index2][1] = eij[1];

			}
		}
		try{
			// print out the zero edges in B matrix
			PrintWriter writer = new PrintWriter(new File(zerofile), "UTF-8");
			for(int i=0; i<B.length; i++){
				int[][] row = B[i];
				for(int j=0; j<row.length; j++){
					writer.write(String.format("%d", B[i][j][0]));
					if(j != row.length - 1){
						writer.write("\t");
					}
				}
				writer.write("\n");
			}
			writer.close();
			// print out the one edges in B matrix
			writer = new PrintWriter(new File(onefile), "UTF-8");
			for(int i=0; i<B.length; i++){
				int[][] row = B[i];
				for(int j=0; j<row.length; j++){
					writer.write(String.format("%d", B[i][j][1]));
					if(j != row.length - 1){
						writer.write("\t");
					}
				}
				writer.write("\n");
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void printClusterInfo(String filename){
		try {
			_HDPThetaStar theta;
			PrintWriter writer = new PrintWriter(new File(filename));
			for(int k=0; k<m_kBar; k++){
				theta = m_hdpThetaStars[k];
				writer.write(String.format("%d,%d,%d\n", theta.getMemSize(), theta.getEdgeSize(0), theta.getEdgeSize(1)));
			}
			writer.close();
		} catch (FileNotFoundException e) {
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
	
	public void printEdgeCount(String filename){
		try{
			PrintWriter writer = new PrintWriter(filename);
			for(int v: mmb_0)
				writer.write(v+"\t");
			writer.write("\n");
			for(int v: mmb_1)
				writer.write(v+"\t");
			writer.write("\n");
			for(int v: bk_0)
				writer.write(v+"\t");
			writer.write("\n");
			writer.close();
		} catch (IOException e){
			e.printStackTrace();
		}
	}
}
