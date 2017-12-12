package Classifier.supervised.modelAdaptation.MMB;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;

import org.apache.commons.math3.distribution.BinomialDistribution;

import structures._HDPThetaStar;
import structures._HDPThetaStar._Connection;
import Classifier.supervised.modelAdaptation._AdaptStruct;

public class CLRWithMMBMLE extends CLRWithMMB {

	public CLRWithMMBMLE(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel,
			double[] betas) {
		super(classNo, featureSize, featureMap, globalModel, betas);
	} 

	public CLRWithMMBMLE(int classNo, int featureSize, String globalModel,
			double[] betas) {
		super(classNo, featureSize, globalModel, betas);
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

			// MLE of sentiment model, \rho and B matrix
			lastLikelihood = calculate_M_step();
			estRho();
			estB();
		}
		
		// EM iteration.
		for(int i=0; i<m_numberOfIterations; i++){
			// Cluster assignment, thinning to reduce auto-correlation.
			calculate_E_step();
			calculate_E_step_Edge();

			curLikelihood = calculate_M_step();
			estRho();
			estB();
			delta = (lastLikelihood - curLikelihood)/curLikelihood;
			
			if (i%m_thinning==0)
				evaluateModel();
			
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

			// MLE of sentiment model, \rho and B matrix
			calculate_M_step();
			estRho();
			estB();
		}
		
		try{
			PrintWriter writer = new PrintWriter(new File(tracefile));
			// EM iteration.
			for(int i=0; i<m_numberOfIterations; i++){
				// Cluster assignment, thinning to reduce auto-correlation.
				calculate_E_step();
				calculate_E_step_Edge();

				likelihoodY = calculate_M_step();
				estRho();
				likelihoodX = accumulateLikelihoodX();
				likelihoodE = estB();
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
	
	// MLE of B_gh = (m+a-1)/(m+n+a+b-2), parameter is prior
	public double calcBgh(int[] e, double a, double b){
		return (e[1] + a -1) / (e[0]  + e[1] + a + b - 2);
	}
	
	// Utilize MLE to estimate B
	protected double estB(){
		double likelihoodE = 0, B_gh = 0;
		_HDPThetaStar theta_g;
		_Connection connection;
		int e_1 = 0, e_0 = 0;
		for(int g=0; g<m_kBar; g++){
			theta_g = m_hdpThetaStars[g];
			// calculate B_gh one by one
			HashMap<_HDPThetaStar, _Connection> connectionMap = theta_g.getConnectionMap();
			for(_HDPThetaStar theta_h: connectionMap.keySet()){
				// the probability is already calculated
				if(theta_h.getIndex() < g) continue;
				connection = connectionMap.get(theta_h);
				B_gh = calcBgh(connection.getEdge(), m_abcd[0], m_abcd[1]);
				
				// B_gh is symmetric
				connection.setProb(B_gh);
				theta_h.getConnection(theta_g).setProb(B_gh);
				
				e_0 = connection.getEdgeCount(0);
				e_1 = connection.getEdgeCount(1);
				
				// calculate likelihood = (m+n)log\rho + mlogB_gh + nlog(1-B_gh)
				likelihoodE += (e_0+e_1)*Math.log(m_rho) + e_1*Math.log(B_gh)+e_1*Math.log(1-B_gh);
			}
		}
		return likelihoodE;
	}
	
	@Override
	// we use B estimated by MLE to calculate the likelihood
	public double calcLogLikelihoodE(_HDPThetaStar theta_g, _HDPThetaStar theta_h, int e){
		
		double prob = 0;
		_Connection connection;
		// some background model may have non-existing cluster
		if(theta_g.isValid() && theta_h.isValid()){
			connection = theta_g.getConnection(theta_h);
			prob = Math.log(m_rho) + Math.log(connection.getProb());
			return e == 1 ? prob : Math.log(1 - Math.exp(prob));
		} else{
			System.err.println("[Bug]Invalid thetas!");
			return 0;
		}
	}
	
	@Override
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
					p_mmb_0 = Math.exp(calcLogLikelihoodEinC(m_indicator[i][j], m_indicator[j][i], 0));
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
	
	/**e=0: \int_{B_gh} \rho*(1-B_{gh})*prior d_{B_{gh}} 
	/* e=1: \int_{B_gh} \rho*B_{gh}*prior d_{B_{gh}} 
	/* prior is Beta(a+e_1, b+e_0)***/
	@Override
	public double calcLogLikelihoodEinC(_HDPThetaStar theta_g, _HDPThetaStar theta_h, int e){
		
		_Connection connection;
		// some background model may have non-existing cluster
		if(theta_g.isValid() && theta_h.isValid()){
			connection = theta_g.getConnection(theta_h);
			if(e == 0)
				return Math.log(m_rho) + Math.log(1-connection.getProb());
			else {
				return Math.log(m_rho) + Math.log(connection.getProb());
			}
		} else{
			System.err.println("[Bug]Invalid thetas!");
			return 0;
		}
	}
	

}
