package Classifier.semisupervised;

import java.io.IOException;
import java.util.Arrays;

import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import utils.Utils;

public class GaussianFieldsByRandomWalk extends GaussianFields {
	double m_difference; //The difference between the previous labels and current labels.
	double m_eta; //The parameter used in random walk. 
	double[] m_fu_last; // result from last round of random walk
	
	double m_delta; // convergence criterion for random walk
	boolean m_weightedAvg; // random walk strategy: True - weighted average; False - majority vote
	boolean m_simFlag; //This flag is used to determine whether we'll consider similarity as weight or not.
	
	//Default constructor without any default parameters.
	public GaussianFieldsByRandomWalk(_Corpus c, String classifier, double C){
		super(c, classifier, C);
		
		m_eta = 0.1;
		m_labelRatio = 0.1;
		m_delta = 1e-5;
		m_weightedAvg = true;
		m_simFlag = false;
	}	
	
	//Constructor: given k and kPrime
	public GaussianFieldsByRandomWalk(_Corpus c, String classifier, double C,
			double ratio, int k, int kPrime, double alhpa, double beta, double delta, double eta, boolean weightedAvg){
		super(c, classifier, C, ratio, k, kPrime);
		
		m_alpha = alhpa;
		m_beta = beta;
		m_delta = delta;
		m_eta = eta;
		m_weightedAvg = weightedAvg;
		m_simFlag = false;
	}
	
	@Override
	public String toString() {
		if (m_weightedAvg)
			return String.format("Gaussian Fields by random walk [C:%s, k:%d, k':%d, r:%.3f, alpha:%.3f, beta:%.3f, eta:%.3f, discount:%.3f]", m_classifier, m_k, m_kPrime, m_labelRatio, m_alpha, m_beta, m_eta, m_discount);
		else
			return String.format("Random walk by majority vote[C:%s, k:%d, k':%d, r:%.3f, alpha:%.3f, beta:%.3f, eta:%.3f, discount:%.3f, simWeight:%s]", m_classifier, m_k, m_kPrime, m_labelRatio, m_alpha, m_beta, m_eta, m_discount, m_simFlag);
	}
	
	public void setSimilarity(boolean simFlag){
		m_simFlag = simFlag;
	}
	
	//The random walk algorithm to generate new labels for unlabeled data.
	//Take the average of all neighbors as the new label until they converge.
	void randomWalkByWeightedSum(){//construct the sparse graph on the fly every time
		//double wL = m_alpha / (m_k + m_beta*m_kPrime), wU = m_beta * wL;
		double wL = m_alpha, wU = m_beta;
		
		/**** Construct the C+scale*\Delta matrix and Y vector. ****/
		for (int i = 0; i < m_U; i++) {
			double wijSumU = 0, wijSumL = 0;
			double fSumU = 0, fSumL = 0;
			
			/****Construct the top k' unlabeled data for the current data.****/
			for (int j = 0; j < m_U; j++) {
				if (j == i)
					continue;
				m_kUU.add(new _RankItem(j, getCache(i, j)));
			}
			
			/****Get the sum of k'UU******/
			for(_RankItem n: m_kUU){
				wijSumU += n.m_value; //get the similarity between two nodes.
//				fSumU += n.m_value * m_fu_last[n.m_index];
				fSumU += n.m_value * m_fu[n.m_index];
			}
			m_kUU.clear();
			
			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++)
				m_kUL.add(new _RankItem(m_U + j, getCache(i, m_U + j)));
			
			/****Get the sum of kUL******/
			for(_RankItem n: m_kUL){
				wijSumL += n.m_value;
				fSumL += n.m_value * m_Y[n.m_index];
			}
			m_kUL.clear();
			
			m_fu[i] = m_eta * (fSumL*wL + fSumU*wU) / (wijSumL*wL + wijSumU*wU) + (1-m_eta) * m_Y[i];
			if (Double.isNaN(m_fu[i])) {
				System.out.format("Encounter NaN in random walk!\nfSumL: %.3f, fSumU: %.3f, wijSumL: %.3f, wijSumU: %.3f\n", fSumL, fSumU, wijSumL, wijSumU);
				System.exit(-1);				
			}
		}
	}
	
	//Take the majority of all neighbors(k+k') as the new label until they converge.
	void randomWalkByMajorityVote(){//construct the sparse graph on the fly every time
		double similarity = 0;
		int label;
//		double wL = m_eta * m_alpha / (m_k + m_beta*m_kPrime), wU = m_eta * m_beta * wL;
		double wL = m_eta*m_alpha, wU = m_eta*m_beta;
		
		/**** Construct the C+scale*\Delta matrix and Y vector. ****/
		for (int i = 0; i < m_U; i++) {
			Arrays.fill(m_cProbs, 0);
			
			/****Construct the top k' unlabeled data for the current data.****/
			for (int j = 0; j < m_U; j++) {
				if (j == i)
					continue;				
				m_kUU.add(new _RankItem(j, getCache(i, j)));
			}
			
			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++)
				m_kUL.add(new _RankItem(m_U + j, getCache(i, m_U + j)));
			
			for(_RankItem n: m_kUU){
				label = getLabel(m_fu[n.m_index]); //Item n's label.
				similarity = n.m_value;
				//We use beta to represent how much we trust the labeled data. The larger, the more trustful.
				m_cProbs[label] += m_simFlag?similarity*wU:wU; 
			}
			m_kUU.clear();
			
			for(_RankItem n: m_kUL){
				label = (int)m_Y[n.m_index];//Get the item's label from Y array.
				similarity = n.m_value;
				
				m_cProbs[label] += m_simFlag?similarity*wL:wL; 
			}
			m_kUL.clear();
			
			label = (int) m_Y[i];//SVM's predition.
			m_cProbs[label] += 1-m_eta; 
			
			m_fu[i] = Utils.maxOfArrayIndex(m_cProbs);
		}
	} 
	
	double updateFu() {
		m_difference = 0;
		for(int i = 0; i < m_U; i++){
			m_difference += Math.abs(m_fu[i] - m_fu_last[i]);
			m_fu_last[i] = m_fu[i];//record the last result
		}
		return m_difference/m_U;
	}
	
	//The test for random walk algorithm.
	public double test(){
		/***Construct the nearest neighbor graph****/
		constructGraph(false);
		
		if (m_fu_last==null || m_fu_last.length<m_U)
			m_fu_last = new double[m_U]; //otherwise we can reuse the current memory
		
		//initialize fu and fu_last
		for(int i=0; i<m_U; i++) {
			m_fu[i] = m_Y[i];
			m_fu_last[i] = m_Y[i];//random walk starts from multiple learner
		}
		
		/***use random walk to solve matrix inverse***/
		System.out.println("Random walk starts:");
		int iter = 0;
		double diff = 0;
		do {
			if (m_weightedAvg)
				randomWalkByWeightedSum();	
			else
				randomWalkByMajorityVote();
			
			diff = updateFu();
			System.out.format("Iteration %d, converge to %.3f...\n", ++iter, diff);
		} while(diff > m_delta);
		
		/***get some statistics***/
		for(int i = 0; i < m_U; i++){
			for(int j=0; j<m_classNo; j++)
				m_pYSum[j] += Math.exp(-Math.abs(j-m_fu[i]));			
		}
		
		//temporary injected code
		SimilarityCheck();
		
		/***evaluate the performance***/
		double acc = 0;
		int pred, ans;
		for(int i = 0; i < m_U; i++) {
			//pred = getLabel(m_fu[i]);
			pred = getLabel(m_fu[i]);
			ans = m_testSet.get(i).getYLabel();
			m_TPTable[pred][ans] += 1;
			
			if (pred != ans) {
				if (m_debugOutput!=null)
					debug(m_testSet.get(i));
			} else {
				if (m_debugOutput!=null && Math.random()<0.02)
					debug(m_testSet.get(i));
				acc ++;
			}
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
		
		return acc/m_U;
	}
	
	@Override
	protected void debug(_Doc d){
		int id = d.getID();
		_RankItem item;
		double sim, wijSumU=0, wijSumL=0;
		double fSumU = 0, fSumL = 0;
		
		try {
			m_debugWriter.write(String.format("%d\t%.4f(%d*,%d)\t%d\n", d.getYLabel(), m_fu[id], getLabel(m_fu[id]), getLabel3(m_fu[id]), (int)m_Y[id]));
		
			double mean = 0, sd = 0;
			//find top five labeled
			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++)
				m_kUL.add(new _RankItem(j + m_U, getCache(id, m_U + j)));
			
			/****Get the sum of kUL******/
			for(_RankItem n: m_kUL) {
				wijSumL += n.m_value; //get the similarity between two nodes.
				fSumL += n.m_value*m_Y[n.m_index];
				sd += n.m_value * n.m_value;
			}
			
			mean = wijSumL / m_k;
			sd = Math.sqrt(sd/m_k - mean*mean);
			
			/****Get the top 5 elements from kUL******/
			for(int k=0; k<5; k++){
				item = m_kUL.get(k);
				sim = item.m_value/wijSumL;
				
				if (k==0)
					m_debugWriter.write(String.format("L(%.2f)\t[%d:%.4f, ", fSumL/wijSumL, (int)m_Y[item.m_index], sim));
				else if (k==4)
					m_debugWriter.write(String.format("%d:%.4f]\t%.3f\t%.3f\n", (int)m_Y[item.m_index], sim, mean, sd));
				else
					m_debugWriter.write(String.format("%d:%.4f, ", (int)m_Y[item.m_index], sim));
			}
			m_kUL.clear();
			mean = 0;
			sd = 0;
			
			//find top five unlabeled
			/****Construct the top k' unlabeled data for the current data.****/
			for (int j = 0; j < m_U; j++) {
				if (j == id)
					continue;
				m_kUU.add(new _RankItem(j, getCache(id, j)));
			}
			
			/****Get the sum of k'UU******/
			for(_RankItem n: m_kUU) {
				wijSumU += n.m_value; //get the similarity between two nodes.
				fSumU += n.m_value*m_fu[n.m_index];
				sd += n.m_value * n.m_value;
			}
			
			mean = wijSumU / m_kPrime;
			sd = Math.sqrt(sd/m_kPrime - mean*mean);
			
			/****Get the top 5 elements from k'UU******/
			for(int k=0; k<5; k++){
				item = m_kUU.get(k);
				sim = item.m_value/wijSumU;
				
				if (k==0)
					m_debugWriter.write(String.format("U(%.2f)\t[%.2f:%.4f, ", fSumU/wijSumU, m_fu[item.m_index], sim));
				else if (k==4)
					m_debugWriter.write(String.format("%.2f:%.4f]\t%.3f\t%.3f\n", m_fu[item.m_index], sim, mean, sd));
				else
					m_debugWriter.write(String.format("%.2f:%.4f, ", m_fu[item.m_index], sim));
			}
			m_kUU.clear();
			m_debugWriter.write("\n");		
		} catch (IOException e) {
			e.printStackTrace();
		}
	} 
}
