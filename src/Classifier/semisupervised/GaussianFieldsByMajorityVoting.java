package Classifier.semisupervised;

import java.util.Arrays;

import structures._Corpus;
import structures._RankItem;
import utils.Utils;

public class GaussianFieldsByMajorityVoting extends GaussianFieldsByRandomWalk {
	boolean m_simFlag; //This flag is used to determine whether we'll consider similarity as weight or not.
	double m_threshold; //The threshold for threshold-based majority voting.
	
	//The default Constructor, majority voting without similarities.
	public GaussianFieldsByMajorityVoting(_Corpus c, String classifier, double C){
		super(c, classifier, C);
		m_simFlag = false;
	}	
	
	//The constructor for majority voting with similarities. If we want consider similarity, use setSimilarity() below. 
	public GaussianFieldsByMajorityVoting(_Corpus c, String classifier, double C, double ratio, int k, int kPrime, double alpha, double beta, double delta, double eta, boolean storeGraph){
		super(c, classifier, C, ratio, k, kPrime, alpha, beta, delta, eta, storeGraph);
		m_simFlag = false;
	}
	
	//The constructor for threshold based majority voting.
	public GaussianFieldsByMajorityVoting(_Corpus c, String classifier, double C, double ratio, int k, int kPrime, double alpha, double beta, double delta, double eta, boolean storeGraph, double threshold){
		super(c, classifier, C, ratio, k, kPrime, alpha, beta, delta, eta, storeGraph);
		m_simFlag = false;
		m_threshold = threshold;
	}
	
	public void setSimilarity(boolean simFlag){
		m_simFlag = simFlag;
	}
	
	@Override
	public String toString() {
		return String.format("Gaussian Fields by Majority Voting [C:%s, k:%d, k':%d, r:%.3f, alpha:%.3f, beta:%.3f, eta:%.3f]", m_classifier, m_k, m_kPrime, m_labelRatio, m_alpha, m_beta, m_eta);
	}
	
	//Take the majority of all neighbors(k+k') as the new label until they converge.
	void randomWalk(){//construct the sparse graph on the fly every time
		double similarity = 0;
		int label;
		
		/**** Construct the C+scale*\Delta matrix and Y vector. ****/
		for (int i = 0; i < m_U; i++) {
			Arrays.fill(m_cProbs, 0);
			
			/****Construct the top k' unlabeled data for the current data.****/
			for (int j = 0; j < m_U; j++) {
				if (j == i)
					continue;
				
				similarity = getCache(i, j);
				if (m_threshold>0) {
					if(similarity > m_threshold)
						m_kUU.add(new _RankItem(j, similarity));
				} else 
					m_kUU.add(new _RankItem(j, similarity));
			}
			
			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++){
				similarity = getCache(i, m_U + j);
				if(m_threshold>0) {
					if(similarity > m_threshold)
						m_kUL.add(new _RankItem(m_U + j, similarity));
				} else 
					m_kUL.add(new _RankItem(m_U + j, similarity));
			}
			
			for(_RankItem n: m_kUU){
				label = getLabel(m_fu[n.m_index]); //Item n's label.
				//We use beta to represent how much we trust the labeled data. The larger, the more trustful.
				m_cProbs[label] += m_simFlag?similarity*m_beta:m_beta; 
				
				label = (int) m_Y[n.m_index];//SVM's predition.
				m_cProbs[label] += m_simFlag?similarity*m_eta:m_eta; 
			}
			m_kUU.clear();
			
			for(_RankItem n: m_kUL){
				label = (int)m_Y[n.m_index];//Get the item's label from Y array.
				m_cProbs[label] += m_simFlag?similarity*m_alpha:m_alpha; 
			}
			m_kUL.clear();
			
			m_fu[i] = Utils.maxOfArrayIndex(m_cProbs);
		}
	} 
}