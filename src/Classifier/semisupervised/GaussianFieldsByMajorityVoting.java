package Classifier.semisupervised;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import utils.Utils;

public class GaussianFieldsByMajorityVoting extends GaussianFieldsByRandomWalk {
	boolean m_simFlag; //This flag is used to determine whether we'll consider similarity or not.
	double m_threshold; //The threshold for threshold-based majority voting.
	boolean m_thresholdFlag; //The flag is used to determine whether we'll use threshold-based majority voting or not.
	
	ArrayList<_RankItem> m_KUU; //It does not have a specific length for nearest unlabeled data.
	ArrayList<_RankItem> m_KUL; //It does not have a specific length for nearest labeled data.
	
	//The default Constructor, majority voting without similarities.
	public GaussianFieldsByMajorityVoting(_Corpus c, String classifier, double C){
		super(c, classifier, C);
		m_simFlag = false;
		m_thresholdFlag = false;
	}	
	
	//The constructor for majority voting with similarities. If we want consider similarity, use setSimilarity() below. 
	public GaussianFieldsByMajorityVoting(_Corpus c, String classifier, double C, double ratio, int k, int kPrime, double alpha, double beta, double delta, double eta, boolean storeGraph){
		super(c, classifier, C, ratio, k, kPrime, alpha, beta, delta, eta, storeGraph);
		m_simFlag = false;
		m_thresholdFlag = false;
	}
	
	//The constructor for threshold based majority voting.
	public GaussianFieldsByMajorityVoting(_Corpus c, String classifier, double C, double ratio, int k, int kPrime, double alpha, double beta, double delta, double eta, boolean storeGraph, double threshold){
		super(c, classifier, C, ratio, k, kPrime, alpha, beta, delta, eta, storeGraph);
		m_simFlag = false;
		m_thresholdFlag = true;
		m_threshold = threshold;
		m_KUU = new ArrayList<_RankItem>();
		m_KUL = new ArrayList<_RankItem>();
	}
	
	public void setSimilarity(){
		m_simFlag = true;
	}
	
	@Override
	public String toString() {
		return String.format("Gaussian Fields by Majority Voting [C:%s, k:%d, k':%d, r:%.3f, alpha:%.3f, beta:%.3f, eta:%.3f]", m_classifier, m_k, m_kPrime, m_labelRatio, m_alpha, m_beta, m_eta);
	}
	
	//The test for random walk algorithm.
	public double test(){
		/***Construct the nearest neighbor graph****/
		constructGraph(m_storeGraph);
			
		if (m_fu_last==null || m_fu_last.length<m_U)
			m_fu_last = new double[m_U]; //otherwise we can reuse the current memory
			
		//initialize fu and fu_last
		for(int i=0; i<m_U; i++) {
			m_fu[i] = m_Y[i];
			m_fu_last[i] = m_Y[i];//random walk starts from multiple learner
		}
			
		/***use random walk to solve matrix inverse***/
		do {
			if (m_storeGraph)
				randomWalkWithGraph();
			else
				randomWalk();			
		} while(updateFu() > m_delta);
			
		/***get some statistics***/
		for(int i = 0; i < m_U; i++){
			for(int j=0; j<m_classNo; j++)
				m_pYSum[j] += Math.exp(-Math.abs(j-m_fu[i]));			
		}
			
		/***evaluate the performance***/
		double acc = 0;
		int pred, ans;
		for(int i = 0; i < m_U; i++) {
			pred = getLabel(m_fu[i]);
			ans = m_testSet.get(i).getYLabel();
			m_TPTable[pred][ans] += 1;
				
			//To calculate the purity of neighbors. The only different with the test in father class.
			if(!m_thresholdFlag)
				calcPurity(m_testSet.get(i));
			else 
				calcThresholdPurity(m_testSet.get(i));
			if (pred != ans) {
				if (m_debugOutput!=null && !m_thresholdFlag)
					debug(m_testSet.get(i));
				else if(m_debugOutput!=null && m_thresholdFlag)
					thresholdDebug(m_testSet.get(i));
			} else {
				if (m_debugOutput!=null && !m_thresholdFlag && Math.random()<0.02)
					debug(m_testSet.get(i));
				else if(m_debugOutput!=null && m_thresholdFlag && Math.random()<0.02)
					thresholdDebug(m_testSet.get(i));
				acc ++;
			}
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
			
		return acc/m_U;
	}
	
	//Take the majority of all neighbors(k+k') as the new label until they converge.
	void randomWalk(){//construct the sparse graph on the fly every time
//		double wL = m_alpha / (m_k + m_beta*m_kPrime), wU = m_beta * wL;
		double similarity = 0;
		/**** Construct the C+scale*\Delta matrix and Y vector. ****/
		for (int i = 0; i < m_U; i++) {
			double[] stat = new double[m_classNo];
			
			/****Construct the top k' unlabeled data for the current data.****/
			for (int j = 0; j < m_U; j++) {
				if (j == i)
					continue;
				if(!m_thresholdFlag)
					m_kUU.add(new _RankItem(j, getCache(i, j)));
				else{
					similarity = getCache(i, j);
					if(similarity > m_threshold)
						m_KUU.add(new _RankItem(j, similarity));
				}
			}
			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++){
				if(!m_thresholdFlag)
					m_kUL.add(new _RankItem(m_U + j, getCache(i, m_U + j)));
				else{
					similarity = getCache(i, m_U + j);
					if(similarity > m_threshold)
						m_KUL.add(new _RankItem(m_U + j, similarity));
				}
			}
			
			if(!m_simFlag && !m_thresholdFlag){
				/**No.1: majority voting without similarity.**/
				for(_RankItem n: m_kUU){
					int labelFu = (int) m_fu[n.m_index]; //Item n's label.
					//We use beta to represent how much we trust the labeled data. The larger, the more trustful.
					stat[labelFu] += (1 - m_beta) * m_eta; 
					int labelSVM = (int) m_Y[n.m_index];//SVM's predition.
					stat[labelSVM] += (1 - m_beta) * (1-m_eta);
				}
				m_kUU.clear();
				
				for(_RankItem n: m_kUL){
					int label = (int) m_Y[n.m_index];//Get the item's label from Y array.
					stat[label] += m_beta;
				}
				m_kUL.clear();
				m_fu[i] = Utils.maxOfArrayIndex(stat);	
			
			} else if(m_simFlag && !m_thresholdFlag){
				/**No.2: majority voting with similarity.****/
				for(_RankItem n: m_kUU){
					int labelFu = (int) m_fu[n.m_index]; //Item n's label.
					//Every unlabeled data get two votes: one from SVM and another from previous votes.
					stat[labelFu] += (1 - m_beta) * m_eta * n.m_value;
					int labelSVM = (int) m_Y[n.m_index];
					stat[labelSVM] += (1 - m_beta) * (1-m_eta) * n.m_value;
				}
				m_kUU.clear();
				
				for(_RankItem n: m_kUL){
					int label = (int) m_Y[n.m_index];//Get the item's label from Y array.
					stat[label] += m_beta * n.m_value;
				}
				m_kUL.clear();
				m_fu[i] = Utils.maxOfArrayIndex(stat);		
				
			} else if(!m_simFlag && m_thresholdFlag){
				/**No.3: threshold based majority voting without similarity.***/
				for(_RankItem n: m_KUU){
					int labelFu = (int) m_fu[n.m_index]; //Item n's label.
					//We use beta to represent how much we trust the labeled data. The larger, the more trustful.
					stat[labelFu] += (1 - m_beta) * m_eta; 
					int labelSVM = (int) m_Y[n.m_index];//SVM's predition.
					stat[labelSVM] += (1 - m_beta) * (1-m_eta);
				}
				m_KUU.clear();
				
				for(_RankItem n: m_KUL){
					int label = (int) m_Y[n.m_index];//Get the item's label from Y array.
					stat[label] += m_beta;
				}
				m_KUL.clear();
				m_fu[i] = Utils.maxOfArrayIndex(stat);	
			} else{
				/**No.4: majority voting based majority voting with similarity.****/
				for(_RankItem n: m_KUU){
					int labelFu = (int) m_fu[n.m_index]; //Item n's label.
					//Every unlabeled data get two votes: one from SVM and another from previous votes.
					stat[labelFu] += (1 - m_beta) * m_eta * n.m_value;
					int labelSVM = (int) m_Y[n.m_index];
					stat[labelSVM] += (1 - m_beta) * (1-m_eta) * n.m_value;
				}
				m_KUU.clear();
				
				for(_RankItem n: m_KUL){
					int label = (int) m_Y[n.m_index];//Get the item's label from Y array.
					stat[label] += m_beta * n.m_value;
				}
				m_KUL.clear();
				m_fu[i] = Utils.maxOfArrayIndex(stat);		
			}
		}
	} 
	

	//It needs different debug functions to get a sense of how this method works.
	protected void debug(_Doc d){
		int id = d.getID();
		double sameL = 0, sameU = 0;
		try {
			m_debugWriter.write("===============================================================================\n");
			m_debugWriter.write(String.format("Label:%d, Predicted:%.4f, SVM:%d\n", d.getYLabel(), m_fu[id], (int)m_Y[id]));

			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++){
				m_kUL.add(new _RankItem(j, getCache(id, m_U + j)));
			}
			for(_RankItem n: m_kUL){
				int index = m_U + n.m_index;
				if(m_Y[index]==d.getYLabel())
					sameL++;
			}
			sameL = sameL / (m_kUL.size() + 0.0001);
			m_debugWriter.write(String.format("Largest: %.4f, Purity: %.4f\n", m_kUL.get(0).m_value, sameL));
			m_kUL.clear();
				
			for (int j = 0; j < m_U; j++) {
				if (j == id)
					continue;
				m_kUU.add(new _RankItem(j, getCache(id, j)));
			}
			for(_RankItem n: m_kUU){
				if(m_testSet.get(n.m_index).getYLabel()==d.getYLabel())
					sameU++;
			}
			sameU = sameU / (m_kUU.size() + 0.0001);
			m_debugWriter.write(String.format("Largest: %.4f, purity: %.4f\n", m_kUU.get(0).m_value, sameU));
			m_kUU.clear();
		} catch (IOException e) {
			e.printStackTrace();
		}
	} 	
	//This method will be called for every test sample.
	protected void calcThresholdPurity(_Doc d){
		double sameL = 0, sameU = 0, similarity = 0;
		int id = d.getID();
		
		/****Construct the top k labeled data for the current data.****/
		for (int j = 0; j < m_L; j++){
			similarity = getCache(id, m_U + j);
			if(similarity > m_threshold){
				m_KUL.add(new _RankItem(j, similarity));
				if(m_Y[m_U + j]==d.getYLabel())
					sameL++;
			}		
		}
		sameL = sameL / (m_KUL.size() + 0.0001);
		m_KUL.clear();
		
		/****Construct the top k' unlabeled data for the current data.****/
		for (int j = 0; j < m_U; j++) {
			if (j == id)
				continue;
			similarity = getCache(id, j);
			if(similarity > m_threshold){
				m_KUU.add(new _RankItem(j, similarity));
				if(m_testSet.get(j).getYLabel()==d.getYLabel())//shall we use true label?
					sameU++;
			}
		}
		sameU = sameU / (m_KUU.size() + 0.0001);
		m_KUU.clear();
		m_debugStat.add(new double[]{sameL, sameU});
	} 
	
	//It needs different debug functions since they have different data structure to store neighbors.
	protected void thresholdDebug(_Doc d){
		int id = d.getID();
		double similarity, sameL = 0, sameU = 0;
		try {
			m_debugWriter.write("===============================================================================\n");
			m_debugWriter.write(String.format("Label:%d, Predicted:%.4f, SVM:%d\n", d.getYLabel(), m_fu[id], (int)m_Y[id]));

			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++){
				similarity = getCache(id, m_U + j);
				if(similarity > m_threshold){
					m_KUL.add(new _RankItem(j, similarity));
					if(m_Y[m_U + j]==d.getYLabel())
						sameL++;
				}
			}
			sameL = sameL / (m_KUL.size() + 0.0001);
			if(m_KUL.size()>0){
				Collections.sort(m_KUL);
				m_debugWriter.write(String.format("Labeled data: %d, largest: %.4f, purity: %.4f\n", m_KUL.size(), m_KUL.get(m_KUL.size()-1).m_value, sameL));
			} else
				m_debugWriter.write("Empty KUL.\n");
			m_KUL.clear();
			
			for (int j = 0; j < m_U; j++) {
				if (j == id)
					continue;
				similarity = getCache(id, j);
				if(similarity > m_threshold){
					m_KUU.add(new _RankItem(j, similarity));
					if(m_testSet.get(j).getYLabel()==d.getYLabel())
						sameU++;
				}
			}
			sameU = sameU / (m_KUU.size() + 0.0001);
			if(m_KUU.size()>0){
				Collections.sort(m_KUU);
				m_debugWriter.write(String.format("Unlabeled data: %d, largest: %.4f, purity: %.4f\n", m_KUU.size(), m_KUU.get(m_KUU.size()-1).m_value, sameU));
			} else
				m_debugWriter.write("Empty KUU.\n");
			m_KUU.clear();
		} catch (IOException e) {
			e.printStackTrace();
		}
	} 	
	
	public int predict(_Doc doc) {
		return -1; //we don't support this in transductive learning
	}
	
	@Override
	public void saveModel(String modelLocation) {
		
	}
}