package Classifier.semisupervised;

import java.io.IOException;

import structures._Corpus;
import structures._Doc;
import structures._RankItem;

public class GaussianFieldsByRandomWalk extends GaussianFields {
	double m_difference; //The difference between the previous labels and current labels.
	double m_eta; //The parameter used in random walk. 
	double[] m_fu_last; // result from last round of random walk
	
	double m_delta; // convergence criterion for random walk
	boolean m_storeGraph; // shall we precompute and store the graph
	
	//Default constructor without any default parameters.
	public GaussianFieldsByRandomWalk(_Corpus c, String classifier, double C){
		super(c, classifier, C);
		
		m_eta = 0.1;
		m_labelRatio = 0.1;
		m_delta = 1e-5;
		m_storeGraph = false;
	}	
	
	//Constructor: given k and kPrime
	public GaussianFieldsByRandomWalk(_Corpus c, String classifier, double C,
			double ratio, int k, int kPrime, double alhpa, double beta, double delta, double eta, boolean storeGraph){
		super(c, classifier, C, ratio, k, kPrime);
		
		m_alpha = alhpa;
		m_beta = beta;
		m_delta = delta;
		m_eta = eta;
		m_storeGraph = storeGraph;
	}
	
	@Override
	public String toString() {
		return String.format("Gaussian Fields by random walk [C:%s, k:%d, k':%d, r:%.3f, alpha:%.3f, beta:%.3f, eta:%.3f, discount:%.3f]", m_classifier, m_k, m_kPrime, m_labelRatio, m_alpha, m_beta, m_eta, m_discount);
	}
	
	//The random walk algorithm to generate new labels for unlabeled data.
	//Take the average of all neighbors as the new label until they converge.
	void randomWalk(){//construct the sparse graph on the fly every time
		double wL = m_alpha / (m_k + m_beta*m_kPrime), wU = m_beta * wL;
		
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
		}
	}
	
	//based on the precomputed sparse graph
	void randomWalkWithGraph(){
		double wij, wL = m_alpha / (m_k + m_beta*m_kPrime), wU = m_beta * wL;
		
		/**** Construct the C+scale*\Delta matrix and Y vector. ****/
		for (int i = 0; i < m_U; i++) {
			double wijSumU = 0, wijSumL = 0;
			double fSumU = 0, fSumL = 0;
			int j = 0;
			
			/****Get the sum of k'UU******/
			for (; j < m_U; j++) {
				if (j == i) 
					continue;
				wij = m_graph.getQuick(i, j); //get the similarity between two nodes.
				if (wij == 0)
					continue;
				
				wijSumU += wij;
				//fSumU += wij * m_fu_last[j];//use the old results
				fSumU += wij * m_fu[j];//use the updated results immediately
			}
			
			/****Get the sum of kUL******/
			for (; j<m_U+m_L; j++) {
				wij = m_graph.getQuick(i, j); //get the similarity between two nodes.
				if (wij == 0)
					continue;
				
				wijSumL += wij;
				fSumL += wij * m_Y[j];
			}
			
			m_fu[i] = m_eta * (fSumL*wL + fSumU*wU) / (wijSumL*wL + wijSumU*wU) + (1-m_eta) * m_Y[i];
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
