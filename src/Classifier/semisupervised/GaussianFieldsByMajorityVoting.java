package Classifier.semisupervised;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;

import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;

public class GaussianFieldsByMajorityVoting extends GaussianFieldsByRandomWalk {

	double m_difference; //The difference between the previous labels and current labels.
	double m_eta; //The parameter used in random walk. 
	double[] m_fu_last; // result from last round of random walk
	
	double m_delta; // convergence criterion for random walk
	boolean m_storeGraph; // shall we precompute and store the graph
	
	public GaussianFieldsByMajorityVoting(_Corpus c, String classifier, double C){
		super(c, classifier, C);
	}	
	
	//Constructor: given k and kPrime
	public GaussianFieldsByMajorityVoting(_Corpus c, String classifier, double C, double ratio, int k, int kPrime, double alpha, double beta, double delta, double eta, boolean storeGraph){
		super(c, classifier, C, ratio, k, kPrime, alpha, beta, delta, eta, storeGraph);
	}

	@Override
	public String toString() {
		return String.format("Gaussian Fields by Majority Voting [C:%s, k:%d, k':%d, r:%.3f, alpha:%.3f, beta:%.3f, eta:%.3f]", m_classifier, m_k, m_kPrime, m_labelRatio, m_alpha, m_beta, m_eta);
	}
	
	//The random walk algorithm to generate new labels for unlabeled data.
	//Take the majority of all neighbors as the new label until they converge.
	void randomWalk(){//construct the sparse graph on the fly every time
		double wL = m_alpha / (m_k + m_beta*m_kPrime), wU = m_beta * wL;
		
		/**** Construct the C+scale*\Delta matrix and Y vector. ****/
		for (int i = 0; i < m_U; i++) {
			double[] stat = new double[m_classNo];
			double wijSumU = 0, wijSumL = 0;
			double fSumU = 0, fSumL = 0;
			
			/****Construct the top k' unlabeled data for the current data.****/
			for (int j = 0; j < m_U; j++) {
				if (j == i)
					continue;
				m_kUU.add(new _RankItem(j, getCache(i, j)));
			}
			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++)
				m_kUL.add(new _RankItem(m_U + j, getCache(i, m_U + j)));
			
			/****Get the sum of k'UU******/
			for(_RankItem n: m_kUU){
				wijSumU += n.m_value; //get the similarity between two nodes.
				int labelFu = (int) m_fu[n.m_index]; //Get its current label.

//				stat[label]++; 
				//Every unlabeled data get two votes from SVM and previous votes.
				stat[labelFu] += m_eta * n.m_value;
				int labelSVM = (int) m_Y[n.m_index];
				stat[labelSVM] += (1-m_eta) * n.m_value;
				//stat[label] += label * n.m_value;
				
				fSumU += n.m_value * m_fu[n.m_index];
//				stat[label] += n.m_value * m_fu[n.m_index];
			}
			m_kUU.clear();
			
			/****Get the sum of kUL******/
			for(_RankItem n: m_kUL){
				wijSumL += n.m_value;
				stat[n.m_label]++;
				int label = (int) m_Y[n.m_index];
				stat[label] += n.m_value;
				fSumL += n.m_value * m_Y[n.m_index];
				stat[n.m_label] += n.m_value * m_Y[n.m_index];
			}
			m_kUL.clear();
			
			if(wijSumL!=0 || wijSumU!=0){
				//Different ways of getting m_fu.
				m_fu[i] = m_eta * (fSumL*wL + fSumU*wU) / (wijSumL*wL + wijSumU*wU) + (1-m_eta) * m_Y[i];
				//This is just the majority of the both labeled and unlabeled data.
//				m_fu[i] = Utils.maxOfArrayIndex(stat);
			}
			if(Double.isNaN(m_fu[i]))
				System.out.println("NaN detected!!!");
		}
	} 
}