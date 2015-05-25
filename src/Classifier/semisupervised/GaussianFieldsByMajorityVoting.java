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
	double m_gamma;
	
	public GaussianFieldsByMajorityVoting(_Corpus c, String classifier, double C){
		super(c, classifier, C);
		m_gamma = 0.5;
	}	
	
	//Constructor: given k and kPrime
	public GaussianFieldsByMajorityVoting(_Corpus c, String classifier, double C, double ratio, int k, int kPrime, double alpha, double beta, double delta, double eta, boolean storeGraph){
		super(c, classifier, C, ratio, k, kPrime, alpha, beta, delta, eta, storeGraph);
		m_gamma = 0.5;
	}
	
	public GaussianFieldsByMajorityVoting(_Corpus c, String classifier, double C, double ratio, int k, int kPrime, double alpha, double beta, double delta, double eta, boolean storeGraph, double gamma){
		super(c, classifier, C, ratio, k, kPrime, alpha, beta, delta, eta, storeGraph);
		m_gamma = gamma;
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
				m_kUL.add(new _RankItem(m_U + j, getCache(i, m_U + j), (int)m_Y[m_U + j]));
			
			/**No.1: majority of its neighbors without similarity.**/
			/****Get the sum of k'UU******/
			for(_RankItem n: m_kUU){
				int labelFu = (int) m_fu[n.m_index]; //Item n's label.
				stat[labelFu] += m_eta;
				int labelSVM = (int) m_Y[n.m_index];//SVM's predition.
				stat[labelSVM] += 1-m_eta;
			}
			m_kUU.clear();
			
			/****Get the sum of kUL******/
			for(_RankItem n: m_kUL){
				stat[n.m_label]++;
			}
			m_kUL.clear();
			m_fu[i] = Utils.maxOfArrayIndex(stat);			

//			/**No.2: majority of its neighbors with similarity.****/
//			/****Get the sum of k'UU******/
//			for(_RankItem n: m_kUU){
//				wijSumU += n.m_value; //get the similarity between two nodes.
//				int labelFu = (int) m_fu[n.m_index]; //Item n's label.
//				//Every unlabeled data get two votes: one from SVM and another from previous votes.
//				stat[labelFu] += (1 - m_gamma) * m_eta * n.m_value;
//				int labelSVM = (int) m_Y[n.m_index];
//				stat[labelSVM] += (1 - m_gamma) * (1-m_eta) * n.m_value;
////				fSumU += n.m_value * labelFu;
//			}
//			m_kUU.clear();
//			
//			/****Get the sum of kUL******/
//			for(_RankItem n: m_kUL){
////				wijSumL += n.m_value;
//				stat[n.m_label] += m_gamma * n.m_value;
////				fSumL += n.m_value * m_Y[n.m_index];
//			}
//			m_kUL.clear();
//			m_fu[i] = Utils.maxOfArrayIndex(stat);		
			
			if(Double.isNaN(m_fu[i]))
				System.out.println("NaN detected!!!");
		}
	} 
}