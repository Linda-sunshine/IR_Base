package Classifier.semisupervised;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;

public class GaussianFieldsByMajorityVoting extends GaussianFieldsByRandomWalk {
	boolean m_simFlag;
	double m_threshold;
	
	ArrayList<_RankItem> m_KUU; //It does not have a specific length for nearest unlabeled data.
	ArrayList<_RankItem> m_KUL; //It does not have a specific length for nearest labeled data.
	public GaussianFieldsByMajorityVoting(_Corpus c, String classifier, double C){
		super(c, classifier, C);
		m_simFlag = false;
		m_KUU = new ArrayList<_RankItem>();
		m_KUL = new ArrayList<_RankItem>();
		m_threshold = 3.0;
	}	
	
	//Constructor: given k and kPrime
	public GaussianFieldsByMajorityVoting(_Corpus c, String classifier, double C, double ratio, int k, int kPrime, double alpha, double beta, double delta, double eta, boolean storeGraph){
		super(c, classifier, C, ratio, k, kPrime, alpha, beta, delta, eta, storeGraph);
		m_simFlag = false;
		m_KUU = new ArrayList<_RankItem>();
		m_KUL = new ArrayList<_RankItem>();
		m_threshold = 3.75;
	}
	
	//Constructor: given k and kPrime
	public GaussianFieldsByMajorityVoting(_Corpus c, String classifier, double C, double ratio, int k, int kPrime, double alpha, double beta, double delta, double eta, boolean storeGraph, boolean simFlag){
		super(c, classifier, C, ratio, k, kPrime, alpha, beta, delta, eta, storeGraph);
		m_simFlag = simFlag;
		m_KUU = new ArrayList<_RankItem>();
		m_KUL = new ArrayList<_RankItem>();
		m_threshold = 3.0;
	}
	
	@Override
	public String toString() {
		return String.format("Gaussian Fields by Majority Voting [C:%s, k:%d, k':%d, r:%.3f, alpha:%.3f, beta:%.3f, eta:%.3f]", m_classifier, m_k, m_kPrime, m_labelRatio, m_alpha, m_beta, m_eta);
	}
	
	//The random walk algorithm to generate new labels for unlabeled data.
	//Take the majority of all neighbors as the new label until they converge.
//	void randomWalk(){//construct the sparse graph on the fly every time
////		double wL = m_alpha / (m_k + m_beta*m_kPrime), wU = m_beta * wL;
//		
//		/**** Construct the C+scale*\Delta matrix and Y vector. ****/
//		for (int i = 0; i < m_U; i++) {
//			double[] stat = new double[m_classNo];
//			
//			/****Construct the top k' unlabeled data for the current data.****/
//			for (int j = 0; j < m_U; j++) {
//				if (j == i)
//					continue;
//				m_kUU.add(new _RankItem(j, getCache(i, j)));
//			}
//			/****Construct the top k labeled data for the current data.****/
//			for (int j = 0; j < m_L; j++)
//				m_kUL.add(new _RankItem(m_U + j, getCache(i, m_U + j)));
//			
//			if(!m_simFlag){
//				/**No.1: majority of its neighbors without similarity.**/
//				/****Get the sum of k'UU******/
//				for(_RankItem n: m_kUU){
//					int labelFu = (int) m_fu[n.m_index]; //Item n's label.
//					//We use beta to represent how much we trust the labeled data. The larger, the more trustful.
//					stat[labelFu] += (1 - m_beta) * m_eta; 
//					int labelSVM = (int) m_Y[n.m_index];//SVM's predition.
//					stat[labelSVM] += (1 - m_beta) * (1-m_eta);
//				}
//				m_kUU.clear();
//				
//				/****Get the sum of kUL******/
//				for(_RankItem n: m_kUL){
//					int label = (int) m_Y[n.m_index];//Get the item's label from Y array.
//					stat[label] += m_beta;
//				}
//				m_kUL.clear();
////				stat[1] /= 4.0;
//				m_fu[i] = Utils.maxOfArrayIndex(stat);	
//			
//			} else{
//				/**No.2: majority of its neighbors with similarity.****/
//				/****Get the sum of k'UU******/
//				for(_RankItem n: m_kUU){
//					int labelFu = (int) m_fu[n.m_index]; //Item n's label.
//					//Every unlabeled data get two votes: one from SVM and another from previous votes.
//					stat[labelFu] += (1 - m_beta) * m_eta * n.m_value;
//					int labelSVM = (int) m_Y[n.m_index];
//					stat[labelSVM] += (1 - m_beta) * (1-m_eta) * n.m_value;
//				}
//				m_kUU.clear();
//				
//				/****Get the sum of kUL******/
//				for(_RankItem n: m_kUL){
//					int label = (int) m_Y[n.m_index];//Get the item's label from Y array.
//					stat[label] += m_beta * n.m_value;
//				}
//				m_kUL.clear();
//				
//				m_fu[i] = Utils.maxOfArrayIndex(stat);		
//				
//				if(Double.isNaN(m_fu[i]))
//					System.out.println("NaN detected!!!");
//			}
//		}
//	} 
	
	void randomWalk(){//construct the sparse graph on the fly every time
		
		for (int i = 0; i < m_U; i++) {
			double similarity = 0;
			double[] stat = new double[m_classNo];
			
			/****Construct the top k' unlabeled data which pass the threshold.****/
			for (int j = 0; j < m_U; j++) {
				if (j == i)
					continue;
				similarity = getCache(i, m_U + j);
				if(similarity > m_threshold)
					m_KUU.add(new _RankItem(j, similarity));
//				m_kUU.add(new _RankItem(j, getCache(i, j)));
			}
			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++){
				similarity = getCache(i, m_U + j);
				if(similarity > m_threshold)
					m_KUL.add(new _RankItem(m_U + j, similarity));
			}
			
			if(!m_simFlag){
				/**No.1: majority of its neighbors without similarity.**/
				/****Get the sum of k'UU******/
				for(_RankItem n: m_KUU){
					int labelFu = (int) m_fu[n.m_index]; //Item n's label.
					//We use beta to represent how much we trust the labeled data. The larger, the more trustful.
					stat[labelFu] += (1 - m_beta) * m_eta; 
					int labelSVM = (int) m_Y[n.m_index];//SVM's predition.
					stat[labelSVM] += (1 - m_beta) * (1-m_eta);
				}
				m_KUU.clear();
				
				/****Get the sum of KUL******/
				for(_RankItem n: m_KUL){
					int label = (int) m_Y[n.m_index];//Get the item's label from Y array.
					stat[label] += m_beta;
				}
				m_KUL.clear();
//				stat[1] /= 4.0;
				m_fu[i] = Utils.maxOfArrayIndex(stat);	
			
			} else{
				/**No.2: majority of its neighbors with similarity.****/
				/****Get the sum of k'UU******/
				for(_RankItem n: m_KUU){
					int labelFu = (int) m_fu[n.m_index]; //Item n's label.
					//Every unlabeled data get two votes: one from SVM and another from previous votes.
					stat[labelFu] += (1 - m_beta) * m_eta * n.m_value;
					int labelSVM = (int) m_Y[n.m_index];
					stat[labelSVM] += (1 - m_beta) * (1-m_eta) * n.m_value;
				}
				m_KUU.clear();
				
				/****Get the sum of KUL******/
				for(_RankItem n: m_KUL){
					int label = (int) m_Y[n.m_index];//Get the item's label from Y array.
					stat[label] += m_beta * n.m_value;
				}
				m_KUL.clear();
				
				m_fu[i] = Utils.maxOfArrayIndex(stat);		
				
				if(Double.isNaN(m_fu[i]))
					System.out.println("NaN detected!!!");
			}
		}
	} 

	//It needs different debug functions to get a sense of how this method works.
	protected void debug(_Doc d){
		int id = d.getID();
		double similarity, sameL = 0, sameU = 0;
		try {
			m_debugWriter.write("===============================================================================\n");
//			m_debugWriter.write(String.format("Label:%d, prodID:%s, fu:%.4f, SVM:%d, Content:%s\n", d.getYLabel(), d.getItemID(), m_fu[id], (int)m_Y[id], d.getSource()));
			m_debugWriter.write(String.format("Label:%d, Predicted:%.4f, SVM:%d\n", d.getYLabel(), m_fu[id], (int)m_Y[id]));

			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++){
				similarity = getCache(id, m_U + j);
				if(similarity > m_threshold){
					m_KUL.add(new _RankItem(j, similarity));
					if(m_labeled.get(j).getYLabel()==d.getYLabel())
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
				similarity = getCache(id, m_U + j);
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
	protected void tmpDebug(_Doc d){
		double sameL = 0, sameU = 0, similarity = 0;
		int id = d.getID();
		_RankItem item;
		_Doc neighbor;
		
//		System.out.println(m_KUL.size());
		/****Construct the top k labeled data for the current data.****/
		for (int j = 0; j < m_L; j++){
			similarity = getCache(id, m_U + j);
			if(similarity > m_threshold)
				m_KUL.add(new _RankItem(j, similarity));
		}

		for(int i=0; i < m_KUL.size(); i++){
			neighbor = m_labeled.get(m_KUL.get(i).m_index);
			if(neighbor.getYLabel() == d.getYLabel())
				sameL++;
		}
		sameL = sameL / (m_KUL.size() + 0.0001);
		m_KUL.clear();
		
		/****Construct the top k' unlabeled data for the current data.****/
		for (int j = 0; j < m_U; j++) {
			if (j == id)
				continue;
			similarity = getCache(id, m_U + j);
			if(similarity > m_threshold)
				m_KUU.add(new _RankItem(j, similarity));
		}

		for(int i=0; i < m_KUU.size(); i++){
			item = m_KUU.get(i);
			neighbor = m_testSet.get(item.m_index);
			if(neighbor.getYLabel() == d.getYLabel())
				sameU++;
		}
		sameU = sameU / (m_KUU.size() + 0.0001);
		m_KUU.clear();
		m_debugStat.add(new double[]{sameL, sameU});
	} 	
	
	public int predict(_Doc doc) {
		return -1; //we don't support this in transductive learning
	}
	
	//Save the parameters for classification.
	@Override
	public void saveModel(String modelLocation) {
		
	}
	
	//Construct the look-up table for the later debugging use.
	public void setFeaturesLookup(HashMap<String, Integer> featureNameIndex){
		m_IndexFeature = new HashMap<Integer, String>();
		for(String f: featureNameIndex.keySet()){
			m_IndexFeature.put(featureNameIndex.get(f), f);
		}
	}
}