package SanityCheck;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import Ranker.LambdaRank.OptimizationType;
import SanityCheck.BaseSanityCheck.SimType;
import Classifier.metricLearning.L2RMetricLearning;
import Classifier.supervised.SVM;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.SolverType;
import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._QUPair;
import structures._Query;
import structures._RankItem;
import utils.Utils;

public class AnnotatedSanityCheck extends L2RMetricLearning{
	double[] m_MAPs;
	ArrayList<_Doc> m_allDocs;
	SimType m_sType;
	int m_numberOfCores;
	Object m_MAPLock = new Object();
	PrintWriter m_writer;
	boolean m_flip = false; //Whether we need to flip the feature vector for the negative groups.
	
	/*** Key: group index, value: documents arraylist.
	1: pos pos pos +; 2: pos pos neg +; 3: pos neg neg +; 4: neg neg neg -; 5: neg neg pos -; 6: neg pos pos -; 0: the others.***/
	HashMap<Integer, ArrayList<_Doc>> m_groupDocs;
	
	public AnnotatedSanityCheck(_Corpus c, String classifier, double C, int topK, SimType sType) {
		super(c, classifier, C, topK);
		m_sType = sType;
		m_groupDocs = new HashMap<Integer, ArrayList<_Doc>>();
	}
	
	// Write out the weights of each learning to rank.
	public void initWriter(String filename) throws FileNotFoundException{
		m_writer = new PrintWriter(new File(filename));
	}

	// Load the file with IDs and human annotations into different groups.
	public void loadAnnotatedFile(String filename){
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			_Doc doc;
			// group: sentiment group.
			int ID = 0, group = 0, lineCount = 0;
			String[] strs;
			while((line = reader.readLine()) != null){
				if(lineCount%2 == 0){
					strs = line.split(",");//The first is the index, ignore it.
					ID = Integer.valueOf(strs[1]); // The ID of the document in the whole corpus.
					if(Integer.valueOf(strs[3]) == 0)
						group = 0;
					else if((Integer.valueOf(strs[3]) == 1) || (Integer.valueOf(strs[3]) == 4))
						group = 1; // One-polarity reviews.
					else 
						group = 2; // Mix-polarity reviews.
//					group = Integer.valueOf(strs[3]);
					
					doc = m_corpus.getCollection().get(ID);
					if(doc.getStnLabels() == null)
						System.out.println("No stn labels for this document.");
					if(!m_groupDocs.containsKey(group))
						m_groupDocs.put(group, new ArrayList<_Doc>());
					doc.setGroupNo(group);
					m_groupDocs.get(group).add(doc);
					
				}
				lineCount++;
			}
			System.out.format("%d reviews loaded into sytem.\n", lineCount/2);
			
			// Merge all the docuements into one set.
			m_allDocs = new ArrayList<_Doc>();
			for(int groupNo: m_groupDocs.keySet())
				m_allDocs.addAll(m_groupDocs.get(groupNo));
			
			m_MAPs = new double[m_groupDocs.size() -1];
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void calculatePrecision(double threshold){
		int trueG = 0, predG = 0;
		int[][] TPTable = new int[2][2];
//		// Sort all the docs based on the neg ratio.
//		Collections.sort(m_allDocs, new Comparator<_Doc>(){
//			public int compare(_Doc d1, _Doc d2){
//				if(d1.getNegRatio() < d2.getNegRatio())
//					return 1;
//				else return -1;
//			}
//		});
		for(_Doc d: m_allDocs){
			if(d.getStnLabels() != null && d.getGroupNo() != 0){
				// Single polarity.
				if(d.getPosRatio() <= threshold || Math.abs(1-d.getPosRatio()) <= threshold)
					predG = 0;
				else
					predG = 1;
				if(d.getGroupNo() == 1 || d.getGroupNo() == 4)
					trueG = 0;
				else 
					trueG = 1;
				TPTable[trueG][predG]++;
			}
		}
		for(int i=0; i<TPTable.length; i++){
			for(int j=0; j<TPTable[i].length; j++)
				System.out.print(TPTable[i][j] + "\t");
			System.out.println();
		}
	}
	// Compare the human annotation and machine annotation.
	public void compareAnnotation(String filename) throws FileNotFoundException{
		int[] stnStmLabels;
		PrintWriter writer = new PrintWriter(new File(filename));
		for(int groupNo: m_groupDocs.keySet()){
			for(_Doc d: m_groupDocs.get(groupNo)){
				writer.format("%d\t%.4f\n", groupNo, d.getPosRatio());
				stnStmLabels = d.getStnStmLabels();
				if(stnStmLabels != null){
					for(int l: stnStmLabels)
						writer.write(l+"\t");
				}
				writer.format("\n%s\n", d.getSource());
			}
		}
		writer.close();
	}
	public void diffGroupLOOCV(){
		for(int groupNo: m_groupDocs.keySet()){
			if(groupNo == 0) 
				continue;
			else
				LOOCV(groupNo, m_groupDocs.get(groupNo));
		}
		System.out.println("---------------------------------");
		for(double m: m_MAPs)
			System.out.format("%.4f\t", m);
		System.out.println();
		m_writer.close();
	}
	
	public double[] getMAPs(){
		return m_MAPs;
	}
	
	// Leave-one-out cross validation.
	public void LOOCV(final int groupNo, final ArrayList<_Doc> groupDocs){
//		if(groupNo > 3)
//			m_flip = true;
		m_numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		m_writer.format("-----------------------%d------------------------\n", groupNo);
		
		for(int k=0; k<m_numberOfCores; k++){
			threads.add((new Thread() {
				int core;
				public void run() {
					try{
						for(int i=0; i+core < groupDocs.size(); i += m_numberOfCores){
							System.out.println("----Current index is " + (i+core));
							// Leave one out to construct the training set.
							_Doc testDoc = groupDocs.get(i+core); // Get the test document.
							ArrayList<_Doc> trainSet = new ArrayList<_Doc>(groupDocs);
							trainSet.remove(i+core);
							
							// Train L2R model.
							if(m_sType == SimType.ST_L2R){
								double[] weights = trainL2R(trainSet, testDoc);
								for(double w: weights)
									m_writer.format("%.4f\t", w);
								m_writer.write("\n");
								// Get the permutation of for the test query and calculate corresponding AP.
								synchronized(m_MAPLock){
									m_MAPs[groupNo-1] += permutate(trainSet, testDoc, weights);
								}
							} 
							else {
								synchronized(m_MAPLock){
									m_MAPs[groupNo-1] += permutate(trainSet, testDoc, null);
								}
							}
						}
					} catch(Exception ex) {
						ex.printStackTrace(); 
					}
				}
		
				private Thread initialize(int core) {
					this.core = core;
					return this;
				}
			}).initialize(k));
	
			threads.get(k).start();
		}

		for(int k=0;k<m_numberOfCores;++k){
			try {
				threads.get(k).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		}
		// Calculate the MAP for all the test documents.
		m_MAPs[groupNo-1] /= groupDocs.size();
		System.out.format("G: %d, map: %.4f\t", groupNo, m_MAPs[groupNo-1]);
	}
	
	public double[] reverseWeights(double[] weights){
		if(weights == null)
			return null;
		for(int i=0; i<weights.length; i++)
			weights[i] = -weights[i];
		return weights;
	}
//	// Leave-one-out cross validation in single-thread.
//	public void LOOCV(int groupNo, ArrayList<_Doc> groupDocs){
//			
//		for(int i=0; i < groupDocs.size(); i++){
//			// Leave one out to construct the training set.
//			_Doc testDoc = groupDocs.get(i); // Get the test document.
//			ArrayList<_Doc> trainSet = new ArrayList<_Doc>(groupDocs);
//			trainSet.remove(i);
//								
//			// Train L2R model.
//			if(m_sType == SimType.ST_L2R){
//				double[] weights = trainL2R(trainSet, testDoc);
//				m_MAPs[groupNo-1] += permutate(trainSet, testDoc, weights);
//			} 
//			else{
//				m_MAPs[groupNo-1] += permutate(trainSet, testDoc, null);
//			}
//		}
//		// Calculate the MAP for all the test documents.
//		m_MAPs[groupNo-1] /= groupDocs.size();
//		System.out.format("G: %d, map: %.4f\t", groupNo, m_MAPs[groupNo-1]);
//	}	
	
	public double[] trainL2R(ArrayList<_Doc> trainSet, _Doc testDoc){
		ArrayList<_Query> queries = createTrainingCorpus(trainSet, testDoc);
		
		ArrayList<Feature[]> fvs = new ArrayList<Feature[]>();
		ArrayList<Integer> labels = new ArrayList<Integer>();
		
		for(_Query q: queries)
			q.extractPairs4RankSVM(fvs, labels, m_flip);
	
		Model rankSVM = SVM.libSVMTrain(fvs, labels, RankFVSize, SolverType.L2R_L1LOSS_SVC_DUAL, m_tradeoff, -1);
		double[] weights = rankSVM.getFeatureWeights();
		System.out.format("RankSVM training performance:\nMAP: %.4f\n", evaluate(OptimizationType.OT_MAP));	
		return weights;
	}
	
	// Train different learning to rank models for different groups.
	public ArrayList<_Query> createTrainingCorpus(ArrayList<_Doc> trainSet, _Doc testDoc){
		_Query q;
		_Doc di, dj;
		int pairSize = 0, posQ = 0, negQ = 0;
		double relevant = 0, irrelevant = 0;
		ArrayList<_Query> queries = new ArrayList<_Query>();

		for(int i=0; i<trainSet.size(); i++){
			di = trainSet.get(i);
			relevant = 0;
			irrelevant = 0;
			
			// Construct query-document pairs among all the 100 files.
			for(int j=0; j<m_allDocs.size(); j++){
				dj = m_allDocs.get(j);
				// Filter the test document and current query document.
				if(dj.getID() != di.getID() && dj.getID() != testDoc.getID()){
					if(di.getYLabel() == dj.getYLabel())
						relevant++;
					else 
						irrelevant++;
				}
			}
			
			// Judge if the document has both relevant and irrelevant documents.
			if(relevant == 0 || irrelevant == 0)
				continue;
			else if(di.getYLabel() == 1)
				posQ++;
			else 
				negQ++;
				
			q = new _Query();
			queries.add(q);
			for(int j=0; j<m_allDocs.size(); j++){
				dj = m_allDocs.get(j);
				if(dj.getID() != di.getID() && dj.getID() != testDoc.getID())
					q.addQUPair(new _QUPair(di.getYLabel() == dj.getYLabel()?1:0, genRankingFV(di, dj)));
			}
			pairSize += q.createRankingPairs();
		}
//		normalize();
		System.out.format("Generate %d(%d:%d) ranking pairs for L2R model training...\n", pairSize, posQ, negQ);
		return queries;
	}
	
	// For the current test document, get the permutation of all the remaining documents.
	public double permutate(ArrayList<_Doc> trainSet, _Doc testDoc, double[] weights){
		double count = 0, totalCount = 0;
		double sim = 0, AP = 0;
		_Doc di;
		MyPriorityQueue<_RankItem> rankDocs = new MyPriorityQueue<_RankItem>(m_topK);
		for(int i=0; i<m_allDocs.size(); i++){
			di = m_allDocs.get(i);
			if(di.getID() != testDoc.getID()){
				sim = calcSimilarity(di, testDoc, weights);
				rankDocs.add(new _RankItem(i, sim));
			}
		}
		for(_RankItem r: rankDocs){
			totalCount++;
			if((m_allDocs.get(r.m_index).getYLabel() == testDoc.getYLabel())){
				count++;
				AP += count / totalCount;
			}
			System.out.print(r.m_value+"\t");
		}
		System.out.format("\nAP: %.4f\n", AP/count);
		return AP / count; // AP for the test query.
	}
	
	//NOTE: this similarity is no longer symmetric!!
	public double calcSimilarity(_Doc di, _Doc dj, double[] weights) {
		double similarity = 0;
		if(m_sType == SimType.ST_L2R)
			similarity = Utils.dotProduct(weights, genRankingFV(di, dj));	
		else if(m_sType == SimType.ST_BoW)
			similarity = getBoWSim(di, dj);
		else if(m_sType == SimType.ST_TP)
			similarity = Math.exp(-getTopicalSim(di, dj));
		else if(m_sType == SimType.ST_BoWTP){

			similarity = getBoWSim(di, dj) + Math.exp(-getTopicalSim(di, dj));
		}
		else if(m_sType == SimType.ST_Rand)
			similarity = Math.random();
		return similarity;
	}
}
