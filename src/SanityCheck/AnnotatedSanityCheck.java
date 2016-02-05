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
//					if(Integer.valueOf(strs[3]) == 0)
//						group = 0;
//					else if((Integer.valueOf(strs[3]) == 1) || (Integer.valueOf(strs[3]) == 6))
//						group = 1; // One-polarity reviews.
//					else 
//						group = 2; // Mix-polarity reviews.
					group = Integer.valueOf(strs[3]);
					
					doc = m_corpus.getCollection().get(ID);
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
	
	public double[] trainL2R(ArrayList<_Doc> trainSet, _Doc testDoc){
		ArrayList<_Query> queries = createTrainingCorpus(trainSet, testDoc);
		
		ArrayList<Feature[]> fvs = new ArrayList<Feature[]>();
		ArrayList<Integer> labels = new ArrayList<Integer>();
		
		for(_Query q: queries)
			q.extractPairs4RankSVM(fvs, labels);
		
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

//			if((m_allDocs.get(r.m_index).getYLabel() == testDoc.getYLabel()) && (m_allDocs.get(r.m_index).getGroupNo() == testDoc.getGroupNo())){
				count++;
				AP += count / totalCount;
			}
		}
		System.out.format("AP: %.4f\n", AP/count);
		return AP / count; // AP for the test query.
	}
	
	//NOTE: this similarity is no longer symmetric!!
	public double calcSimilarity(_Doc di, _Doc dj, double[] weights) {
		double similarity = 0;
		if(m_sType == SimType.ST_L2R)
			similarity = Utils.dotProduct(weights, genRankingFV(di, dj));	
//			similarity = Utils.dotProduct(weights, normalize(genRankingFV(di, dj)));	
		else if(m_sType == SimType.ST_BoW)
			similarity = getBoWSim(di, dj);
		else if(m_sType == SimType.ST_TP)
			similarity = Math.exp(-getTopicalSim(di, dj));
		else if(m_sType == SimType.ST_BoWTP){
//			double a = getBoWSim(di, dj);
//			double b = getTopicalSim(di, dj);
			similarity = getBoWSim(di, dj) + Math.exp(-getTopicalSim(di, dj));
		}
		else if(m_sType == SimType.ST_Rand)
			similarity = Math.random();
		return similarity;
	}
	/***
	//Init: trainSet = corpus.getCollection(); remove the testSet to get the trainSet.
	public void rmTestDocs(){
		Collections.sort(m_testIndexes, Collections.reverseOrder());
		for(int i=0; i<m_testIndexes.size(); i++){
			int index = m_testIndexes.get(i);//Why???
			m_trainSet.remove(index);
		}
		System.out.format("There are %d reviews in corpus.\n", m_corpus.getCollection().size());
		System.out.format("There are %d reviews in train set.\n", m_trainSet.size()); 
	}
	
	//Get the size of each group.
	public int[] getGroupSize(){
		int[] gSize = new int[m_groupDocs.size()];
		for(int index: m_groupDocs.keySet()){
			gSize[index] = m_groupDocs.get(index).size();
		}
		return gSize;
	}
	
	// In this function, we will use leave-one-out cross validation to calculate the AP for each query.
	public void LOOCrossValidation(){
		
	}
	//Train liblinear based on the trainSet.
	public double[] trainSVM(){
		double C = 1.0;
		double[] precision = new double[m_groupDocs.size()];
		m_svm = new SVM(m_corpus, C);
		m_svm.train(m_groupDocs);
		for(int index: m_groupDocs.keySet()){
			for(_Doc d: m_groupDocs.get(index)){
				if(m_svm.predict(d) == d.getYLabel())
					precision[index]++;
			}
			precision[index] /= m_groupDocs.get(index).size();
		}
		return precision;
	}
	//Given an array and a value,return the index of the value.
	public static int findValue(double[] a, double val){
		int start = 0, end = a.length -1, middle = 0;
		while(start <= end){
			middle = (start + end)/2;
			if(a[middle] == val)
				return middle;
			else if(a[middle] < val)
				start = middle + 1;
			else 
				end = middle - 1;
		}
		System.err.print("Index not found!");
		return -1;
	}
	
	// Use BoW to calculate the purity.
	public double[] constructPurity(int topK, int flag, String filename) throws FileNotFoundException{
		
		double count = 0, val;
		int in = 0, length;
		_Doc dj, tmp;
		MyPriorityQueue<_RankItem> queue = new MyPriorityQueue<_RankItem>(topK);
		double[] purity = new double[m_groupDocs.size()];
		double[] values;
		
		for(int index: m_groupDocs.keySet()){
			// Access each document.
			PrintWriter writer = new PrintWriter(new File(filename+index+".xls"));
			for(_Doc d: m_groupDocs.get(index)){
				
				//Write out the current document.
//				writer.write("==================================================\n");
				writer.format("trueL:%d\n", d.getYLabel());
				if(flag == 0){
					for(_SparseFeature sf: d.getSparse())
						writer.format("(%s, %.4f)\t", m_features.get(sf.getIndex()), sf.getValue());
				} else{
					length = d.getTopics().length;//Get the length of the topics.
					values = Arrays.copyOf(d.getTopics(), length);
					Arrays.sort(values); //Sort the topic vector.
					
					//Construct the value-index map.
					HashMap<Double, Integer> valIndexMap = new HashMap<Double, Integer>();
					for(int i=0; i<d.getTopics().length; i++)
						valIndexMap.put(d.getTopics()[i], i);
					
					//Print out the value and index.
					for(int j=0; j<length; j++){
						val = values[length-1-j];
						in = valIndexMap.get(val);
						//I want to print topic in a descending order based on values.
						writer.format("(%d, %.4f)\t", in, val);
					}
				}
//					for(int i=0; i<length; i++)
//						writer.format("(%d, %.4f)\t", i, d.getTopics()[i]);
//				}
//				writer.format("\n%s\n", d.getSource());

				// Construct neighborhood.
				for(int i=0; i<m_trainSet.size(); i++){
					dj = m_trainSet.get(i);
					//The i is the index in the trainSet, rather than the collection of corpus.
					if(flag == 0)
						queue.add(new _RankItem(i, Utils.calculateSimilarity(d, dj)));
					else
						queue.add(new _RankItem(i, Math.exp(-Utils.klDivergence(d.getTopics(), dj.getTopics()))));
				}
				
				// Get the purity.
				for(_RankItem item: queue){
					tmp = m_trainSet.get(item.m_index);
					if(d.getYLabel() == tmp.getYLabel())
						count++;
					
//					//Write the neighbors' information.
//					writer.format("trueL:%d\n", tmp.getYLabel());
//					if(flag == 0){
//						for(_SparseFeature sf: tmp.getSparse())
//							writer.format("(%s, %.4f)\t", m_features.get(sf.getIndex()), sf.getValue());
//					} else{
//						for(int j=0; j<tmp.getTopics().length; j++)
//							writer.format("(%d, %.4f)\t", j, tmp.getTopics()[j]);
//					}
//					writer.format("\n%s\n", tmp.getSource());
				}
				purity[index] += count/(double) topK;
				writer.format("Count:%d, Purity:%.4f\n", (int)count, count/(double)topK);
				count = 0;
				queue.clear();
			}
			purity[index] /= m_groupTrainSets.get(index).size();
			System.out.format("Finish writing %d group reviews.\n", index);
			writer.close();
		}
		return purity;
	}

	// Load the file without IDs and human annotations.
	public void loadCheckFile(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(filename), "UTF-8"));
			String line, strLabel;
			int label = 0, lineCount = 0; // Use this count to split different reviews since each review has two lines.
			while ((line = reader.readLine()) != null) {
				if (lineCount % 2 == 0) {
					strLabel = line.split(":")[1].trim();
					label = Integer.valueOf(strLabel);
				} else {
					m_testSet.add(new _Doc(0, line, label));
					m_sourceIndexMap.put(line, lineCount / 2);
				}
				lineCount++;
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	//Find the test document in the collection and add the ID to the test document.
	public void setTestFileIDs(){
		_Doc tmp;
		int ID = 0;
		for(int i=0; i<m_documents.size(); i++){
			tmp = m_documents.get(i);
			if(m_sourceIndexMap.containsKey(tmp.getSource())){
				ID = m_documents.get(i).getID();
				m_testSet.get(m_sourceIndexMap.get(tmp.getSource())).setID(ID);
			}
		}
	}
	
	//Print out the 100 files with indexes, IDs.
	public void printFile(String filename){
		try{
			_Doc tmp;
			PrintWriter writer = new PrintWriter(new File(filename));
			for(int i=0; i<m_testSet.size(); i++){
				tmp = m_testSet.get(i);
				writer.format("%d,%d,%d\n", i, tmp.getID(), tmp.getYLabel());
				writer.format("%s\n", tmp.getSource());
			}
			System.out.format("Write %d files into file %s\n", m_testSet.size(), filename);
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
***/
}
