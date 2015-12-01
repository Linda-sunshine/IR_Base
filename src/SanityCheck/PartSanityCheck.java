package SanityCheck;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import Classifier.supervised.SVM;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;

public class PartSanityCheck extends FurtherPuritySanityCheck{
	
	ArrayList<_Doc> m_testSet; // There are the part of docs need to be checked.
	HashMap<String, Integer> m_sourceIndexMap;
	ArrayList<Integer> m_testIndexes; // The documents loaded as test documents.
	ArrayList<_Doc> m_trainSet;
	SVM m_svm; //liblinear model.
	
	// Key: group index, value: documents arraylist.
	// 1: pos pos pos +; 2: pos pos neg +; 3: pos neg neg +; 4: neg neg neg -; 5: neg neg pos -; 6: neg pos pos -; 0: the others.
	HashMap<Integer, ArrayList<_Doc>> m_groupIndexDocsMap; 
	
	public PartSanityCheck(_Corpus c) {
		super(c);
		m_testSet = new ArrayList<_Doc>();
		m_sourceIndexMap = new HashMap<String, Integer>();
		m_testIndexes = new ArrayList<Integer>();
		m_groupIndexDocsMap = new HashMap<Integer, ArrayList<_Doc>>();
		m_trainSet = new ArrayList<_Doc>(c.getCollection()); //Copy all the files as train set, filter later.
	}
	
	//Load the file with IDs and human annotations.
	public void loadAnnotatedFile(String filename){
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			_Doc doc;
			//group: sentiment group.
			int ID = 0, group = 0, lineCount = 0;
			String[] strs;
			while((line = reader.readLine()) != null){
				if(lineCount%2 == 0){
					strs = line.split(",");//The first is the index, ignore it.
					ID = Integer.valueOf(strs[1]); // The ID of the document in the whole corpus.
					group = Integer.valueOf(strs[3]); // The human annotated group information.
				
					doc = m_corpus.getCollection().get(ID);
					m_testSet.add(doc); //Add it to the test set.
					m_testIndexes.add(ID);
//					m_trainSet.remove(ID); //Remove it from the train set.
					if(!m_groupIndexDocsMap.containsKey(group))
						m_groupIndexDocsMap.put(group, new ArrayList<_Doc>());
					m_groupIndexDocsMap.get(group).add(doc);
				}
				lineCount++;
			}
			System.out.format("%d reviews loaded into sytem.\n", lineCount/2);
			reader.close();
			rmTestDocs();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
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
		int[] gSize = new int[m_groupIndexDocsMap.size()];
		for(int index: m_groupIndexDocsMap.keySet()){
			gSize[index] = m_groupIndexDocsMap.get(index).size();
		}
		return gSize;
	}
	
	//Train liblinear based on the trainSet.
	public double[] trainSVM(){
		double C = 1.0;
		double[] precision = new double[m_groupIndexDocsMap.size()];
		m_svm = new SVM(m_corpus, C);
		m_svm.train(m_trainSet);
		for(int index: m_groupIndexDocsMap.keySet()){
			for(_Doc d: m_groupIndexDocsMap.get(index)){
				if(m_svm.predict(d) == d.getYLabel())
					precision[index]++;
			}
			precision[index] /= m_groupIndexDocsMap.get(index).size();
		}
		return precision;
	}
	// Use BoW to calculate the purity.
	public double[] constructPurity(int topK, int flag, String filename) throws FileNotFoundException{
		
		double count = 0;
		_Doc dj, tmp;
		MyPriorityQueue<_RankItem> queue = new MyPriorityQueue<_RankItem>(topK);
		double[] purity = new double[m_groupIndexDocsMap.size()];
		for(int index: m_groupIndexDocsMap.keySet()){
			// Access each document.
			PrintWriter writer = new PrintWriter(new File(filename+index));
			for(_Doc d: m_groupIndexDocsMap.get(index)){
				
				//Write out the current document.
				writer.write("==================================================\n");
				writer.format("trueL:%d\n", d.getYLabel());
				if(flag == 0){
					for(_SparseFeature sf: d.getSparse())
						writer.format("(%s, %.4f)\t", m_features.get(sf.getIndex()), sf.getValue());
				} else{
					for(int j=0; j<d.getTopics().length; j++)
						writer.format("(%d, %.4f)\t", j, d.getTopics()[j]);
				}
				writer.format("\n%s\n", d.getSource());

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
					
					//Write the neighbors' information.
					writer.format("trueL:%d\n", tmp.getYLabel());
					if(flag == 0){
						for(_SparseFeature sf: tmp.getSparse())
							writer.format("(%s, %.4f)\t", m_features.get(sf.getIndex()), sf.getValue());
					} else{
						for(int j=0; j<tmp.getTopics().length; j++)
							writer.format("(%d, %.4f)\t", j, tmp.getTopics()[j]);
					}
					writer.format("\n%s\n", tmp.getSource());
				}
				purity[index] += count/(double) topK;
				writer.format("Count:%d, Purity:%.4f\n", (int)count, count/(double)topK);
				count = 0;
				queue.clear();
			}
			purity[index] /= m_groupIndexDocsMap.get(index).size();
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
//				System.out.println("test: " + m_testDocs.get(m_sourceIndexMap.get(tmp.getSource())).getSource());
//				System.out.println("collection: " + tmp.getSource());
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
}
