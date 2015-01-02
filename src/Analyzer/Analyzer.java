package Analyzer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;

import opennlp.tools.util.InvalidFormatException;
import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import structures._stat;
import utils.Utils;

public abstract class Analyzer {
	
	protected _Corpus m_corpus;
	protected int m_classNo; //This variable is just used to init stat for every feature. How to generalize it?
	int[] m_classMemberNo; //Store the number of members in a class.
	protected int m_Ngram; 
	
	protected ArrayList<String> m_featureNames; //ArrayList for features
	protected HashMap<String, Integer> m_featureNameIndex;//key: content of the feature; value: the index of the feature
	protected HashMap<String, _stat> m_featureStat; //Key: feature Name; value: the stat of the feature
	
	/** for time-series features **/
	//The length of the window which means how many labels will be taken into consideration.
	private LinkedList<_Doc> m_preDocs;	
	
	public Analyzer(String tokenModel, int classNo) throws InvalidFormatException, FileNotFoundException, IOException{
		m_corpus = new _Corpus();
		
		m_classNo = classNo;
		m_classMemberNo = new int[classNo];
		
		m_featureNames = new ArrayList<String>();
		m_featureNameIndex = new HashMap<String, Integer>();//key: content of the feature; value: the index of the feature
		m_featureStat = new HashMap<String, _stat>();
		m_preDocs = new LinkedList<_Doc>();
	}	
	
	public void reset() {
		Arrays.fill(m_classMemberNo, 0);
		m_featureNames.clear();
		m_featureNameIndex.clear();
		m_featureStat.clear();
		m_corpus.reset();
	}
	
	//Load all the files in the directory.
	public void LoadDirectory(String folder, String suffix) throws IOException {
		File dir = new File(folder);
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix)) {
				LoadDoc(f.getAbsolutePath());
			} else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix);
		}
		System.out.println();
	}
	
	abstract public void LoadDoc(String filename);
	
	//Save all the features and feature stat into a file.
	protected void SaveCVStat(String finalLocation) throws FileNotFoundException{
		if (finalLocation==null || finalLocation.isEmpty())
			return;
		
		PrintWriter writer = new PrintWriter(new File(finalLocation));
		for(int i = 0; i < m_featureNames.size(); i++){
			writer.print(m_featureNames.get(i));
			_stat temp = m_featureStat.get(m_featureNames.get(i));
			for(int j = 0; j < temp.getDF().length; j++)
				writer.print("\t" + temp.getDF()[j]);
			for(int j = 0; j < temp.getTTF().length; j++)
				writer.print("\t" + temp.getTTF()[j]);
			writer.println();
		}
		writer.close();
	}
	
	//Add one more token to the current vocabulary.
	protected void expandVocabulary(String token) {
		m_featureNameIndex.put(token, m_featureNames.size()); // set the index of the new feature.
		m_featureNames.add(token); // Add the new feature.
		m_featureStat.put(token, new _stat(m_classNo));
	}
		
	//Return corpus without parameter and feature selection.
	public _Corpus returnCorpus(String finalLocation) throws FileNotFoundException {
		SaveCVStat(finalLocation);
		System.out.format("Feature vector contructed for %d documents...\n", m_corpus.getSize());
		for(int c:m_classMemberNo)
			System.out.print(c + " ");
		System.out.println();
		
		//store the feature names into corpus
		m_corpus.setFeatures(m_featureNames);
		m_corpus.setMasks(); // After collecting all the documents, shuffle all the documents' labels.
		return m_corpus;
	}
	
	//Give the option, which would be used as the method to calculate feature value and returned corpus, calculate the feature values.
	public void setFeatureValues(String fValue, int norm) {
		ArrayList<_Doc> docs = m_corpus.getCollection(); // Get the collection of all the documents.
		int N = docs.size();
		if (fValue.equals("TF")){
			//the original feature is raw TF
		} else if (fValue.equals("TFIDF")) {
			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double TF = sf.getValue() / temp.getTotalDocLength();// normalized TF
					double DF = Utils.sumOfArray(stat.getDF());
					double TFIDF = TF * Math.log((N + 1) / DF);
					sf.setValue(TFIDF);
				}
			}
		} else if (fValue.equals("BM25")) {
			double k1 = 1.5; // [1.2, 2]
			double b = 0.75; // (0, 1000]
			// Iterate all the documents to get the average document length.
			double navg = 0;
			for (int k = 0; k < N; k++)
				navg += docs.get(k).getTotalDocLength();
			navg /= N;

			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double n = temp.getTotalDocLength() / navg;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double BM25 = Math.log((N - DF + 0.5) / (DF + 0.5)) * TF * (k1 + 1) / (k1 * (1 - b + b * n) + TF);
					sf.setValue(BM25);
				}
			}
		} else if (fValue.equals("PLN")) {
			double s = 0.5; // [0, 1]
			// Iterate all the documents to get the average document length.
			double navg = 0;
			for (int k = 0; k < N; k++)
				navg += docs.get(k).getTotalDocLength();
			navg /= N;

			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double n = temp.getTotalDocLength() / navg;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double PLN = (1 + Math.log(1 + Math.log(TF)) / (1 - s + s * n)) * Math.log((N + 1) / DF);
					sf.setValue(PLN);
				}
			}
		} else {
			//The default value is just keeping the raw count of every feature.
			System.out.println("No feature value is set, keep the raw count of every feature.");
		}
		
		//rank the documents by product and time in all the cases
		Collections.sort(m_corpus.getCollection());
		if (norm == 1){
			for(_Doc d:docs)			
				Utils.L1Normalization(d.getSparse());
		} else if(norm == 2){
			for(_Doc d:docs)			
				Utils.L2Normalization(d.getSparse());
		} else {
			System.out.println("No normalizaiton is adopted here or wrong parameters!!");
		}
		
		System.out.format("Text feature generated for %d documents...\n", m_corpus.getSize());
	}
	
	//Select the features and store them in a file.
	public void featureSelection(String location, String featureSelection, double startProb, double endProb, int threshold) throws FileNotFoundException {
		FeatureSelector selector = new FeatureSelector(startProb, endProb, threshold);

		System.out.println("*******************************************************************");
		if (featureSelection.equals("DF"))
			selector.DF(m_featureStat);
		else if (featureSelection.equals("IG"))
			selector.IG(m_featureStat, m_classMemberNo);
		else if (featureSelection.equals("MI"))
			selector.MI(m_featureStat, m_classMemberNo);
		else if (featureSelection.equals("CHI"))
			selector.CHI(m_featureStat, m_classMemberNo);
		
		m_featureNames = selector.getSelectedFeatures();
		SaveCV(location, featureSelection, startProb, endProb, threshold); // Save all the features and probabilities we get after analyzing.
		System.out.println(m_featureNames.size() + " features are selected!");
	}
	
	//Save all the features and feature stat into a file.
	protected void SaveCV(String featureLocation, String featureSelection, double startProb, double endProb, int threshold) throws FileNotFoundException {
		if (featureLocation==null || featureLocation.isEmpty())
			return;
		
		System.out.format("Saving controlled vocabulary to %s...\n", featureLocation);
		PrintWriter writer = new PrintWriter(new File(featureLocation));
		//print out the configurations as comments
		writer.format("#NGram:%d\n", m_Ngram);
		writer.format("#Selection:%s\n", featureLocation);
		writer.format("#Start:%f\n", startProb);
		writer.format("#End:%f\n", endProb);
		writer.format("#DF_Cut:%d\n", threshold);
		
		//print out the features
		for (int i = 0; i < m_featureNames.size(); i++)
			writer.println(m_featureNames.get(i));
		writer.close();
	}
	
	//Return the number of features.
	public int getFeatureSize(){
		return m_featureNames.size();
	}
	
	//Sort the documents.
	public void setTimeFeatures(int window){//must be called before return corpus
		if (window<1) 
			return;
		
		//Sort the documents according to time stamps.
		ArrayList<_Doc> docs = m_corpus.getCollection();
		
		/************************time series analysis***************************/
		double norm = 1.0 / m_classMemberNo.length, avg = 0;
		int count = 0;//for computing the moving average
		String lastItemID = null;
		for(int i = 0; i < docs.size(); i++){
			_Doc doc = docs.get(i);			
			
			if (lastItemID == null)
				lastItemID = doc.getItemID();
			else if (lastItemID != doc.getItemID()) {
				m_preDocs.clear(); // reviews for a new category of products
				lastItemID = doc.getItemID();
				
				//clear for moving average
				avg = 0;
				count = 0;
			}
			
			avg += doc.getYLabel();
			count += 1;
			
			if(m_preDocs.size() < window){
				m_preDocs.add(doc);
				m_corpus.removeDoc(i);
				m_classMemberNo[doc.getYLabel()]--;
				i--;
			} else{
				doc.createSpVctWithTime(m_preDocs, m_featureNames.size(), avg/count, norm);
				m_preDocs.remove();
				m_preDocs.add(doc);
			}
		}
		System.out.format("Time-series feature set for %d documents!\n", m_corpus.getSize());
	}
	
	// added by Md. Mustafizur Rahman for Topic Modelling
	public double[] get_back_ground_probabilty()
	{
		double back_ground_probabilty [] = new double [m_featureNameIndex.size()];
		
		for(int i = 0; i<m_featureNameIndex.size();i++)
		{
			String featureName = m_featureNames.get(i);
			_stat stat =  m_featureStat.get(featureName);
			back_ground_probabilty[i] = Utils.sumOfArray(stat.getTTF());
		}
		
		double sum = Utils.sumOfArray(back_ground_probabilty) + back_ground_probabilty.length;//add one smoothing
		for(int i = 0; i<m_featureNameIndex.size();i++)
			back_ground_probabilty[i] = (1.0 + back_ground_probabilty[i]) / sum;
		return back_ground_probabilty;
	}	
}
