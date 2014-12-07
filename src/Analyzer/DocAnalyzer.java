package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;

import opennlp.tools.util.InvalidFormatException;
import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import structures._stat;
import utils.Utils;

public class DocAnalyzer extends Analyzer {
	
	private int m_window; //The length of the window which means how many labels will be taken into consideration.
	private LinkedList<Integer> m_YLabelQueue;
	private LinkedList<_SparseFeature[]> m_SpVctQueue;
	
	//Constructor.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, String fs) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo);
		if(providedCV != null)
			this.LoadCV(providedCV);
		if(fs != null){
			this.m_isFetureSelected = true;
			this.featureSelection = fs;
		}	
	}	
	
	//Constructor with ngram and fValue.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, String fs, int Ngram) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo, Ngram);
		if(providedCV != null)
			this.LoadCV(providedCV);
		if(fs != null){
			this.m_isFetureSelected = true;
			this.featureSelection = fs;
		}
	}
	
	//Constructor with ngram and time series analysis.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, String fs, int Ngram, boolean timeFlag, int window) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo, Ngram);
		if(providedCV != null)
			this.LoadCV(providedCV);
		if(fs != null){
			this.m_isFetureSelected = true;
			this.featureSelection = fs;
		}
		this.m_timeFlag = timeFlag;
		this.m_window = window;
		this.m_YLabelQueue = new LinkedList<Integer>();
		this.m_SpVctQueue = new LinkedList<_SparseFeature[]>();
	}
	
	//Load the features from a file and store them in the m_featurNames.
	public boolean LoadCV(String filename) {
		int count = 0;
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			//StringBuffer buffer = new StringBuffer(1024);
			String line;

			while ((line = reader.readLine()) != null) {
				//buffer.append(line);
				this.m_featureNames.add(line);
			}
			reader.close();
		} catch(IOException e){
				System.err.format("[Error]Failed to open file %s!!", filename);
				e.printStackTrace();
				return false;
		}
		//Indicate we can only use the loaded features to construct the feature vector!!
		m_isCVLoaded = true;
		
		//Set the index of the features.
		for(String f: this.m_featureNames){
			this.m_featureNameIndex.put(f, count);
			this.m_featureIndexName.put(count, f);
			this.m_featureStat.put(f, new _stat(this.m_classNo));
			count++;
		}
		return true; // if loading is successful
	}
	
	//Save all the features and feature stat into a file.
	public PrintWriter SaveCV(String featureLocation) throws FileNotFoundException {
		// File file = new File(path);
		PrintWriter writer = new PrintWriter(new File(featureLocation));
		for (int i = 0; i < this.m_featureNames.size(); i++)
			writer.println(this.m_featureNames.get(i));
		writer.close();
		return writer;
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
	}

	//Load a document and analyze it.
	public void LoadDoc(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;

			while ((line = reader.readLine()) != null) {
				buffer.append(line);
			}
			reader.close();
			//How to generalize it to several classes???? 
			if(filename.contains("pos")){
				//Collect the number of documents in one class.
				AnalyzeDoc(new _Doc(m_corpus.getSize(), buffer.toString(), 0));
				this.m_classMemberNo[0]++;
			}else if(filename.contains("neg")){
				AnalyzeDoc(new _Doc(m_corpus.getSize(), buffer.toString(), 1));
				this.m_classMemberNo[1]++;
			}
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
		}
		//this.m_corpus.sizeAddOne();
	}

	/*Analyze a document and add the analyzed document back to corpus.	
	 *In the case CV is not loaded, we need two if loops to check. 
	 * The first is if the term is in the vocabulary.***I forgot to check this one!
	 * The second is if the term is in the sparseVector.
	 * In the case CV is loaded, we still need two if loops to check.*/
	public void AnalyzeDoc(_Doc doc) {
		try{
			String[] tokens = TokenizerNormalizeStemmer(doc.getSource());//Three-step analysis.
			doc.setTotalLength(tokens.length); //set the length of the document.
			HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); //Collect the index and counts of features.
			int index = 0;
			double value = 0;
			//Construct the sparse vector.
			for(String token:tokens) {
				//CV is not loaded, take all the tokens as features.
				if(!m_isCVLoaded){
					if (m_featureNameIndex.containsKey(token)) {
						index = m_featureNameIndex.get(token);
						if(spVct.containsKey(index)){
							value = spVct.get(index) + 1;
							spVct.put(index, value);
							this.m_featureStat.get(token).addOneTTF(doc.getYLabel());
						} else{
							spVct.put(index, 1.0);
							this.m_featureStat.get(token).addOneDF(doc.getYLabel());
							this.m_featureStat.get(token).addOneTTF(doc.getYLabel());
						}
					} 
					else{
						//indicate we allow the analyzer to dynamically expand the feature vocabulary
						expandVocabulary(token);//update the m_featureNames.
						updateFeatureStat(token);
						index = m_featureNameIndex.get(token);
						spVct.put(index, 1.0);
						this.m_featureStat.get(token).addOneDF(doc.getYLabel());
						this.m_featureStat.get(token).addOneTTF(doc.getYLabel());					
					}
				} else if (m_featureNameIndex.containsKey(token)) {//CV is loaded.
					index = m_featureNameIndex.get(token);
					if(spVct.containsKey(index)){
						value = spVct.get(index) + 1;
						spVct.put(index, value);
						
					} else {
						spVct.put(index, 1.0);
						this.m_featureStat.get(token).addOneDF(doc.getYLabel());
					}
					this.m_featureStat.get(token).addOneTTF(doc.getYLabel());
				}
				//if the token is not in the vocabulary, nothing to do.
			}
			//If the timeflag is not set.
			if(!this.m_timeFlag){
				doc.createSpVct(spVct);
				doc.L2Normalization(doc.getSparse());
				m_corpus.addDoc(doc);
			} else { //If the timeflag is set.
				if(this.m_YLabelQueue.size() < m_window){
					this.m_YLabelQueue.add(doc.getYLabel());
					this.m_SpVctQueue.add(doc.createSpVct(spVct));
				}
				else{
					this.m_YLabelQueue.add(doc.getYLabel());
					this.m_SpVctQueue.add(doc.createSpVct(spVct));
					this.m_SpVctQueue.remove();
					this.m_YLabelQueue.remove();
				}
				//
				if(this.m_YLabelQueue.size() == m_window && this.m_SpVctQueue.size() == m_window){
					doc.createSpVctWithTime(this.m_YLabelQueue, this.m_SpVctQueue, this.m_featureNames.size());
					// doc.L2Normalization(doc.getSparse());//Normalize the sparse.
					m_corpus.addDoc(doc);
				}
			}
		}catch(Exception e) {e.printStackTrace();}
	}
	
	//Return the total number of words in vocabulary.
	public int getFeatureSize(){
		return this.m_featureNames.size();
	}
	
	//Set the counts of every feature with respect to the collected class number.
	public void setFeatureConfiguration() {
		//Initialize the counts of every feature.
		for (String featureName: this.m_featureStat.keySet()){
			this.m_featureStat.get(featureName).initCount(this.m_classNo);
		}
		for (String featureName : this.m_featureStat.keySet()) {
			this.m_featureStat.get(featureName).setCounts(this.m_classMemberNo);
		}
	}
	
	//Select the features and store them in a file.
	public void featureSelection(String location, double startProb, double endProb, int threshold) throws FileNotFoundException{
		FeatureSelector selector = new FeatureSelector(startProb, endProb);
		this.m_corpus.setMasks(); // After collecting all the documents, shuffle all the documents' labels.
		this.setFeatureConfiguration(); //Construct the table for features.
		
		if(this.m_isFetureSelected){
			System.out.println("*******************************************************************");
			if (this.featureSelection.equals("DF")){
				this.m_featureNames = selector.DF(this.m_featureStat, threshold);
			}
			else if(this.featureSelection.equals("IG")){
				this.m_featureNames = selector.IG(this.m_featureStat, this.m_classMemberNo, threshold);
			}
			else if(this.featureSelection.equals("MI")){
				this.m_featureNames = selector.MI(this.m_featureStat, this.m_classMemberNo, threshold);
			}
			else if(this.featureSelection.equals("CHI")){
				this.m_featureNames = selector.CHI(this.m_featureStat, this.m_classMemberNo, threshold);
			}
		}
		this.SaveCV(location); // Save all the features and probabilities we get after analyzing.
		System.out.println(this.m_featureNames.size() + " features are selected!");
	}
}	

