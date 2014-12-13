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
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import structures._stat;
import utils.Utils;

public abstract class Analyzer {
	
	protected _Corpus m_corpus;
	protected Tokenizer m_tokenizer;
	protected SnowballStemmer m_stemmer;
	protected int m_classNo; //This variable is just used to init stat for every feature. How to generalize it?
	int[] m_classMemberNo; //Store the number of members in a class.
	
	//added by Hongning to manage feature vocabulary
	/* Indicate if we can allow new features.After loading the CV file, the flag is set to true, 
	 * which means no new features will be allowed.*/
	protected boolean m_isCVLoaded; 
	/* Indicate if the user has specified a feature selection method.
	 * If the user do not provides a feature selection method, then all terms will be chosen as CV. 
	 * So the default value is true.*/
	protected boolean m_isFetureSelected; 
	private int m_Ngram; 
	
	protected ArrayList<String> m_featureNames; //ArrayList for features
	protected HashMap<String, Integer> m_featureNameIndex;//key: content of the feature; value: the index of the feature
	protected HashMap<Integer, String> m_featureIndexName;//value: the index of the feature; key: content of the feature; 
	protected HashMap<String, _stat> m_featureStat; //Key: feature Name; value: the stat of the feature
	protected String featureSelection = "DF";
	
	/** for time-series features **/
	//The length of the window which means how many labels will be taken into consideration.
	private LinkedList<_Doc> m_preDocs;	
	
	public Analyzer(String tokenModel, int classNo) throws InvalidFormatException, FileNotFoundException, IOException{
		this.m_corpus = new _Corpus();
		this.m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		this.m_stemmer = new englishStemmer();
		this.m_classNo = classNo;
		this.m_classMemberNo = new int[classNo];
		
		this.m_isCVLoaded = false;
		this.m_isFetureSelected = false;
		this.m_Ngram = 1;
		
		this.m_featureNames = new ArrayList<String>();
		this.m_featureNameIndex = new HashMap<String, Integer>();//key: content of the feature; value: the index of the feature
		this.m_featureIndexName = new HashMap<Integer, String>();//value: content of the feature; key: the index of the feature
		this.m_featureStat = new HashMap<String, _stat>();
		this.featureSelection = "DF";
		
		this.m_preDocs = new LinkedList<_Doc>();
	}	
	
	public Analyzer(String tokenModel, int classNo, int Ngram) throws InvalidFormatException, FileNotFoundException, IOException{
		this.m_corpus = new _Corpus();
		this.m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		this.m_stemmer = new englishStemmer();
		this.m_classNo = classNo;
		this.m_classMemberNo = new int[classNo];
		
		this.m_isCVLoaded = false;
		this.m_isFetureSelected = false;
		this.m_Ngram = Ngram;
		
		this.m_featureNames = new ArrayList<String>();
		this.m_featureNameIndex = new HashMap<String, Integer>();//key: content of the feature; value: the index of the feature
		this.m_featureIndexName = new HashMap<Integer, String>();//value: content of the feature; key: the index of the feature
		this.m_featureStat = new HashMap<String, _stat>();
		this.featureSelection = "DF";
		
		this.m_preDocs = new LinkedList<_Doc>();
	}	
	
	//Load the features from a file and store them in the m_featurNames.@added by Lin.
	protected boolean LoadCV(String filename) {
		int count = 0;
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			while ((line = reader.readLine()) != null) {
				this.m_featureNames.add(line);
			}
			reader.close();
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
			return false;
		}
		// Indicate we can only use the loaded features to construct the feature
		m_isCVLoaded = true;

		// Set the index of the features.
		for (String f : this.m_featureNames) {
			this.m_featureNameIndex.put(f, count);
			this.m_featureIndexName.put(count, f);
			this.m_featureStat.put(f, new _stat(this.m_classNo));
			count++;
		}
		return true; // if loading is successful
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
	protected PrintWriter SaveCVStat(String finalLocation) throws FileNotFoundException{
		//File file = new File(path);
		PrintWriter writer = new PrintWriter(new File(finalLocation));
		for(int i = 0; i < this.m_featureNames.size(); i++){
			writer.print(this.m_featureNames.get(i));
			_stat temp = this.m_featureStat.get(this.m_featureNames.get(i));
			for(int j = 0; j < temp.getDF().length; j++)
				writer.print("\t" + temp.getDF()[j]);
			for(int j = 0; j < temp.getTTF().length; j++)
				writer.print("\t" + temp.getTTF()[j]);
			writer.println();
		}
		writer.close();
		return writer;
	}
	
	//Tokenizer.
	protected String[] Tokenizer(String source){
		String[] tokens = m_tokenizer.tokenize(source);
		return tokens;
	}
	
	//Normalize.
	protected String Normalize(String token){
		token = Normalizer.normalize(token, Normalizer.Form.NFKC);
		token = token.replaceAll("\\W+", "");
		token = token.toLowerCase();
		return token;
	}
	
	//Snowball Stemmer.
	protected String SnowballStemming(String token){
		m_stemmer.setCurrent(token);
		if(m_stemmer.stem())
			return m_stemmer.getCurrent();
		else
			return token;
	}
	
	//Given a long string, tokenize it, normalie it and stem it, return back the string array.
	protected String[] TokenizerNormalizeStemmer(String source){
		String[] tokens = Tokenizer(source); //Original tokens.
		//Normalize them and stem them.		
		for(int i = 0; i < tokens.length; i++)
			tokens[i] = SnowballStemming(Normalize(tokens[i]));
		
		int tokenLength = tokens.length, N = this.m_Ngram, NgramNo = 0;
		ArrayList<String> Ngrams = new ArrayList<String>();
		
		//Collect all the grams, Ngrams, N-1grams...
		while(N > 0){
			NgramNo = tokenLength - N + 1;
			for(int i = 0; i < NgramNo; i++){
				StringBuffer Ngram = new StringBuffer(128);
				if(tokens[i].equals("")){ i++;}
				else{
					for(int j = 0; j < N; j++){
						if (j == 0) Ngram.append(tokens[i + j]);		
						else {
							if(!tokens[i + j].equals("")){
								Ngram.append("-" + tokens[i+j]);
							} else { 
								i++;
								break;
							}
						}
					}
					Ngrams.add(Ngram.toString());
				}
				N--;
			}
			return Ngrams.toArray(new String[Ngrams.size()]);
		}
	}
	
	//Add one more token to the current vocabulary.
	protected void expandVocabulary(String token) {
		m_featureNames.add(token); // Add the new feature.
		m_featureNameIndex.put(token, (m_featureNames.size() - 1)); // set the index of the new feature.
		m_featureIndexName.put((m_featureNames.size() - 1), token);
	}
	
	//With a new feature added into the vocabulary, add the stat into stat arraylist.
	public void updateFeatureStat(String token) {
		this.m_featureStat.put(token, new _stat(this.m_classNo));
	}
		
	//Return corpus without parameter and feature selection.
	public _Corpus returnCorpus(String finalLocation)throws FileNotFoundException {
		this.m_corpus.setMasks(); // After collecting all the documents, shuffle all the documents' labels.
		this.SaveCVStat(finalLocation);
		return this.m_corpus;
	}
	
	//Give the option, which would be used as the method to calculate feature value and returned corpus, calculate the feature values.
	public void setFeatureValues(String fValue, int norm) {
		ArrayList<_Doc> docs = m_corpus.getCollection(); // Get the collection of all the documents.
		int N = docs.size();
		if (fValue.equals("TF")){
			for(int i = 0; i < docs.size(); i++){
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				for(_SparseFeature sf: sfs){
					double TF = sf.getValue() / temp.getTotalDocLength();// normalized TF
					sf.setValue(TF);
				}
			}
		} else if (fValue.equals("TFIDF")) {
			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureIndexName.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double TF = sf.getValue() / temp.getTotalDocLength();// normalized TF
					double DF = Utils.sumOfArray(stat.getDF());
					double TFIDF = TF * Math.log((N + 1) / DF);
					sf.setValue(TFIDF);
				}
			}
		} else if (fValue.equals("BM25")) {
			double k1 = 1.5; // [1.2, 2]
			double b = 10; // (0, 1000]
			// Iterate all the documents to get the average document length.
			double navg = 0;
			for (int k = 0; k < N; k++)
				navg += docs.get(k).getTotalDocLength();
			navg = navg / N;

			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureIndexName.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double n = temp.getTotalDocLength();
					double BM25 = Math.log((N - DF + 0.5) / (DF + 0.5)) * TF * (k1 + 1) / (k1 * (1 - b + b * n / navg) + TF);
					sf.setValue(BM25);
				}
			}
		} else if (fValue.equals("PLN")) {
			double s = 0.5; // [0, 1]
			// Iterate all the documents to get the average document length.
			double navg = 0;
			for (int k = 0; k < N; k++)
				navg += docs.get(k).getTotalDocLength();
			navg = navg / N;

			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureIndexName.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double n = temp.getTotalDocLength();
					double PLN = (1 + Math.log(1 + Math.log(TF)) / (1 - s + s * n / navg)) * Math.log((N + 1) / DF);
					sf.setValue(PLN);
				}
			}
		} else {
			//The default value is just keeping the raw count of every feature.
			System.out.println("No feature value is set, keep the raw count of every feature.");
		}
		if (norm == 1){
			for(_Doc d:docs)			
				Utils.L1Normalization(d.getSparse());
		} else if(norm == 2){
			for(_Doc d:docs)			
				Utils.L2Normalization(d.getSparse());
		} else {
			System.out.println("No normalizaiton is adopted here or wrong parameters!!");
		}
		
	}
	
	//Set the counts of every feature with respect to the collected class number.
	public void setFeatureConfiguration() {
		// Initialize the counts of every feature.
		for (String featureName : this.m_featureStat.keySet()) {
			this.m_featureStat.get(featureName).initCount(this.m_classNo);		
			this.m_featureStat.get(featureName).setCounts(this.m_classMemberNo);
		}
	}
	
	//Select the features and store them in a file.
	public void featureSelection(String location, double startProb, double endProb, int threshold) throws FileNotFoundException {
		FeatureSelector selector = new FeatureSelector(startProb, endProb);
		this.m_corpus.setMasks(); // After collecting all the documents, shuffle all the documents' labels.
		this.setFeatureConfiguration(); // Construct the table for features.

		if (this.m_isFetureSelected) {
			System.out.println("*******************************************************************");
			if (this.featureSelection.equals("DF")) {
				this.m_featureNames = selector.DF(this.m_featureStat, threshold);
			} else if (this.featureSelection.equals("IG")) {
				this.m_featureNames = selector.IG(this.m_featureStat, this.m_classMemberNo, threshold);
			} else if (this.featureSelection.equals("MI")) {
				this.m_featureNames = selector.MI(this.m_featureStat, this.m_classMemberNo, threshold);
			} else if (this.featureSelection.equals("CHI")) {
				this.m_featureNames = selector.CHI(this.m_featureStat, this.m_classMemberNo, threshold);
			}
		}
		this.SaveCV(location); // Save all the features and probabilities we get after analyzing.
		System.out.println(this.m_featureNames.size() + " features are selected!");
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
	
	//Return the number of features.
	public int getFeatureSize(){
		return this.m_featureNames.size();
	}
	
	//Sort the documents.
	public void setTimeFeatures(int window){
		if (window<1)
			return;
		
		//Sort the documents according to time stamps.
		ArrayList<_Doc> docs = m_corpus.getCollection();
		
		Collections.sort(docs, new Comparator<_Doc>(){
			public int compare(_Doc d1, _Doc d2){
				if(d1.getTimeStamp() == d2.getTimeStamp())
					return 0;
					return d1.getTimeStamp() < d2.getTimeStamp() ? -1 : 1;
			}
		});		
		
		/************************time series analysis***************************/
		double norm = 1.0 / m_classMemberNo.length;
		for(int i = 0; i < docs.size(); i++){
			_Doc doc = docs.get(i);
			if(m_preDocs.size() < window){
				m_preDocs.add(doc);
				this.m_corpus.removeDoc(i);
				this.m_classMemberNo[doc.getYLabel()]--;
				i--;
			}
			else{
				doc.createSpVctWithTime(m_preDocs, this.m_featureNames.size(), norm);
				m_preDocs.remove();
				m_preDocs.add(doc);
			}
		}
		
		System.out.println("Time-series feature set!");
	}
}
