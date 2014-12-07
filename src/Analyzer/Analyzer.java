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
	protected boolean m_timeFlag;
	
	//added by Hongning to manage feature vocabulary
	/* Indicate if we can allow new features.After loading the CV file, the flag is set to true, 
	 * which means no new features will be allowed.*/
	protected boolean m_isCVLoaded = false; 
	
	/* Indicate if the user has specified a feature selection method.
	 * If the user do not provides a feature selection method, then all terms will be chosen as CV. 
	 * So the default value is true.*/
	protected boolean m_isFetureSelected = false; 
	private int m_Ngram; 
	
	protected ArrayList<String> m_featureNames; //ArrayList for features
	protected HashMap<String, Integer> m_featureNameIndex;//key: content of the feature; value: the index of the feature
	protected HashMap<Integer, String> m_featureIndexName;//value: the index of the feature; key: content of the feature; 
	protected HashMap<String, _stat> m_featureStat; //Key: feature Name; value: the stat of the feature
	protected String featureSelection = "DF";
	
	public Analyzer(String tokenModel, int classNo) throws InvalidFormatException, FileNotFoundException, IOException{
		this.m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		this.m_classNo = classNo;
		this.m_classMemberNo = new int[classNo];
		
		this.m_corpus = new _Corpus();
		this.m_stemmer = new englishStemmer();
		this.m_featureNames = new ArrayList<String>();
		this.m_featureNameIndex = new HashMap<String, Integer>();//key: content of the feature; value: the index of the feature
		this.m_featureIndexName = new HashMap<Integer, String>();//value: content of the feature; key: the index of the feature
		this.m_featureStat = new HashMap<String, _stat>();
		this.m_timeFlag = false;
		this.m_Ngram = 1;
	}	
	
	public Analyzer(String tokenModel, int classNo, int Ngram) throws InvalidFormatException, FileNotFoundException, IOException{
		this.m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		this.m_classNo = classNo;
		this.m_classMemberNo = new int[classNo];
		
		this.m_corpus = new _Corpus();
		this.m_stemmer = new englishStemmer();
		this.m_featureNames = new ArrayList<String>();
		this.m_featureNameIndex = new HashMap<String, Integer>();//key: content of the feature; value: the index of the feature
		this.m_featureIndexName = new HashMap<Integer, String>();//value: content of the feature; key: the index of the feature
		this.m_featureStat = new HashMap<String, _stat>();
		this.m_timeFlag = false;
		this.m_Ngram = Ngram;
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
	
	//Save all the features and feature stat into a file.
	protected PrintWriter SaveCVStat(String finalLocation) throws FileNotFoundException{
		//File file = new File(path);
		PrintWriter writer = new PrintWriter(new File(finalLocation));
		for(int i = 0; i < this.m_featureNames.size(); i++){
			writer.print("\nfeature: " + this.m_featureNames.get(i));
			_stat temp = this.m_featureStat.get(this.m_featureNames.get(i));
			for(int j = 0; j < temp.getDF().length; j++){
				writer.print("\tDF[" + j + "]:" + temp.getDF()[j]);
			}
			for(int j = 0; j < temp.getTTF().length; j++){
				writer.print("\tTTF[" + j + "]:" + temp.getTTF()[j]);
			}
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
				for(int j = 0; j < N; j++){
					if (j==0) 
						Ngram.append(tokens[i+j]);
					else 
						Ngram.append("-" + tokens[i+j]);
				}
				Ngrams.add(Ngram.toString());
			}
			N--;
		}
		return Ngrams.toArray(new String[Ngrams.size()]);
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
	public _Corpus setFeatureValues(_Corpus c, String fValue) {
		HashMap<String, _stat> featureStat = this.m_featureStat;
		HashMap<Integer, String> featureIndexName = this.m_featureIndexName;
		ArrayList<_Doc> docs = c.getCollection(); // Get the collection of all the documents.
		int N = docs.size();
		if (fValue.equals("TFIDF")) {
			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				for (_SparseFeature sf : sfs) {
					String featureName = featureIndexName.get(sf.getIndex());
					_stat stat = featureStat.get(featureName);
					double TF = sf.getValue() / temp.getTotalDocLength();// normalized
																			// TF
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
					String featureName = featureIndexName.get(sf.getIndex());
					_stat stat = featureStat.get(featureName);
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
					String featureName = featureIndexName.get(sf.getIndex());
					_stat stat = featureStat.get(featureName);
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double n = temp.getTotalDocLength();
					double PLN = (1 + Math.log(1 + Math.log(TF)) / (1 - s + s * n / navg)) * Math.log((N + 1) / DF);
					sf.setValue(PLN);
				}
			}
		} else
			return c; // If no feature value is selected, then the default is TF.
		return c;
	}
}
