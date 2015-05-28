package Analyzer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Set;

import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import structures._Doc;
import structures._SparseFeature;
import structures._stat;
import utils.Utils;

public class DocAnalyzer extends Analyzer {
	protected Tokenizer m_tokenizer;
	protected SnowballStemmer m_stemmer;
	protected SentenceDetectorME m_stnDetector;
	Set<String> m_stopwords;
	 
	protected boolean m_releaseContent;
	
	//Constructor with ngram and fValue.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException{
		super(classNo, threshold);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();
		m_stnDetector = null; // indicating we don't need sentence splitting
		
		m_Ngram = Ngram;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
	}
	
	//Constructor with ngram and fValue and sentence check.
	public DocAnalyzer(String tokenModel, String stnModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException{
		super(classNo, threshold);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();
		
		if (stnModel!=null)
			m_stnDetector = new SentenceDetectorME(new SentenceModel(new FileInputStream(stnModel)));
		else
			m_stnDetector = null;
		
		m_Ngram = Ngram;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
	}
	
	public void setReleaseContent(boolean release) {
		m_releaseContent = release;
	}
	
	public void LoadStopwords(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) {
				line = SnowballStemming(Normalize(line));
				if (!line.isEmpty())
					m_stopwords.add(line);
			}
			reader.close();
			System.out.format("Loading %d stopwords from %s\n", m_stopwords.size(), filename);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
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
		
		if (Utils.isNumber(token))
			return "NUM";
		else
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
	
	protected boolean isLegit(String token) {
		return !token.isEmpty() 
			&& !m_stopwords.contains(token)
			&& token.length()>1
			&& token.length()<20;
	}
	
	protected boolean isBoundary(String token) {
		return token.isEmpty();//is this a good checking condition?
	}
	
	//Given a long string, tokenize it, normalie it and stem it, return back the string array.
	protected String[] TokenizerNormalizeStemmer(String source){
		String[] tokens = Tokenizer(source); //Original tokens.
		//Normalize them and stem them.		
		for(int i = 0; i < tokens.length; i++)
			tokens[i] = SnowballStemming(Normalize(tokens[i]));
		
		LinkedList<String> Ngrams = new LinkedList<String>();
		int tokenLength = tokens.length, N = m_Ngram;		
		for(int i=0; i<tokenLength; i++) {
			String token = tokens[i];
			boolean legit = isLegit(token);
			if (legit)
				Ngrams.add(token);//unigram
			
			//N to 2 grams
			if (!isBoundary(token)) {
				for(int j=i-1; j>=Math.max(0, i-N+1); j--) {	
					if (isBoundary(tokens[j]))
						break;//touch the boundary
					
					token = tokens[j] + "-" + token;
					legit |= isLegit(tokens[j]);
					if (legit)//at least one of them is legitimate
						Ngrams.add(token);
				}
			}
		}
		
		return Ngrams.toArray(new String[Ngrams.size()]);
	}

	//Load a movie review document and analyze it.
	//this is only specified for this type of review documents
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
			}else if(filename.contains("neg")){
				AnalyzeDoc(new _Doc(m_corpus.getSize(), buffer.toString(), 1));
			}
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
		}
	}
	
	protected HashMap<Integer, Double> constructSpVct(String[] tokens, int y) {
		int index = 0;
		double value = 0;
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
		
		for (String token : tokens) {
			// CV is not loaded, take all the tokens as features.
			if (!m_isCVLoaded) {
				if (m_featureNameIndex.containsKey(token)) {
					index = m_featureNameIndex.get(token);
					if (spVct.containsKey(index)) {
						value = spVct.get(index) + 1;
						spVct.put(index, value);
					} else {
						spVct.put(index, 1.0);
						m_featureStat.get(token).addOneDF(y);
					}
				} else {// indicate we allow the analyzer to dynamically expand the feature vocabulary
					expandVocabulary(token);// update the m_featureNames.
					index = m_featureNameIndex.get(token);
					spVct.put(index, 1.0);
					m_featureStat.get(token).addOneDF(y);
				}
				m_featureStat.get(token).addOneTTF(y);
			} else if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
				index = m_featureNameIndex.get(token);
				if (spVct.containsKey(index)) {
					value = spVct.get(index) + 1;
					spVct.put(index, value);
				} else {
					spVct.put(index, 1.0);
					m_featureStat.get(token).addOneDF(y);
				}
				m_featureStat.get(token).addOneTTF(y);
			}
			// if the token is not in the vocabulary, nothing to do.
		}
		
		return spVct;
	}
	
	public void rollBack(HashMap<Integer, Double> spVct, int y){
		if (!m_isCVLoaded) {
			setFeatureIndexName();//Get the current index-feature lookup table.
			for(int index: spVct.keySet()){
				String token = m_featureIndexName.get(index);
				_stat stat = m_featureStat.get(token);
				if(Utils.sumOfArray(stat.getDF())==1){//If the feature is the first time to show in feature set.
					m_featureNameIndex.remove(index);
					m_featureStat.remove(token);
					m_featureNames.remove(token);
				}
				else{//If the feature is not the first time to show in feature set.
					m_featureStat.get(token).minusOneDF(y);
					m_featureStat.get(token).minusNTTF(y, spVct.get(index));
				}
			}
		} else{//If CV is loaded, we can minus the DF and TTF directly.
//			setFeatureIndexName(); //Add this step to LoadCV to avoid repetitive operations.
			for(int index: spVct.keySet()){
				String token = m_featureIndexName.get(index);
				m_featureStat.get(token).minusOneDF(y);
				m_featureStat.get(token).minusNTTF(y, spVct.get(index));
			}
		}
	}
	
	/*Analyze a document and add the analyzed document back to corpus.
	 *In the case CV is not loaded, we need two if loops to check.
	 * The first is if the term is in the vocabulary.***I forgot to check this one!
	 * The second is if the term is in the sparseVector.
	 * In the case CV is loaded, we still need two if loops to check.*/
	protected boolean AnalyzeDoc(_Doc doc) {
//		if(doc.getYLabel() == 1 && m_classMemberNo[1] >= 3185)
//			return true;
//		else{
		String[] tokens = TokenizerNormalizeStemmer(doc.getSource());// Three-step analysis.
		int y = doc.getYLabel();
		// Construct the sparse vector.
		HashMap<Integer, Double> spVct = constructSpVct(tokens, y);
		
		if (spVct.size()>=m_lengthThreshold) {//temporary code for debugging purpose
			doc.createSpVct(spVct);
			m_corpus.addDoc(doc);
			m_classMemberNo[y]++;
			if (m_releaseContent)
				doc.clearSource();
			return true;
		} else{
			rollBack(spVct, y);
			return false;
		}
//		}
	}
	
	// adding sentence splitting function, modified for HTMM
	protected boolean AnalyzeDocWithStnSplit(_Doc doc) {
//		if(doc.getYLabel() == 1 && m_classMemberNo[1] >= 3185)
//			return true;
//		else{
			String[] sentences = m_stnDetector.sentDetect(doc.getSource());
			HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
			ArrayList<_SparseFeature[]> stnList = new ArrayList<_SparseFeature[]>(); // to avoid empty sentences
			int y = doc.getYLabel();
			
			for(String sentence : sentences) {
				String[] tokens = TokenizerNormalizeStemmer(sentence);// Three-step analysis.			
				HashMap<Integer, Double> sentence_vector = constructSpVct(tokens, y);			
				if (sentence_vector.size()>0) {//avoid empty sentence
					stnList.add(Utils.createSpVct(sentence_vector));
					Utils.mergeVectors(sentence_vector, spVct);
				}
			} // End For loop for sentence	
		
			//the document should be long enough
			if (spVct.size()>=m_lengthThreshold && stnList.size()>1) { 
				doc.createSpVct(spVct);
				doc.setSentences(stnList);
				m_corpus.addDoc(doc);
				m_classMemberNo[y]++;
				if (m_releaseContent)
					doc.clearSource();
				return true;
			} else{
				rollBack(spVct, y);
				return false;
			}
		}
//	}
}	

