package Analyzer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.Normalizer;
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
import utils.Utils;

public class DocAnalyzer extends Analyzer {
	
	protected int m_lengthThreshold;

	protected Tokenizer m_tokenizer;
	protected SnowballStemmer m_stemmer;
	protected SentenceDetectorME m_sentencedetector;
	Set<String> m_stopwords;
	
	/* Indicate if we can allow new features.After loading the CV file, the flag is set to true, 
	 * which means no new features will be allowed.*/
	protected boolean m_isCVLoaded; 
	protected boolean m_sentence_check = false;
	
	protected boolean m_releaseContent;
	
	//Constructor.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();
		
		m_Ngram = 1;
		m_lengthThreshold = 5;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
	}	
	
	//Constructor with ngram and fValue.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();
		
		m_Ngram = Ngram;
		m_lengthThreshold = threshold;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
	}
	
	//Constructor with ngram and fValue and sentence check.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, boolean sentence_check) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();
		m_sentencedetector = new SentenceDetectorME(new SentenceModel(new FileInputStream("./data/Model/en-sent.bin")));
		
		m_Ngram = Ngram;
		m_lengthThreshold = threshold;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
		m_sentence_check = true;
	}
	
	public void setReleaseContent(boolean release) {
		m_releaseContent = release;
	}
	
	//Load the features from a file and store them in the m_featurNames.@added by Lin.
	protected boolean LoadCV(String filename) {
		if (filename==null || filename.isEmpty())
			return false;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			while ((line = reader.readLine()) != null) {
				if (line.startsWith("#")){
					if (line.startsWith("#NGram")) {//has to be decoded
						int pos = line.indexOf(':');
						m_Ngram = Integer.valueOf(line.substring(pos+1));
					}
						
				} else 
					expandVocabulary(line);
			}
			reader.close();
			
			System.out.format("%d feature words loaded from %s...\n", m_featureNames.size(), filename);
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			return false;
		}
		
		return true; // if loading is successful
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
			}else if(filename.contains("neg")){
				AnalyzeDoc(new _Doc(m_corpus.getSize(), buffer.toString(), 1));
			}
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
		}
	}
	
	//Given a long string, return a set of sentences using .?! as delimiter
	// added by Md. Mustafizur Rahman for HTMM Topic Modelling 
	protected String[] findSentence(String source){
		String regexp = "[.?!]+"; 
	    String [] sentences;
	    sentences = source.split(regexp);
	    return sentences;
	}
	

	/*Analyze a document and add the analyzed document back to corpus.	
	 *In the case CV is not loaded, we need two if loops to check. 
	 * The first is if the term is in the vocabulary.***I forgot to check this one!
	 * The second is if the term is in the sparseVector.
	 * In the case CV is loaded, we still need two if loops to check.*/
	//Analyze the document as usual.
	//modified for HTMM
	protected boolean AnalyzeDoc(_Doc doc, boolean sentence_check) {
		try {
			String[] sentences = m_sentencedetector.sentDetect(doc.getSource());
			HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
			doc.set_number_of_sentences(sentences.length);
			
			int sentence_index = 0;
			for(String sentence : sentences) {
				String[] tokens = TokenizerNormalizeStemmer(sentence);// Three-step analysis.			
				int index = 0;
				double value = 0;
				HashMap<Integer, Double> sentence_vector = new HashMap<Integer, Double>(); 
				
				// Construct the sparse vector.
				for (String token : tokens) {
					// CV is not loaded, take all the tokens as features.
					if (!m_isCVLoaded) {
						if (m_featureNameIndex.containsKey(token)) {
							index = m_featureNameIndex.get(token);
							if (spVct.containsKey(index)) {
								value = spVct.get(index) + 1;
								spVct.put(index, value);
								if(sentence_vector.containsKey(index)){
									value = sentence_vector.get(index) + 1;
									sentence_vector.put(index, value);
								} else {
									sentence_vector.put(index, 1.0);
								}
													
							} else {
								spVct.put(index, 1.0);
								sentence_vector.put(index, 1.0);
								m_featureStat.get(token).addOneDF(doc.getYLabel());
							}
						} else {// indicate we allow the analyzer to dynamically expand the feature vocabulary
							expandVocabulary(token);// update the m_featureNames.
							index = m_featureNameIndex.get(token);
							spVct.put(index, 1.0);
							sentence_vector.put(index, 1.0);
					    	m_featureStat.get(token).addOneDF(doc.getYLabel());
						}
		
						m_featureStat.get(token).addOneTTF(doc.getYLabel());
					} else if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
						index = m_featureNameIndex.get(token);
						if (spVct.containsKey(index)) {
							value = spVct.get(index) + 1;
							spVct.put(index, value);
							if(sentence_vector.containsKey(index)){
								value = sentence_vector.get(index) + 1;
								sentence_vector.put(index, value);
							} else {
								sentence_vector.put(index, 1.0);
							}
						} else {
							spVct.put(index, 1.0);
							sentence_vector.put(index, 1.0);
						
							m_featureStat.get(token).addOneDF(doc.getYLabel());
						}
						m_featureStat.get(token).addOneTTF(doc.getYLabel());
					}
				// if the token is not in the vocabulary, nothing to do.
				}// End for loop for token
				doc.createSentenceVct(sentence_vector, sentence_index);	
				sentence_index++;
			} // End For loop for sentence	
		
			if (spVct.size()>=m_lengthThreshold) {//temporary code for debugging purpose 
				doc.createSpVct(spVct);
				m_corpus.addDoc(doc);
				m_classMemberNo[doc.getYLabel()]++;
				
				if (m_releaseContent)
					doc.clearSource();
				return true;
			} else
				return false;
			
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
	}
	
	
	/*Analyze a document and add the analyzed document back to corpus.
	 *In the case CV is not loaded, we need two if loops to check.
	 * The first is if the term is in the vocabulary.***I forgot to check this one!
	 * The second is if the term is in the sparseVector.
	 * In the case CV is loaded, we still need two if loops to check.*/
	//Analyze the document as usual.
	protected boolean AnalyzeDoc(_Doc doc) {
		try {
			String[] tokens = TokenizerNormalizeStemmer(doc.getSource());// Three-step analysis.
			HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
			int index = 0;
			double value = 0;
			// Construct the sparse vector.
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
							m_featureStat.get(token).addOneDF(doc.getYLabel());
						}
					} else {// indicate we allow the analyzer to dynamically expand the feature vocabulary
						expandVocabulary(token);// update the m_featureNames.
						index = m_featureNameIndex.get(token);
						spVct.put(index, 1.0);
						m_featureStat.get(token).addOneDF(doc.getYLabel());
					}
					m_featureStat.get(token).addOneTTF(doc.getYLabel());
				} else if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
					index = m_featureNameIndex.get(token);
					if (spVct.containsKey(index)) {
						value = spVct.get(index) + 1;
						spVct.put(index, value);
					} else {
						spVct.put(index, 1.0);
						m_featureStat.get(token).addOneDF(doc.getYLabel());
					}
					m_featureStat.get(token).addOneTTF(doc.getYLabel());
				}
				// if the token is not in the vocabulary, nothing to do.
			}
			if (spVct.size()>=m_lengthThreshold) {//temporary code for debugging purpose
				doc.createSpVct(spVct);
				m_corpus.addDoc(doc);
				m_classMemberNo[doc.getYLabel()]++;
				if (m_releaseContent)
					doc.clearSource();
				return true;
			} else
				return false;
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
	}
	
	
}	

