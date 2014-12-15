package Analyzer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._stat;

public class DocAnalyzer extends Analyzer {

	//Constructor.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, String fs) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo);
		if(providedCV != null)
			LoadCV(providedCV);
		if(fs != null){
			m_isFetureSelected = true;
			featureSelection = fs;
		}	
	}	
	
	//Constructor with ngram and fValue.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, String fs, int Ngram) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo, Ngram);
		if(!providedCV.equals(""))
			LoadCV(providedCV);
		if(!fs.equals("")){
			m_isFetureSelected = true;
			featureSelection = fs;
		}
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
				m_featureNames.add(line);
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
		for(String f: m_featureNames){
			m_featureNameIndex.put(f, count);
			m_featureIndexName.put(count, f);
			m_featureStat.put(f, new _stat(m_classNo));
			count++;
		}
		return true; // if loading is successful
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
				m_classMemberNo[0]++;
			}else if(filename.contains("neg")){
				AnalyzeDoc(new _Doc(m_corpus.getSize(), buffer.toString(), 1));
				m_classMemberNo[1]++;
			}
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
		}
	}

	/*Analyze a document and add the analyzed document back to corpus.	
	 *In the case CV is not loaded, we need two if loops to check. 
	 * The first is if the term is in the vocabulary.***I forgot to check this one!
	 * The second is if the term is in the sparseVector.
	 * In the case CV is loaded, we still need two if loops to check.*/
	//Analyze the document as usual.
	public void AnalyzeDoc(_Doc doc) {
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
						updateFeatureStat(token);
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
			
			if (spVct.size()>=5) {//temporary code for debugging purpose 
				doc.createSpVct(spVct);
				m_corpus.addDoc(doc);
				m_classMemberNo[doc.getYLabel()]++;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}	

