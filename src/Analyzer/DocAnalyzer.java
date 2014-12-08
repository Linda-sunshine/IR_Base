package Analyzer;

import java.io.BufferedReader;
import java.io.File;
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
	//Analyze the document as usual.
	public void AnalyzeDoc(_Doc doc) {
		try {
			String[] tokens = TokenizerNormalizeStemmer(doc.getSource());// Three-step analysis.
			doc.setTotalLength(tokens.length); // set the length of the document.
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
							this.m_featureStat.get(token).addOneTTF(doc.getYLabel());
						} else {
							spVct.put(index, 1.0);
							this.m_featureStat.get(token).addOneDF(doc.getYLabel());
							this.m_featureStat.get(token).addOneTTF(doc.getYLabel());
						}
					} else {
						// indicate we allow the analyzer to dynamically expand the feature vocabulary
						expandVocabulary(token);// update the m_featureNames.
						updateFeatureStat(token);
						index = m_featureNameIndex.get(token);
						spVct.put(index, 1.0);
						this.m_featureStat.get(token).addOneDF(doc.getYLabel());
						this.m_featureStat.get(token).addOneTTF(doc.getYLabel());
					}
					// CV is loaded.
				} else if (m_featureNameIndex.containsKey(token)) {
					index = m_featureNameIndex.get(token);
					if (spVct.containsKey(index)) {
						value = spVct.get(index) + 1;
						spVct.put(index, value);
					} else {
						spVct.put(index, 1.0);
						this.m_featureStat.get(token).addOneDF(doc.getYLabel());
					}
					this.m_featureStat.get(token).addOneTTF(doc.getYLabel());
				}
				// if the token is not in the vocabulary, nothing to do.
			}
			doc.createSpVct(spVct);
			doc.L2Normalization(doc.getSparse());
			m_corpus.addDoc(doc);
			this.m_corpus.sizeAddOne();
			this.m_classMemberNo[doc.getYLabel()]++;
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}	

