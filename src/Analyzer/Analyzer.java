package Analyzer;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import structures._Corpus;
import structures._stat;


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
	
	protected ArrayList<String> m_featureNames; //ArrayList for features
	protected HashMap<String, Integer> m_featureNameIndex;//key: content of the feature; value: the index of the feature
	protected HashMap<Integer, String> m_featureIndexName;//value: the index of the feature; key: content of the feature; 
	protected HashMap<String, _stat> m_featureStat; //Key: feature Name; value: the stat of the feature
	//protected FeatureSelection m_selector; //An instance of selecting different features.
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
	}	
	//abstract protected _Corpus returnCorpus(String location, String fs) throws FileNotFoundException;
}
