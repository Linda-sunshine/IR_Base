package topicmodels;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;

import com.sun.xml.internal.fastinfoset.algorithm.BuiltInEncodingAlgorithm.WordListener;

import Analyzer.ParentChildAnalyzer;
import structures._APPQuery;
import structures._ChildDoc;
import structures._ChildDoc4APP;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import structures._Word;
import utils.Utils;

public class APPLanguageModel extends languageModelBaseLine{
	
	HashMap<Integer, Double> m_wordSstatInDescription;
	HashMap<Integer, Double> m_wordSstatInReview;
	
	ArrayList<_APPQuery> m_APPQueries;
	
	double m_MuDescription;
	double m_MuReview;
	double m_allWordFrequencyInDescription;
	double m_allWordFrequencyInReview;
	
	double m_eta;
	
	public APPLanguageModel(_Corpus c, double mu_description, double mu_review, double eta, ArrayList<_APPQuery> appQueryList) {
		super(c, mu_description);
		
		m_allWordFrequencyInDescription = 0;
		m_allWordFrequencyInReview = 0;
		
		m_MuDescription = mu_description;
		m_MuReview = mu_review;
		
		m_wordSstatInDescription = new HashMap<Integer, Double>();
		m_wordSstatInReview = new HashMap<Integer, Double>();
		
		m_eta = eta;
		m_APPQueries = appQueryList;
	}
	
	protected void generateReferenceModel(){
		m_allWordFrequencyInDescription = 0;
		
		for(_Doc d:m_corpus.getCollection()){
			if(d instanceof _ParentDoc){
				_ParentDoc pDoc = (_ParentDoc)d;
				_SparseFeature[] fv = pDoc.getSparse();
				
				for(int i=0; i<fv.length; i++){
					int wid = fv[i].getIndex();
					double val = fv[i].getValue();
					
					m_allWordFrequencyInDescription += val;
					if(m_wordSstatInDescription.containsKey(wid)){
						double oldVal = m_wordSstatInDescription.get(wid);
						m_wordSstatInDescription.put(wid, oldVal+val);
					}else{
						m_wordSstatInDescription.put(wid, val);
					}
				}
			}else if(d instanceof _ChildDoc){
				_ChildDoc cDoc = (_ChildDoc)d;
				
				_SparseFeature[] fv = cDoc.getSparse();
				
				for(int i=0; i<fv.length; i++){
					int wid = fv[i].getIndex();
					double val = fv[i].getValue();
					
					m_allWordFrequencyInReview += val;
					if(m_wordSstatInReview.containsKey(wid)){
						double oldVal = m_wordSstatInReview.get(wid);
						m_wordSstatInReview.put(wid, oldVal+val);
	
					}else{
						m_wordSstatInReview.put(wid, val);
					}
				}
			}
			
		}
		
		for(int wid:m_wordSstatInDescription.keySet()){
			double val = m_wordSstatInDescription.get(wid);
			double prob = val/m_allWordFrequencyInDescription;
			
			m_wordSstatInDescription.put(wid, prob);
		}
		
		for(int wid:m_wordSstatInReview.keySet()){
			double val = m_wordSstatInReview.get(wid);
			double prob = val/m_allWordFrequencyInReview;
			
			m_wordSstatInReview.put(wid, prob);
		}
	}

	protected double getReferenceProbInDscription(int wid){
		return m_wordSstatInDescription.get(wid);
	}
	
	protected double getReferenceProbInReview(int wid){
		return m_wordSstatInReview.get(wid);
	}
	
	protected void printTopAPP4Query(String filePrefix){
		String topAPP4QueryFile = filePrefix + "/topAPP4Query.txt";
		
		try{
			PrintWriter pw = new PrintWriter(new File(topAPP4QueryFile));
			
			for(_APPQuery appQuery:m_APPQueries){
				pw.print(appQuery.getQueryID()+"\t");
				
				for(_Doc d:m_corpus.getCollection()){
					if(d instanceof _ParentDoc){
						_ParentDoc pDoc = (_ParentDoc)d;
						
						double likelihood = rankAPP4QueryByLM(appQuery, pDoc);
						pw.print(d.getTitle()+":"+likelihood);
						pw.print("\t");
					}
				}
				
				pw.println();
			}
			pw.flush();
			pw.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	protected double rankAPP4QueryByLM(_APPQuery appQuery, _ParentDoc pDoc){
		double queryLikelihood = 0.0;		
		
		_SparseFeature[] pFv = pDoc.getSparse();
		double descriptionLen = pDoc.getTotalDocLength();
		
		System.out.println("child num\t"+pDoc.m_childDocs.size());
		_ChildDoc4APP cDoc4app = null;
		for(_ChildDoc cDoc:pDoc.m_childDocs)
			cDoc4app = (_ChildDoc4APP)cDoc;
		
		double reviewLen = 0;
		_SparseFeature[] cFv = null;
		if(cDoc4app!=null){
			reviewLen = cDoc4app.getTotalDocLength();
		
			cFv = cDoc4app.getSparse();
		}
		
		double alphaDescription = m_MuDescription/(m_MuDescription+descriptionLen);
		double alphaReview = m_MuReview/(m_MuReview+reviewLen);
		
		for(_Word w:appQuery.getWords()){
			double wordLikelihood = 0;
			int wid = w.getIndex();
			
			double featureDescriptionVal = 0;
			int featureIndex = Utils.indexOf(pFv, wid);
			if(featureIndex != -1){
				featureDescriptionVal = pFv[featureIndex].getValue();
			}
			
			double smoothingProbInDescription = 0;
			smoothingProbInDescription = (1-alphaDescription)*featureDescriptionVal/descriptionLen;
			smoothingProbInDescription += alphaDescription*getReferenceProbInDscription(wid);
			
			double smoothingProbInReview = 0;
			if(cDoc4app!=null){
				double featureReviewVal = 0;
				featureIndex = Utils.indexOf(cFv, wid);
				if(featureIndex != -1){
					featureReviewVal = cFv[featureIndex].getValue();
				}
			
				smoothingProbInReview = (1-alphaReview)*featureReviewVal/reviewLen;
				smoothingProbInReview += alphaReview*getReferenceProbInReview(wid);
			}
			
			wordLikelihood = (1-m_eta)*smoothingProbInDescription+(m_eta)*smoothingProbInReview;
			queryLikelihood += Math.log(wordLikelihood);
		}
		
		return queryLikelihood;
	}
	
	public static void main(String[] args) throws IOException, ParseException {	
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram. 
		// The way of calculating the feature value, which can also be "TFIDF",
		// "BM25"
		String featureValue = "BM25";
		int norm = 0;//The way of normalization.(only 1 and 2)
		int lengthThreshold = 5; //Document length threshold
		int minimunNumberofSentence = 2; // each document should have at least 2 sentences
		
		/*****parameters for the two-topic topic model*****/
		String topicmodel = "languageModel"; // 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTSM, ParentChild_Gibbs
	
		String category = "tablet";
		int number_of_topics = 20;
		boolean loadNewEggInTrain = true; // false means in training there is no reviews from NewEgg
		boolean setRandomFold = false; // false means no shuffling and true means shuffling
		int loadAspectSentiPrior = 0; // 0 means nothing loaded as prior; 1 = load both senti and aspect; 2 means load only aspect 
		
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3;//these two parameters must be larger than 1!!!
		double converge = 1e-9, lambda = 0.9; // negative converge means do not need to check likelihood convergency
		int varIter = 10;
		double varConverge = 1e-5;
		int topK = 20, number_of_iteration = 50, crossV = 1;
		int gibbs_iteration = 2000, gibbs_lag = 50;
		gibbs_iteration = 4;
		gibbs_lag = 2;
		double burnIn = 0.4;
		boolean display = true, sentence = false;
		
		// most popular items under each category from Amazon
		// needed for docSummary
		String tabletProductList[] = {"B008GFRDL0"};
		String cameraProductList[] = {"B005IHAIMA"};
		String phoneProductList[] = {"B00COYOAYW"};
		String tvProductList[] = {"B0074FGLUM"};
		
		/*****The parameters used in loading files.*****/
		String amazonFolder = "./data/amazon/tablet/topicmodel";
		String newEggFolder = "./data/NewEgg";
		String articleType = "Tech";
		articleType = "APP";
//		articleType = "GadgetsArticles";
		String articleFolder = String.format("./data/ParentChildTopicModel/%sArticles", articleType);
		String commentFolder = String.format("./data/ParentChildTopicModel/%sComments", articleType);
		
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = null;
		String posModel = null;
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM") || topicmodel.equals("HTSM") || topicmodel.equals("LRHTSM"))
		{
			stnModel = "./data/Model/en-sent.bin"; //Sentence model.
			posModel = "./data/Model/en-pos-maxent.bin"; // POS model.
			sentence = true;
		}
		
		String fvFile = String.format("./data/Features/fv_%dgram_topicmodel_%s.txt", Ngram, articleType);
		//String fvFile = String.format("./data/Features/fv_%dgram_topicmodel.txt", Ngram);
		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_topicmodel.txt", Ngram);
	
		String aspectList = "./data/Model/aspect_"+ category + ".txt";
		String aspectSentiList = "./data/Model/aspect_sentiment_"+ category + ".txt";
		
		String pathToPosWords = "./data/Model/SentiWordsPos.txt";
		String pathToNegWords = "./data/Model/SentiWordsNeg.txt";
		String pathToNegationWords = "./data/Model/negation_words.txt";
		String pathToSentiWordNet = "./data/Model/SentiWordNet_3.0.0_20130122.txt";

		File rootFolder = new File("./data/results");
		if(!rootFolder.exists()){
			System.out.println("creating root directory"+rootFolder);
			rootFolder.mkdir();
		}
		
		Calendar today = Calendar.getInstance();
		String filePrefix = String.format("./data/results/%s-%s-%s%s-%s", today.get(Calendar.MONTH), today.get(Calendar.DAY_OF_MONTH), 
						today.get(Calendar.HOUR_OF_DAY), today.get(Calendar.MINUTE), topicmodel);
		
		File resultFolder = new File(filePrefix);
		if (!resultFolder.exists()) {
			System.out.println("creating directory" + resultFolder);
			resultFolder.mkdir();
		}
		
		String infoFilePath = filePrefix + "/Information.txt";
		////store top k words distribution over topic
		String topWordPath = filePrefix + "/topWords.txt";
		
		/*****Parameters in feature selection.*****/
		String stopwords = "./data/Model/stopwords.dat";
		String featureSelection = "DF"; //Feature selection method.
		double startProb = 0.5; // Used in feature selection, the starting point of the features.
		double endProb = 0.999; // Used in feature selection, the ending point of the features.
		int DFthreshold = 30; // Filter the features with DFs smaller than this threshold.

//		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.		
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		
		System.out.println("Creating feature vectors, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel, posModel);
//		newEggAnalyzer analyzer = new newEggAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel, posModel, category, 2);
		
		/***** parent child topic model *****/
		ParentChildAnalyzer analyzer = new ParentChildAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);

		analyzer.LoadParentDirectory(articleFolder, suffix);
		analyzer.LoadChildDirectory(commentFolder, suffix);
		
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM") || topicmodel.equals("HTSM") || topicmodel.equals("LRHTSM"))
		{
			analyzer.setMinimumNumberOfSentences(minimunNumberofSentence);
			analyzer.loadPriorPosNegWords(pathToSentiWordNet, pathToPosWords, pathToNegWords, pathToNegationWords);
		}
		
//		analyzer.LoadNewEggDirectory(newEggFolder, suffix); //Load all the documents as the data set.
//		analyzer.LoadDirectory(amazonFolder, suffix);				
		
//		analyzer.setFeatureValues(featureValue, norm);
		_Corpus c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.

		String queryFile = "./data/APP/queryID.txt";
		analyzer.loadQuery(queryFile);
		
		double muDscription = 1000;
		double muReview = 300;
		double eta = 0.4;
		APPLanguageModel lm = new APPLanguageModel(c, muDscription, muReview, eta, analyzer.m_Queries);
		lm.generateReferenceModel();
		lm.printTopAPP4Query(filePrefix);

	}
	
}
