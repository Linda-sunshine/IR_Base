package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Calendar;
import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import structures._Stn;
import structures._stat;
import utils.Utils;
import Analyzer.ParentChildAnalyzer;

public class BM25Correspondence {
	public BM25Correspondence() {

	}
	
	public void setFV(_Corpus c) {
		// Get the collection of all the documents.
		ArrayList<_Doc> docs = c.getCollection();
		int N = docs.size();
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
			double n = temp.getTotalDocLength() / navg, avgIDF = 0;
			for (_SparseFeature sf : sfs) {
				String featureName = c.getFeature(sf.getIndex());
				_stat stat = c.m_featureStat.get(featureName);
				double TF = sf.getValue();
				double DF = Utils.sumOfArray(stat.getDF());
				double IDF = Math.log((N - DF + 0.5) / (DF + 0.5));
				double BM25 = IDF * TF * (k1 + 1) / (k1 * (1 - b + b * n) + TF);
				sf.setValue(BM25);
				avgIDF += IDF;
			}

			// compute average IDF
			temp.setAvgIDF(avgIDF / sfs.length);
			if (temp instanceof _ParentDoc){
				for (_Stn stnObj : temp.getSentences()) {
					_SparseFeature[] stnFS = stnObj.getFv();
					for (_SparseFeature sf : stnFS) {
						String featureName = c.getFeature(sf.getIndex());
						_stat stat = c.m_featureStat.get(featureName);
						double TF = sf.getValue();
//						System.out.println("TF\t" + TF);
						double DF = Utils.sumOfArray(stat.getDF());
						double IDF = Math.log((N - DF + 0.5) / (DF + 0.5));
						double BM25 = IDF * TF * (k1 + 1)
								/ (k1 * (1 - b + b * n) + TF);
//						System.out.println("BM25\t" + BM25);
						sf.setValue(BM25);
					}
				}
			}
			
		}
	}
	
	public void rankChild4Parent(_Corpus c, String TopChild4ParentFile){
		System.out.println("rank child 4 parent");
		
		try {
			PrintWriter pw = new PrintWriter(new File(TopChild4ParentFile));

			for (_Doc doc : c.getCollection()) {
				if (doc instanceof _ParentDoc) {
					pw.print(doc.getName() + "\t");
					
					for(_ChildDoc cDoc : ((_ParentDoc) doc).m_childDocs) {
						double childSim = computeSimilarity(
								doc.getSparse(), cDoc.getSparse());
						pw.print(cDoc.getName()+":"+childSim+"\t");
					}
					pw.println();
					
				}
			}
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}
	
	public void rankStn4Child(_Corpus c, String TopStn4ChildFile){
		System.out.println("rank stn");
		
		try {
			PrintWriter pw = new PrintWriter(new File(TopStn4ChildFile));

			for (_Doc doc : c.getCollection()) {
				if (doc instanceof _ParentDoc) {
					pw.print(doc.getName() + "\t" + ((_ParentDoc) doc).m_childDocs.size());
					double stnTopicSimilarity = 0.0;
					double docTopicSimilarity = 0.0;
					pw.println();
					for (_ChildDoc cDoc : ((_ParentDoc) doc).m_childDocs) {
						pw.print(cDoc.getName() + "\t");
//						docTopicSimilarity = computeSimilarity(doc.getSparse(),
//								cDoc.getSparse());
//						pw.print(docTopicSimilarity);
						for (_Stn stnObj : doc.getSentences()) {
							stnTopicSimilarity = computeSimilarity(
									stnObj.getFv(), cDoc.getSparse());
							pw.print( (stnObj.getIndex() + 1) + ":"
									+ stnTopicSimilarity+"\t");
						}
						pw.println();
					}
					
				}
			}
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}
	
	public void rankChild4Stn(_Corpus c, String TopChild4StnFile){
		System.out.println("rank child");
		
		try {
			PrintWriter pw = new PrintWriter(new File(TopChild4StnFile));

			for (_Doc doc : c.getCollection()) {
				if (doc instanceof _ParentDoc) {
					pw.print(doc.getName() + "\t" + ((_ParentDoc) doc).getSenetenceSize());
					double stnTopicSimilarity = 0.0;
					double docTopicSimilarity = 0.0;
					pw.println();
					
					for(_Stn stnObj: doc.getSentences()){
						pw.print((stnObj.getIndex()+1)+"\t");
						for(_ChildDoc cDoc : ((_ParentDoc) doc).m_childDocs) {
							double childSim = computeSimilarity(
									stnObj.getFv(), cDoc.getSparse());
							pw.print(cDoc.getName()+":"+childSim+"\t");
						}
						pw.println();
					}
					
				}
			}
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public void discoverSpecificComments(_Corpus c, String similarityFile) {
		System.out.println("topic similarity");

		try {
			PrintWriter pw = new PrintWriter(new File(similarityFile));

			for (_Doc doc : c.getCollection()) {
				if (doc instanceof _ParentDoc) {
					pw.print(doc.getName() + "\t");
					double stnTopicSimilarity = 0.0;
					double docTopicSimilarity = 0.0;
					for (_ChildDoc cDoc : ((_ParentDoc) doc).m_childDocs) {
						pw.print(cDoc.getName() + ":");
						docTopicSimilarity = computeSimilarity(doc.getSparse(),
								cDoc.getSparse());
						pw.print(docTopicSimilarity);
						for (_Stn stnObj : doc.getSentences()) {
							stnTopicSimilarity = computeSimilarity(
									stnObj.getFv(), cDoc.getSparse());
							pw.print(":" + (stnObj.getIndex() + 1) + ":"
									+ stnTopicSimilarity);
						}
						pw.print("\t");
					}
					pw.println();
				}
			}
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}

	public double computeSimilarity(_SparseFeature[] spVct1,
			_SparseFeature[] spVct2) {
		return Utils.cosine(spVct1, spVct2);
	}
	
	public void outputDF(_Corpus c, String DFfile){
		System.out.println("output DF");
		try {
			PrintWriter pw = new PrintWriter(new File(DFfile));

			for(String feature:c.m_featureStat.keySet()){
				_stat stat = c.m_featureStat.get(feature);
				double DF = Utils.sumOfArray(stat.getDF());
				pw.print(feature+"\t");
				pw.print(DF+"\n");
			}
			
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public void identifyCopiedStnCmnt(_Corpus c, String fileName){
		ArrayList<String> copiedPairsList = new ArrayList<String>();
		
		System.out.println("recoginize copied sentences");

		for(_Doc d: c.getCollection()){
			if(d instanceof _ParentDoc){
				for(_Stn stnObj: d.getSentences()){
					
					_SparseFeature[] stnSF = stnObj.getFv();
					for(_ChildDoc cDoc:((_ParentDoc) d).m_childDocs){
						_SparseFeature[] docSF = cDoc.getSparse();
						
						boolean copiedFlag = true;
						for(_SparseFeature fv: stnSF){
							int wid = fv.getIndex();
							if(Utils.indexOf(docSF, wid) ==-1){
								copiedFlag = false;
							}
						}
						
						if(copiedFlag == true){
							String copiedStnName = cDoc.getName()+":"+(stnObj.getIndex()+1);
							copiedPairsList.add(copiedStnName);
						}
					}
				}
			}
		}
		
		try {
			PrintWriter pw = new PrintWriter(new File(fileName));
			
			for(String stnName:copiedPairsList){
				pw.println(stnName);	
			}
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
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
		String topicmodel = "BM25Corr"; // 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTSM, ParentChild_Gibbs
	
		String category = "tablet";
		int number_of_topics = 20;
		boolean loadNewEggInTrain = true; // false means in training there is no reviews from NewEgg
		boolean setRandomFold = false; // false means no shuffling and true means shuffling
		int loadAspectSentiPrior = 0; // 0 means nothing loaded as prior; 1 = load both senti and aspect; 2 means load only aspect 
		
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = topicmodel.equals("LDA_Gibbs")?200:5.0;//these two parameters must be larger than 1!!!
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

		BM25Correspondence bm25Corr = new BM25Correspondence();
		bm25Corr.setFV(c);
		String similarityFile = filePrefix + "/topicSimilarity.txt";
		String TopChild4StnFile = filePrefix + "/topChild4Stn.txt";
		String TopStn4ChildFile = filePrefix + "/topStn4Child.txt";
		String TopChild4ParentFile = filePrefix + "/topChild4Parent.txt";
		
		String copiedStnFile = filePrefix+"/copiedStn.txt";
		bm25Corr.identifyCopiedStnCmnt(c, copiedStnFile);
		
//		bm25Corr.rankChild4Stn(c, TopChild4StnFile);
//		bm25Corr.rankStn4Child(c, TopStn4ChildFile);
//		bm25Corr.rankChild4Parent(c, TopChild4ParentFile);
//		bm25Corr.discoverSpecificComments(c, similarityFile);
//		String DFFile = filePrefix+"/df.txt";
//		bm25Corr.outputDF(c, DFFile);
	}
}
