/*
 * Implementation of Sentence Topic/Opinion
 *   - Different THETAs for different sentiments: THETA[S]
 *   - Positive/Negative
 * Author: Yohan Jo
 */
package sto2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.Vector;

import util.DoubleMatrix;
import util.IntegerMatrix;
import util.OrderedDocument;
import util.Sentence;
import util.SentiWord;
import util.Utility;
import util.Word;

public class STO2Core1 {
	private int numUniqueWords;
	private int numTopics;
	private int numSenti;
	private int numRealIterations;
	private int numDocuments;
	private List<String> wordList = null;
	private int numProbWords = 100;
	
	public String inputDir = null;
	public String outputDir = null;
	private Integer intvalTmpOutput = null;
	
	int mod;
	private double alpha;
	private double sumAlpha;
	private double [] betas;  // betas[3]: Common Words, Corresponding Lexicon, The Other Lexicons
	private double [] sumBeta;  // sumBeta[senti]
	private double [] gammas;
	private double sumGamma;
	
	public DoubleMatrix [] Phi; // Phi[senti][word][topic]
	public DoubleMatrix [] Theta;  // Theta[senti][document][topic]
	public DoubleMatrix Pi;
	
	public List<TreeSet<Integer>> sentiWordsList;
	
	private IntegerMatrix [] matrixSWT;
	private IntegerMatrix [] matrixSDT;
	private IntegerMatrix matrixDS;
	
	private int[][] sumSTW;  // sumSTW[S][T]
	private int[][] sumDST;  // sumDST[D][S]
	private int[] sumDS;  // sumDS[D]
	
	private double[][] probTable;
	
	private List<OrderedDocument> documents;
	final private int maxSentenceLength = 50;
	
	public static void main(String [] args) throws Exception {
		int numTopics = 10;
		int numIterations = 100;
		int numSenti = 2;
		int numThreads = 1;
		int mod = 11;
		String inputDir = "./data/input/";
		String outputDir = "./data/output/";
		String dicDir = "./data/input/";
		double alpha = 0.1;
		double [] betas = new double[3];
		betas[0] = 0.001;
		betas[1] = 0.1;
		betas[0] = 0.0;
		
		double [] gammas =  new double[numSenti];
		gammas[0] = 1; gammas[1] = 1;
		String [] betasStr = null;
		String [] gammasStr = null;
		boolean randomInit = false;
		
		String sentiFilePrefix = "SentiWords-";
		String wordListFileName = "selected_combine_fv.txt";
		String docListFileName = "DocumentList.txt";
		String wordDocFileName = "BagOfSentences_pros_cons.txt";
		
/*		*//*** Options ***//*
		for (int i = 0; i < args.length/2; i++) {
			String option = args[2*i];
			String value = args[2*i+1];
			if (option.equals("-t")) numTopics = Integer.valueOf(value);
			else if (option.equals("-s")) numSenti = Integer.valueOf(value);
			else if (option.equals("-i")) numIterations = Integer.valueOf(value);
			else if (option.equals("-th")) numThreads = Integer.valueOf(value);
			else if (option.equals("-d")) inputDir = value.replaceAll("\\\\", "/").replaceAll("/$", "");
			else if (option.equals("-o")) outputDir = value.replaceAll("\\\\", "/").replaceAll("/$", "");
			else if (option.equals("-dic")) dicDir = value.replaceAll("\\\\", "/").replaceAll("/$", "");
			else if (option.equals("-a")) alpha = Double.valueOf(value);
			else if (option.equals("-b")) betasStr = value.split("/");
			else if (option.equals("-g")) gammasStr = value.split("/");
			else if (option.equals("-r")) randomInit = value.toLowerCase().equals("true")?true:false;
		}*/
		if (inputDir == null) inputDir = ".";
		if (outputDir == null) outputDir = new String(inputDir);
		if (dicDir == null) dicDir = new String(inputDir);

		// Exceptions
		if (!new File(inputDir).exists()) throw new Exception("There's no such an input directory as " + inputDir);
		if (!new File(outputDir).exists()) throw new Exception("There's no such an output directory as " + outputDir);
		if (!new File(dicDir).exists()) throw new Exception("Tehre's no such a dictionary directory as " + dicDir);
		
		/*if (alpha <= 0) throw new Exception("Alpha should be specified as a positive real number.");
		if (betasStr == null) throw new Exception("Beta should be specified as positive real numbers.");
		else {
			betas = new double[3];
			if (betasStr.length != 3) throw new Exception("Betas should be length of 3: Common / Lexicon / Rest");
			else for (int i = 0; i < betas.length; i++) betas[i] = Double.valueOf(betasStr[i]);
		}
		if (gammasStr == null) throw new Exception("Gamma should be specified as positive real numbers.");
		else {
			gammas = new double[numSenti];
			if (gammasStr.length == 1) for (int i = 0; i < gammas.length; i++) gammas[i] = Double.valueOf(gammasStr[0]);
			else if (gammasStr.length == numSenti) for (int i = 0; i < gammas.length; i++) gammas[i] = Double.valueOf(gammasStr[i]);
			else throw new Exception("Gammas have a different size than the number of sentiments");
		}*/
		
		String line;
		
		Vector<String> wordList = new Vector<String>();
		BufferedReader wordListFile = new BufferedReader(new FileReader(new File(inputDir+"/"+wordListFileName)));
		while ((line = wordListFile.readLine()) != null)
			if (line != "") wordList.add(line);
		wordListFile.close();
		
//		Vector<String> docList = new Vector<String>();
//		BufferedReader docListFile = new BufferedReader(new FileReader(new File(inputDir+"/"+docListFileName)));
//		while ((line = docListFile.readLine()) != null)
//			if (line != "") docList.add(line);
//		docListFile.close();

		Vector<OrderedDocument> documents = OrderedDocument.instantiateOrderedDocuments(inputDir+"/"+wordDocFileName, null, null);
		
		/*System.out.println("Documents: "+documents.size());
		System.out.println("Unique Words: "+wordList.size());
*/
		ArrayList<TreeSet<String>> sentiWordsStrList = new ArrayList<TreeSet<String>>();
		for (int s = 0; s < numSenti; s++) {
			String dicFilePath = dicDir + "/" + sentiFilePrefix+s+".txt"; 
			if (new File(dicFilePath).exists()) {
				sentiWordsStrList.add(Utility.makeSetOfWordsFromFile(dicFilePath, true));
			}
		}
		
		ArrayList<TreeSet<Integer>> sentiWordsList = new ArrayList<TreeSet<Integer>>(sentiWordsStrList.size());
		for (Set<String> sentiWordsStr : sentiWordsStrList) {
			TreeSet<Integer> sentiWords = new TreeSet<Integer>();
			for (String word : sentiWordsStr)
				sentiWords.add(wordList.indexOf(word));
			sentiWordsList.add(sentiWords);
		}
		
		// Print the configuration
		System.out.println("Documents: "+documents.size());
		System.out.println("Unique Words: "+wordList.size());
		System.out.println("Topics: "+numTopics);
		System.out.println("Sentiments: "+numSenti+" (dictionary: "+sentiWordsList.size()+")");
		System.out.println("Alpha: "+alpha);
	/*	System.out.println("Beta: ");
		for (String betaStr : betasStr) System.out.print(betaStr+" ");
		System.out.println();
		System.out.print("Gamma: ");
		for (String gammaStr : gammasStr) System.out.print(gammaStr+" ");
		System.out.println();*/
		System.out.println("Iterations: "+numIterations);
		System.out.println("Threads: "+numThreads);
		System.out.println("Input Dir: "+inputDir);
		System.out.println("Dictionary Dir: "+dicDir);
		System.out.println("Output Dir: "+outputDir);
		
		STO2Core1 core = new STO2Core1(numTopics, numSenti, wordList, documents, sentiWordsList, alpha, betas, gammas, mod);
		core.generateTmpOutputFiles(inputDir, outputDir, 1000);
		core.initialization(randomInit);
		core.gibbsSampling(numIterations, numThreads);
		core.generateOutputFiles(outputDir);
		
		core.sampleTestdoc();
		core.docsummary(5);
	}
	
	public STO2Core1(int numTopics, int numSenti, List<String> wordList, List<OrderedDocument> documents, List<TreeSet<Integer>> sentiWordsList, double alpha, double[] betas, double [] gammas, int mod) {
		this.numTopics = numTopics;
		this.numSenti = numSenti;
		this.numUniqueWords = wordList.size();
		this.numDocuments = documents.size();
		this.documents = documents;
		this.wordList = wordList;
		this.sentiWordsList = sentiWordsList;
		this.alpha = alpha;
		this.betas = betas;
		this.gammas = gammas;
		this.sumBeta = new double[numSenti];
		this.mod = mod;
		probTable = new double[numTopics][numSenti];
	}
	
	

	public void initialization(boolean randomInit) {
		sumSTW = new int[numSenti][numTopics];
		sumDST = new int[numDocuments][numSenti];
		sumDS = new int[numDocuments];
		
		matrixSWT = new IntegerMatrix[numSenti];
		for (int i = 0; i < numSenti; i++)
			matrixSWT[i] = new IntegerMatrix(numUniqueWords, numTopics);
		matrixSDT = new IntegerMatrix[numSenti];
		for (int i = 0; i < numSenti; i++)
			matrixSDT[i] = new IntegerMatrix(numDocuments, numTopics);
		matrixDS = new IntegerMatrix(numDocuments, numSenti);
		
		int numTooLongSentences = 0;

		for (OrderedDocument currentDoc : documents){
			int docNo = currentDoc.getDocNo();
			
			for (Sentence sentence : currentDoc.getSentences()) {
				int newSenti = -1;
				int numSentenceSenti = 0;
				for (Word sWord : sentence.getWords()) {
					SentiWord word = (SentiWord) sWord;
					
					int wordNo = word.getWordNo();
					for (int s = 0; s < sentiWordsList.size(); s++) {
						if (sentiWordsList.get(s).contains(wordNo)) {
							if (numSentenceSenti == 0 || s != newSenti) numSentenceSenti++;
							word.lexicon = s;
							newSenti = s;
						}
					}
				}
				sentence.numSenti = numSentenceSenti;
				
				if (randomInit || sentence.numSenti != 1)
					newSenti = (int)(Math.random()*numSenti);
				int newTopic = (int)(Math.random()*numTopics);

				if (sentence.getWords().size() > this.maxSentenceLength) numTooLongSentences++;
				
				if (!(numSentenceSenti > 1 || sentence.getWords().size() > this.maxSentenceLength)) {
					sentence.setTopic(newTopic);
					sentence.setSenti(newSenti);
					
					for (Word sWord : sentence.getWords()) {
						((SentiWord) sWord).setSentiment(newSenti);
						sWord.setTopic(newTopic);
						//System.out.println(docNo+" : "+sWord.wordNo);
						matrixSWT[newSenti].incValue(sWord.wordNo, newTopic);
						sumSTW[newSenti][newTopic]++;
					}
					matrixSDT[newSenti].incValue(docNo, newTopic);
					matrixDS.incValue(docNo, newSenti);
	
					sumDST[docNo][newSenti]++;
					sumDS[docNo]++;
				}
			}
		}
		
		System.out.println("Too Long Sentences: "+numTooLongSentences);
	}
	
	public void gibbsSampling(int numIterations, int numThreads) throws Exception {
		this.sumAlpha = this.alpha * this.numTopics;
		int numSentiWords = 0;
		for (Set<Integer> sentiWords : sentiWordsList) numSentiWords += sentiWords.size();
		double sumBetaCommon = this.betas[0] * (this.numUniqueWords - numSentiWords);
		for (int s = 0; s < numSenti; s++) {
			int numLexiconWords = 0;
			if (this.sentiWordsList.size() > s) numLexiconWords = this.sentiWordsList.get(s).size();
			this.sumBeta[s] = sumBetaCommon + this.betas[1]*numLexiconWords + this.betas[2]*(numSentiWords-numLexiconWords);
		}
		this.sumGamma = 0;
		for (double gamma : this.gammas) this.sumGamma += gamma;
		
		System.out.println("Gibbs sampling started (Iterations: "+numIterations+", Threads: "+numThreads+")");
		
		long startTime, endTime;
		for(int i = 0; i < numIterations; i++){
			
			System.out.println( "  - Iteration " + i);

			for (Set<Integer> sentiWords : this.sentiWordsList) {
				for (int wordNo : sentiWords) {
					if (wordNo < 0 || wordNo >= this.wordList.size()) continue;
					System.out.print(this.wordList.get(wordNo)+"/");
					for (int s = 0; s < numSenti; s++) {
						int sum = 0;
						for (int t = 0; t < numTopics; t++) sum += matrixSWT[s].getValue(wordNo, t);
						System.out.print(sum+"/");
					}
					System.out.print(" ");
				}
				System.out.println();
			}
			
			startTime = new Date().getTime();
			
			
			for (OrderedDocument currentDoc : documents)
			{
				if(currentDoc.getDocNo()%mod!=0)
					sampleForDoc(currentDoc);
			}
			
			endTime = new Date().getTime();
			double seconds = (int)(endTime - startTime)/1000.0;
			int minutes = (int)(seconds * (numIterations - i - 1) / 60);
			System.out.println("    Iteration "+i+" took "+seconds+"s. (Estimated Time: "+(minutes/60)+"h "+(minutes%60)+"m)");
			
			
			this.numRealIterations = i + 1;
			if (this.intvalTmpOutput != null && this.numRealIterations % this.intvalTmpOutput == 0 && this.numRealIterations < numIterations) {
				this.Phi = STO2Util.calculatePhi(matrixSWT, sumSTW, this.betas, this.sumBeta, this.sentiWordsList);
				this.Theta = STO2Util.calculateTheta(matrixSDT, sumDST, this.alpha, this.sumAlpha);
				this.Pi = STO2Util.calculatePi(matrixDS, sumDS, this.gammas, this.sumGamma);
				generateOutputFiles(this.outputDir);
			}
		}
		System.out.println("Gibbs sampling terminated.");
		
		this.Phi = STO2Util.calculatePhi(matrixSWT, sumSTW, this.betas, this.sumBeta, this.sentiWordsList);
		this.Theta = STO2Util.calculateTheta(matrixSDT, sumDST, this.alpha, this.sumAlpha);
		this.Pi = STO2Util.calculatePi(matrixDS, sumDS, this.gammas, this.sumGamma);
	}

	private void sampleTestdoc() throws IOException
	{
		
		String dir = "./data/output/";
		PrintWriter out = new PrintWriter(new FileWriter(new File(dir +"VisReviews.html")));
		double perplexity = 0.0;
		int number_of_test_doc = 0;
		int numIterations = 500; // Gibbs iteration
		double log2 = Math.log(2.0);
		int precision_recall [][] = new int[2][2]; // row for true label and col for predicted label 
		int precision_recall_sentences [][] = new int[2][2]; // row for true label and col for predicted label 
		
		// row for real label, col is for predicted label
		precision_recall[0][0] = 0; // 0 is for neg
		precision_recall[0][1] = 0; // 1 is pos 
		precision_recall[1][0] = 0;
		precision_recall[1][1] = 0;



		precision_recall_sentences[0][0] = 0; // 0 is for neg
		precision_recall_sentences[0][1] = 0; // 1 is pos 
		precision_recall_sentences[1][0] = 0;
		precision_recall_sentences[1][1] = 0;
		
		
		for (OrderedDocument doc : documents)
		{
			if(doc.getDocNo()%mod==0)
			{
				number_of_test_doc++;
				
				for(int i = 0; i < numIterations; i++){
					sampleForDoc(doc);
				
				}
				//System.out.println("Visualizing reviews...");
				String [] sentiColors = {"green","red"};
				
				out.println("<h3>Real Document "+doc.getDocNo()+"</h3>");
				for (Sentence sentence : doc.getSentences()) {
					if (sentence.getSenti() < 0 || sentence.getSenti() >= this.numSenti || sentence.getWords().size() > this.maxSentenceLength) continue;
					out.print("<p style=\"color:"+sentiColors[sentence.label]+";\">T"+sentence.getTopic()+":");
					for (Word word : sentence.getWords()) out.print(" "+this.wordList.get(word.wordNo));
					out.println("</p>");
				}
				
				
				
				//for (OrderedDocument doc : this.documents) {
					out.println("<h3>Predicted Document "+doc.getDocNo()+"</h3>");
					for (Sentence sentence : doc.getSentences()) {
						if (sentence.getSenti() < 0 || sentence.getSenti() >= this.numSenti || sentence.getWords().size() > this.maxSentenceLength) continue;
						out.print("<p style=\"color:"+sentiColors[sentence.getSenti()]+";\">T"+sentence.getTopic()+":");
						for (Word word : sentence.getWords())
						{
							out.print(" "+this.wordList.get(word.wordNo));
							precision_recall[sentence.label][sentence.getSenti()]++;
						}
						out.println("</p>");
						precision_recall_sentences[sentence.label][sentence.getSenti()]++;
						
					}
				//}
					
			// perplexity calculation
			int d = doc.getDocNo();
			double val = 0.0;
			int doc_lenght = 0;
			for (Sentence sentence : doc.getSentences()) {
				
				for (Word word : sentence.getWords()){
					double prob = 0.0; 
					int w = word.wordNo;
					doc_lenght++;
					for (int s = 0; s < numSenti; s++) {
						for (int t = 0; t < numTopics; t++) {
							prob += this.Phi[s].getValue(w, t)*((matrixSDT[s].getValue(d,t) + alpha) / (sumDST[d][s] + sumAlpha))*((matrixDS.getValue(d,s) + gammas[s]) / (sumDS[d] + sumGamma));
						}
					}
					val = val + Math.log(prob);	
				}
			}	
			
			double tmp = Math.pow(2.0, -val/doc_lenght / log2);
			out.println("The perplexity is:" + tmp);
			if(!Double.isNaN(tmp) && !Double.isInfinite(tmp))
				perplexity += tmp;
			} // if part
		}
		
		

		double cons_precision = (double)precision_recall[0][0]/(precision_recall[0][0] + precision_recall[1][0]);
		double pros_precision = (double)precision_recall[1][1]/(precision_recall[0][1] + precision_recall[1][1]);


		double cons_recall = (double)precision_recall[0][0]/(precision_recall[0][0] + precision_recall[0][1]);
		double pros_recall = (double)precision_recall[1][1]/(precision_recall[1][0] + precision_recall[1][1]);

		System.out.println("pros_precision:"+pros_precision+" pros_recall:"+pros_recall);
		System.out.println("cons_precision:"+cons_precision+" cons_recall:"+cons_recall);



		double cons_precision_sentence = (double)precision_recall_sentences[0][0]/(precision_recall_sentences[0][0] + precision_recall_sentences[1][0]);
		double pros_precision_sentence = (double)precision_recall_sentences[1][1]/(precision_recall_sentences[0][1] + precision_recall_sentences[1][1]);


		double cons_recall_sentence = (double)precision_recall_sentences[0][0]/(precision_recall_sentences[0][0] + precision_recall_sentences[0][1]);
		double pros_recall_sentence = (double)precision_recall_sentences[1][1]/(precision_recall_sentences[1][0] + precision_recall_sentences[1][1]);

		System.out.println("pros_precision_sentence:"+pros_precision_sentence+" pros_recall_sentence:"+pros_recall_sentence);
		System.out.println("cons_precision_sentence:"+cons_precision_sentence+" cons_recall_sentence:"+cons_recall_sentence);


		System.out.println("Word Level");
		for(int i=0; i<2; i++)
		{
			for(int j=0; j<2; j++)
			{
				System.out.print(precision_recall[i][j]+",");
			}
			System.out.println();
		}

		System.out.println("Sentence Level");
		for(int i=0; i<2; i++)
		{
			for(int j=0; j<2; j++)
			{
				System.out.print(precision_recall_sentences[i][j]+",");
			}
			System.out.println();
		}

		
		double pros_f1 = 2/(1/pros_precision + 1/pros_recall);
		double cons_f1 = 2/(1/cons_precision + 1/cons_recall);
		
		System.out.println("Word level F1 measure:pros:"+pros_f1+", cons:"+cons_f1);
		
		
		pros_f1 = 2/(1/pros_precision_sentence + 1/pros_recall_sentence);
		cons_f1 = 2/(1/cons_precision_sentence + 1/cons_recall_sentence);
		
		System.out.println("Sentence level F1 measure:pros:"+pros_f1+", cons:"+cons_f1);
		
		out.close();
		perplexity = perplexity/number_of_test_doc;
		System.out.println("perplexity:"+ perplexity);
	} 
	
	

	public void docsummary(int topic)
	{
		for (OrderedDocument d : documents)
		{
			if(d.getDocNo()%100==0)
			{
				
				System.out.println("Doc Id:"+d.getDocNo());
					for(int s=0; s<this.numSenti;s++)
					{
						double max = 0.0;
						int index = -1;
						int sentence_index = 0;
						for (Sentence sentence : d.getSentences()) {
							
							if (sentence.getSenti() < 0 || sentence.getSenti() >= this.numSenti || sentence.getWords().size() > this.maxSentenceLength) {
								sentence_index++;
								continue;
							}
							double prod = 1.0;
							for (Word word : sentence.getWords())
							{
								int w = word.wordNo;
								prod = prod * this.Phi[s].getValue(w, topic)*Theta[s].getValue(d.getDocNo(), topic)*this.Pi.getValue(d.getDocNo(),s);
							}
							
							if(prod>max)
							{
								max = prod;
								index = sentence_index ;
							}
							sentence_index++;
							
							
						}
						
						//print
							sentence_index = 0;
							for (Sentence sentence : d.getSentences()) {
							
							if(sentence_index==index)
							{
								System.out.println("s="+s+" sentiment sentence:");
								for (Word word : sentence.getWords())
								{
									System.out.print(" "+this.wordList.get(word.wordNo));
								}
								
								System.out.println();
							}
							sentence_index++;
						}
				
				}
				
			}
		}
		
	}
	
	
	private void sampleForDoc(OrderedDocument currentDoc) {
		int docNo = currentDoc.getDocNo();
		for (Sentence sentence : currentDoc.getSentences()) {
			if (sentence.getSenti() == -1 || sentence.getWords().size() > this.maxSentenceLength) continue;
			
			Map<Word,Integer> wordCnt = sentence.getWordCnt();
			
			double sumProb = 0;
			
			int oldTopic = sentence.getTopic();
			int oldSenti = sentence.getSenti();
			
			matrixSDT[oldSenti].decValue(docNo, oldTopic);
			matrixDS.decValue(docNo, oldSenti);
			
			sumDST[docNo][oldSenti]--;
			sumDS[docNo]--;

			for (Word sWord : sentence.getWords()) {
				matrixSWT[oldSenti].decValue(sWord.wordNo, oldTopic);
				sumSTW[oldSenti][oldTopic]--;
			}
		
			// Sampling
			for (int si = 0; si < numSenti; si++) {
				boolean trim = false;
				
				// Fast Trimming
				for (Word sWord : wordCnt.keySet()) {
					SentiWord word = (SentiWord) sWord;
					if (word.lexicon != null && word.lexicon != si) {
						trim = true;
						break;
					}
				}
				if (trim) {
					for (int ti = 0; ti < numTopics; ti++)
						probTable[ti][si] = 0;
				}
				else {
					for (int ti = 0; ti < numTopics; ti++) {
						double beta0 = sumSTW[si][ti] + sumBeta[si];
						int m0 = 0;
						double expectTSW = 1;
						
						for (Word sWord : wordCnt.keySet()) {
							SentiWord word = (SentiWord) sWord;
							
							double beta;
							if (word.lexicon == null) beta = this.betas[0];
							else if (word.lexicon == si) beta = this.betas[1];
							else beta = this.betas[2];
							
							double betaw = matrixSWT[si].getValue(word.wordNo, ti) + beta;
	
							int cnt = wordCnt.get(word);
							for (int m = 0; m < cnt; m++) {
								expectTSW *= (betaw + m) / (beta0 + m0);
								m0++;
							}
						}

						probTable[ti][si] = (matrixSDT[si].getValue(docNo, ti) + this.alpha) / (sumDST[docNo][si] + this.sumAlpha)
						* (matrixDS.getValue(docNo, si) + this.gammas[si])
						* expectTSW;
	
						sumProb += probTable[ti][si];
					}
				}
			}

			int newTopic = 0, newSenti = 0;
			double randNo = Math.random() * sumProb;
			double tmpSumProb = 0;
			boolean found = false;
			for (int ti = 0; ti < numTopics; ti++) {
				for (int si = 0; si < numSenti; si++) {
					tmpSumProb += probTable[ti][si];
					if (randNo <= tmpSumProb) {
						newTopic = ti;
						newSenti = si;
						found = true;
					}
					if (found) break;
				}
				if (found) break;
			}
			
			sentence.setTopic(newTopic);
			sentence.setSenti(newSenti);
			
			for (Word sWord : sentence.getWords()) {
				SentiWord word = (SentiWord) sWord;
				word.setTopic(newTopic);
				word.setSentiment(newSenti);
				matrixSWT[newSenti].incValue(word.wordNo, newTopic);
				sumSTW[newSenti][newTopic]++;
			}
			matrixSDT[newSenti].incValue(docNo, newTopic);
			matrixDS.incValue(docNo, newSenti);
			
			sumDST[docNo][newSenti]++;
			sumDS[docNo]++;
		}
	}
	
	public void generateTmpOutputFiles(String inputDir, String outputDir, int interval) throws Exception {
		if (inputDir == null || outputDir == null) throw new Exception("Should specify the input and output dirs for tmp output files");
		if (interval <= 0) throw new Exception("The interval of writing tmp output files should be greater than 0");
		this.inputDir = inputDir;
		this.outputDir = outputDir;
		this.intvalTmpOutput = interval;
	}
	
	public void generateOutputFiles (String dir) throws Exception {
		String prefix = "STO2-T"+numTopics+"-S"+numSenti+"("+sentiWordsList.size()+")-A"+alpha+"-B"+betas[0];
		for (int i = 1; i < betas.length; i++) prefix += ","+betas[i];
		prefix += "-G"+gammas[0];
		for (int i = 1; i < numSenti; i++) prefix += ","+gammas[i];
		prefix += "-I"+numRealIterations;
		
		// Phi
		System.out.println("Writing Phi...");
		PrintWriter out = new PrintWriter(new FileWriter(new File(dir + "/" + prefix + "-Phi.csv")));
		for (int s = 0; s < this.numSenti; s++)
			for (int t = 0; t < this.numTopics; t++)
				out.print(",S"+s+"-T"+t);
		out.println();
		for (int w = 0; w < this.wordList.size(); w++) {
			out.print(this.wordList.get(w));
			for (int s = 0; s < this.numSenti; s++) {
				for (int t = 0; t < this.numTopics; t++) {
					out.print(","+this.Phi[s].getValue(w, t));
				}
			}
			out.println();
		}
		out.close();

		// Theta
		System.out.println("Writing Theta...");
		out = new PrintWriter(new FileWriter(new File(dir + "/" + prefix + "-Theta.csv")));
		for (int s = 0; s < this.numSenti; s++)
			for (int t = 0; t < this.numTopics; t++)
				out.print("S"+s+"-T"+t+",");
		out.println();
		for (int d = 0; d < this.numDocuments; d++) {
			for (int s = 0; s < this.numSenti; s++) {
				for (int t = 0; t < this.numTopics; t++) {
					out.print(this.Theta[s].getValue(d, t)+",");
				}
			}
			out.println();
		}
		out.close();
		
		// Pi
		System.out.println("Writing Pi...");
		this.Pi.writeMatrixToCSVFile(dir + "/" + prefix + "-Pi.csv");
		
		// Most probable words
		System.out.println("Writing the most probable words...");
		out = new PrintWriter(new FileWriter(new File(dir + "/" + prefix + "-ProbWords.csv")));
		for (int s = 0; s < this.numSenti; s++)
			for (int t = 0; t < this.numTopics; t++)
				out.print("S"+s+"-T"+t+",");
		out.println();
		int [][][] wordIndices = new int[this.numSenti][this.numTopics][this.numProbWords];
		for (int s = 0; s < this.numSenti; s++) {
			for (int t = 0; t < this.numTopics; t++) {
				Vector<Integer> sortedIndexList = this.Phi[s].getSortedColIndex(t, this.numProbWords);
				for (int w = 0; w < sortedIndexList.size(); w++)
					wordIndices[s][t][w] = sortedIndexList.get(w);
			}
		}
		for (int w = 0; w < this.numProbWords; w++) {
			for (int s = 0; s < this.numSenti; s++) {
				for (int t = 0; t < this.numTopics; t++) {
					int index = wordIndices[s][t][w];
					out.print(this.wordList.get(index)+" ("+String.format("%.3f", Phi[s].getValue(index,t))+"),");
				}
			}
			out.println();
		}
		out.close();

		/*
		// Result reviews
		System.out.println("Visualizing reviews...");
		String [] sentiColors = {"green","red","black"};
		out = new PrintWriter(new FileWriter(new File(dir + "/" + prefix + "-VisReviews.html")));
		for (OrderedDocument doc : this.documents) {
			out.println("<h3>Document "+doc.getDocNo()+"</h3>");
			for (Sentence sentence : doc.getSentences()) {
				if (sentence.getSenti() < 0 || sentence.getSenti() >= this.numSenti || sentence.getWords().size() > this.maxSentenceLength) continue;
				out.print("<p style=\"color:"+sentiColors[sentence.getSenti()]+";\">T"+sentence.getTopic()+":");
				for (Word word : sentence.getWords()) out.print(" "+this.wordList.get(word.wordNo));
				out.println("</p>");
			}
		}
		out.close();
		
		// Sentence probabilities
		System.out.println("Calculating sentence probabilities...");
		out = new PrintWriter(new FileWriter(new File(dir + "/" + prefix + "-SentenceProb.csv")));
		out.print("Document,Sentence,Length");
		for (int s = 0; s < this.numSenti; s++)
			for (int t = 0; t < this.numTopics; t++)
				out.print(",S"+s+"-T"+t);
		out.println();
		for (int d = 0; d < this.documents.size(); d++) {
			OrderedDocument doc = this.documents.get(d);
			for (int sen = 0; sen < doc.getSentences().size(); sen++) {
				Sentence sentence = doc.getSentences().get(sen);
				if (sentence.numSenti > 1 || sentence.getWords().size() > 50) continue;
				if (sentence.getWords().size() == 0) throw new Exception("WHAT???");
				out.print(d+",\"");
				for (Word word : sentence.getWords())
					out.print(this.wordList.get(word.wordNo)+" ");
				out.print("\","+sentence.getWords().size());
				
				double [][] prod = new double[this.numSenti][this.numTopics];
				double sum = 0;
				for (int s = 0; s < this.numSenti; s++) {
					for (int t = 0; t < this.numTopics; t++) {
						prod[s][t]  = 1;
						for (Word word : sentence.getWords()) prod[s][t] *= this.Phi[s].getValue(word.wordNo, t);
						sum += prod[s][t];
					}
				}
				for (int s = 0; s < this.numSenti; s++) {
					for (int t = 0; t < this.numTopics; t++) {
						out.print("," + (prod[s][t] / sum));
					}
				}
				out.println();
			}
		}
		out.close();
		
		// Sentiment lexicon words distribution
		System.out.println("Calculating sentiment lexicon words distributions...");
		out = new PrintWriter(new FileWriter(new File(dir + "/" + prefix + "-SentiLexiWords.csv")));
		for (Set<Integer> sentiWords : this.sentiWordsList) {
			for (int wordNo : sentiWords) {
				if (wordNo < 0 || wordNo >= this.wordList.size()) continue;
				out.print(this.wordList.get(wordNo));
				for (int s = 0; s < numSenti; s++) {
					int sum = 0;
					for (int t = 0; t < numTopics; t++) sum += matrixSWT[s].getValue(wordNo, t);
					out.print(","+sum);
				}
				out.println();
			}
			out.println();
		}
		out.close();
		*/
	}
}
