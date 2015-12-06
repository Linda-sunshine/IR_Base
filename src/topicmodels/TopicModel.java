package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Random;

import javax.swing.plaf.metal.MetalIconFactory.FolderIcon16;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import structures._Stn;
import structures.annotationType;
import topicmodels.multithreads.TopicModelWorker;
import topicmodels.multithreads.TopicModel_worker.RunType;
import utils.Utils;

public abstract class TopicModel {
	protected int number_of_topics;
	protected int vocabulary_size;
	protected double m_converge;//relative change in log-likelihood to terminate EM
	protected int number_of_iteration;//number of iterations in inferencing testing document
	protected _Corpus m_corpus;	
	
	protected boolean m_logSpace; // whether the computations are all in log space
	
	protected boolean m_LoadnewEggInTrain = true; // check whether newEgg will be loaded in trainSet or Not
	protected boolean m_randomFold = true; // true mean randomly take K fold and test and false means use only 1 fold and use the fixed trainset
	
	//for training/testing split
	protected ArrayList<_Doc> m_trainSet, m_testSet;
	protected double[][] word_topic_sstat; /* fractional count for p(z|d,w) */
	
	//smoothing parameter for p(w|z, \beta)
	protected double d_beta; 	
	
	protected boolean m_display; // output EM iterations
	protected boolean m_collectCorpusStats; // if we will collect corpus-level statistics (for efficiency purpose)
	
	protected boolean m_multithread = false; // by default we do not use multi-thread mode
	protected Thread[] m_threadpool = null;
	protected TopicModelWorker[] m_workers = null;
	protected double[][] prosConsPerFold;
	
	protected int m_trainSize = 0; // varying trainSet size for Amazon; 0 means dataset only from newEgg
	private boolean m_trainSetForASUMJST = false;
	protected String m_category;
	private String filePath;

	
	public PrintWriter infoWriter;
	public PrintWriter summaryWriter;
	public PrintWriter debugWriter;
	
	public TopicModel(int number_of_iteration, double converge, double beta, _Corpus c) {
		this.vocabulary_size = c.getFeatureSize();
		this.number_of_iteration = number_of_iteration;
		this.m_converge = converge;
		this.d_beta = beta;
		this.m_corpus = c;
		
		m_display = true; // by default we will track EM iterations
	}
	
	@Override
	public String toString() {
		return "Topic Model";
	}
	
	public void setDisplay(boolean disp) {
		m_display = disp;
	}
	
	public void setNewEggLoadInTrain(boolean flag){
		if(flag)
			System.out.println("NewEgg is added in Training Set");
		else
			System.out.println("NewEgg is NOT added in Training Set");
		m_LoadnewEggInTrain = flag;
	}
	
	public void setRandomFold(boolean flag){
		m_randomFold = flag;
	}
	
	public void setInforWriter(String path){
		System.out.println("Info File Path: "+ path);
		try{
			infoWriter = new PrintWriter(new File(path));
		}catch(Exception e){
			System.err.println(path+" Not found!!");
		}
	}
	
	public void setSummaryWriter(String path){
		System.out.println("Summary File Path: "+ path);
		try{
			summaryWriter = new PrintWriter(new File(path));
		}catch(Exception e){
			System.err.println(path+" Not found!!");
		}
	}
	
	public void setDebugWriter(String path){
		System.out.println("Debug File Path: "+ path);
		try{
			debugWriter = new PrintWriter(new File(path));
		}catch(Exception e){
			System.err.println(path+" Not found!!");
		}
	}
	
	//initialize necessary model parameters
	protected abstract void initialize_probability(Collection<_Doc> collection);	
	
	// to be called per EM-iteration
	protected abstract void init();
	
	// to be called by the end of EM algorithm 
	protected abstract void finalEst();
	
	// to be call per test document
	protected abstract void initTestDoc(_Doc d);
	
	//estimate posterior distribution of p(\theta|d)
	protected abstract void estThetaInDoc(_Doc d);
	
	// perform inference of topic distribution in the document
	public double inference(_Doc d) {
		initTestDoc(d);//this is not a corpus level estimation
		
		double delta, last = 1, current;
		int  i = 0;
		do {
			current = calculate_E_step(d);
			estThetaInDoc(d);			
			
			delta = (last - current)/last;
			last = current;
		} while (Math.abs(delta)>m_converge && ++i<this.number_of_iteration);
		return current;
	}
		
	//E-step should be per-document computation
	public abstract double calculate_E_step(_Doc d); // return log-likelihood
	
	//M-step should be per-corpus computation
	public abstract void calculate_M_step(int i); // input current iteration to control sampling based algorithm
	
	//compute per-document log-likelihood
	protected abstract double calculate_log_likelihood(_Doc d);
	
	//print top k words under each topic
	public abstract void printTopWords(int k, String topWordPath);
	public abstract void printTopWords(int k);
	public abstract void createWordIntrusionList(String wordIntrusionPath);
	
	// compute corpus level log-likelihood
	protected double calculate_log_likelihood() {
		return 0;
	}
	
	public void EMonCorpus() {
		m_trainSet = m_corpus.getCollection();
		EM();
	}
	
	double multithread_E_step() {
		for(int i=0; i<m_workers.length; i++) {
			m_workers[i].setType(RunType.RT_EM);
			m_threadpool[i] = new Thread(m_workers[i]);
			m_threadpool[i].start();
		}
		
		//wait till all finished
		for(Thread thread:m_threadpool){
			try {
				thread.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		double likelihood = 0;
		for(TopicModelWorker worker:m_workers)
			likelihood += worker.accumluateStats(word_topic_sstat);
		return likelihood;
	}
	
	void multithread_inference() {
		//clear up for adding new testing documents
		for(int i=0; i<m_workers.length; i++) {
			m_workers[i].setType(RunType.RT_inference);
			m_workers[i].clearCorpus();
		}
		
		//evenly allocate the testing work load
		int workerID = 0;
		
		// whether it is in debug mode or not we will always use the testSet for debug
		//if(debugWriter==null){
		for(_Doc d:m_testSet) {
			m_workers[workerID%m_workers.length].addDoc(d);
			workerID++;
		}
//		}else{
//			for(_Doc d:m_corpus.getCollection()) {
//				m_workers[workerID%m_workers.length].addDoc(d);
//				workerID++;
//			}
//		}
//			
		for(int i=0; i<m_workers.length; i++) {
			m_threadpool[i] = new Thread(m_workers[i]);
			m_threadpool[i].start();
		}
		
		//wait till all finished
		for(Thread thread:m_threadpool){
			try {
				thread.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	public void setFilePathForJSTASUM(int m_trainSize, String m_category, String filePath){
		this.m_trainSetForASUMJST = true;
		this.m_trainSize = m_trainSize;
		this.m_category = m_category;
		this.filePath = filePath;
	}
	
	public void generateFileForJSTASUM(){

		try{

			PrintWriter jstTrainCorpusWriter = new PrintWriter(new File(filePath +"JST/" + m_trainSize +"/"+m_category+"/MR.dat"));
			PrintWriter jstTestCorpusWriter = new PrintWriter(new File(filePath +"JST/" + m_trainSize +"/"+m_category+"/MR_test.dat"));
			// for sentence level of JST
			PrintWriter jstTestCorpusForSentenceWriter = new PrintWriter(new File(filePath +"JST/" +  m_trainSize +"/"+m_category+"/MR_test_sentence.dat"));
			PrintWriter jstTestCorpusForSentenceLabelWriter = new PrintWriter(new File(filePath +"JST/" + m_trainSize +"/"+m_category+"/MR_test_label_sentence.dat"));

			PrintWriter asumWriter = new PrintWriter(new File( filePath + "ASUM/"+ m_trainSize +"/"+m_category+"/BagOfSentences_pros_cons.txt"));
			PrintWriter asumFeatureWriter = new PrintWriter(new File( filePath + "ASUM/"+ m_trainSize +"/"+m_category+"/selected_combine_fv.txt"));


			int index = -1;
			String word = "";

			// Training Documnet Generation for JST
			for(_Doc trainDoc:m_trainSet){
				jstTrainCorpusWriter.write("d"+trainDoc.getID()+" ");
				for(_SparseFeature feature :trainDoc.getSparse()){
					index = feature.getIndex(); 
					word = m_corpus.getFeature(index);
					jstTrainCorpusWriter.write(word+" ");
				}
				jstTrainCorpusWriter.write("\n");
			}

			jstTrainCorpusWriter.flush();
			jstTrainCorpusWriter.close();

			index = -1;
			word = "";
			for(_Doc testDoc:m_testSet){
				jstTestCorpusWriter.write("d"+testDoc.getID()+" ");
				for(_SparseFeature feature :testDoc.getSparse()){
					index = feature.getIndex(); 
					word = m_corpus.getFeature(index);
					jstTestCorpusWriter.write(word+" ");
				}
				jstTestCorpusWriter.write("\n");
			}

			jstTestCorpusWriter.flush();
			jstTestCorpusWriter.close();




			index = -1;
			word = "";
			int sentenceCounter = 0;
			for(_Doc testDoc:m_testSet){
				for(_Stn sentence : testDoc.getSentences()){
					jstTestCorpusForSentenceWriter.write("d"+sentenceCounter+" ");
					int sentenceLabel = sentence.getSentenceSenitmentLabel()==-1?3:sentence.getSentenceSenitmentLabel();
					jstTestCorpusForSentenceLabelWriter.write("d"+sentenceCounter+" "+sentenceLabel+"\n");
					for(_SparseFeature feature : sentence.getFv()){
						index = feature.getIndex(); 
						word = m_corpus.getFeature(index);
						jstTestCorpusForSentenceWriter.write(word+" ");
					}
					sentenceCounter++;
					jstTestCorpusForSentenceWriter.write("\n");
				}
			}

			jstTestCorpusForSentenceWriter.flush();
			jstTestCorpusForSentenceWriter.close();

			jstTestCorpusForSentenceLabelWriter.flush();
			jstTestCorpusForSentenceLabelWriter.close();

			// Train Document generation for ASUM 
			index = -1;
			word = "";
			for(_Doc d:m_trainSet){
				asumWriter.write(d.getSenetenceSize()+"\n");
				//writing the itemID
				asumWriter.write(d.getItemID()+"\n");
				for(_Stn sentence : d.getSentences()){
					int sentenceLabel = sentence.getSentenceSenitmentLabel();
					if(sentenceLabel==0) // pros
						sentenceLabel = -1;
					else if(sentenceLabel==1) // cons
						sentenceLabel = -2;
					else if(sentenceLabel==-1) // from Amazon
						sentenceLabel = -3;
					asumWriter.write(sentenceLabel+" ");
					for(_SparseFeature feature : sentence.getFv()){
						index = feature.getIndex(); 
						asumWriter.write(index+" ");
					}
					asumWriter.write("\n");
					
					//writing raw sentence in next line
					asumWriter.write(sentence.getRawSentence().replaceAll("\\r\\n|\\r|\\n", " ")+"\n");
				}
			}


			// Test Document generation for ASUM
			index = -1;
			word = "";
			for(_Doc d:m_testSet){
				int senlen = (-1)*d.getSenetenceSize();
				asumWriter.write(senlen+"\n");
				asumWriter.write(d.getItemID()+"\n");
				for(_Stn sentence : d.getSentences()){
					int sentenceLabel = sentence.getSentenceSenitmentLabel();
					if(sentenceLabel==0) // pros
						sentenceLabel = -1;
					else if(sentenceLabel==1) // cons
						sentenceLabel = -2;
					else if(sentenceLabel==-1) // from Amazon
						sentenceLabel = -3;
					asumWriter.write(sentenceLabel+" ");
					for(_SparseFeature feature : sentence.getFv()){
						index = feature.getIndex(); 
						asumWriter.write(index+" ");
					}
					asumWriter.write("\n");
					asumWriter.write(sentence.getRawSentence().replaceAll("\\r\\n|\\r|\\n", " ")+"\n");
				}
			}

			asumWriter.flush();
			asumWriter.close();

			for(int i=0; i<m_corpus.getFeatureSize();i++){
				asumFeatureWriter.write(m_corpus.getFeature(i)+"\n");
			}

			asumFeatureWriter.flush();
			asumFeatureWriter.close();

		}
		catch(Exception e){
			e.printStackTrace();
			System.err.print("JST and ASUM File Not Found"+this.filePath);
		}
	}
	
	public void calculateTopicTransitionMatrix(String filePath){
		double [] startTopicCount = new double [this.number_of_topics];
		double [][] topicTransitionCountMatrix = new double [this.number_of_topics][this.number_of_topics];
		int topicIndexPrev = -1;
		int topicIndexCurrent = -1;
		try {
			PrintWriter transitionWriter = new PrintWriter(new File(filePath));
			
			/*initialize all the matrix to 0*/
			for(int i=0; i<this.number_of_topics; i++){
				startTopicCount[i] = 0;
				for(int j=0;j<this.number_of_topics;j++){
					topicTransitionCountMatrix[i][j] = 0;
				}
			}

			/*collecting the count*/
			for(_Doc d:m_testSet){
				_Stn sentences[] = d.getSentences();
				topicIndexPrev = sentences[0].getSentencePredictedTopic(); // 0 is the first sentence
				startTopicCount[topicIndexPrev]+=1;
				for(int s=1; s<sentences.length;s++){
					topicIndexCurrent = sentences[s].getSentencePredictedTopic();
					topicTransitionCountMatrix[topicIndexPrev][topicIndexCurrent]+=1;
					topicIndexPrev = topicIndexCurrent;
				}
			}

			/*start normalizing*/
			double sum = 0.0;
			for(int i=0; i<this.number_of_topics; i++){
				sum = sum + startTopicCount[i];
			}
			
			for(int i=0; i<this.number_of_topics; i++){
				startTopicCount[i] = (double) startTopicCount[i]/sum;
			}
			
			sum = 0.0;
			
			for(int i=0; i<this.number_of_topics; i++){
				/*accumulating the sum*/
				sum = 0.0;
				for(int j=0;j<this.number_of_topics;j++){
					sum+=topicTransitionCountMatrix[i][j];
				}
				/*normalize all*/
				transitionWriter.write(startTopicCount[i]+",");
				for(int j=0;j<this.number_of_topics;j++){
					topicTransitionCountMatrix[i][j]= (double)topicTransitionCountMatrix[i][j]/sum;
					transitionWriter.write(topicTransitionCountMatrix[i][j]+",");
				}
				transitionWriter.write("\n");
			}
			
			transitionWriter.flush();
			transitionWriter.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.err.println("Transition File:"+filePath+" Not found!!");
		}
	}
	
	
	public void EM() {	
		long starttime = System.currentTimeMillis();
		
		m_collectCorpusStats = true;
		initialize_probability(m_trainSet);
		
		double delta, last = calculate_log_likelihood(), current;
		int  i = 0;
		do {
			init();
			
			if (m_multithread)
				current = multithread_E_step();
			else {
				current = 0;
				for(_Doc d:m_trainSet)
					current += calculate_E_step(d);
			}
			
			calculate_M_step(i);
			
			if (m_converge>0)
				current += calculate_log_likelihood();//together with corpus-level log-likelihood
			
			if (i>0)
				delta = (last-current)/last;
			else
				delta = 1.0;
			last = current;
			
			if (m_display && i%10==0) {
				if (m_converge>0){
					System.out.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);
					infoWriter.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);
				}else {
					System.out.print(".");
					infoWriter.print(".");
					if (i%200==190){
						System.out.println();
						infoWriter.print("\n");
					}
				}
			}
			
			if (Math.abs(delta)<m_converge)
				break;//to speed-up, we don't need to compute likelihood in many cases
		} while (++i<this.number_of_iteration);
		
		finalEst();
		
		long endtime = System.currentTimeMillis() - starttime;
		System.out.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime/1000);	
		infoWriter.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime/1000);	
	}

	public double Evaluation(int foldNumber) {
		m_collectCorpusStats = false;
		double perplexity = 0, loglikelihood, log2 = Math.log(2.0), sumLikelihood = 0;
		
		if (m_multithread) {
			multithread_inference();
			System.out.println("In thread");
			for(TopicModelWorker worker:m_workers) {
				sumLikelihood += worker.getLogLikelihood();
				perplexity += worker.getPerplexity();
			}
		} else {
			System.out.println("In Normal");
			for(_Doc d:m_testSet) {				
				loglikelihood = inference(d);
				sumLikelihood += loglikelihood;
				perplexity += Math.pow(2.0, -loglikelihood/d.getTotalDocLength() / log2);
			}
			
		}
		perplexity /= m_testSet.size();
		sumLikelihood /= m_testSet.size();
		
		if(this instanceof HTSM)
			calculatePrecisionRecall(foldNumber);

		System.out.format("Test set perplexity is %.3f and log-likelihood is %.3f\n", perplexity, sumLikelihood);
		infoWriter.format("Test set perplexity is %.3f and log-likelihood is %.3f\n", perplexity, sumLikelihood);
		
		return perplexity;
	}

	
	public void debugOutputWrite(){
		debugWriter.println("Doc ID, Source, SentenceIndex,Sentence, ActualSentiment, PredictedSentiment, PredictedTopic, TopicTransitionProbabilty, SentimentTransitionProbabilty");
		for(_Doc d:m_testSet){
			for(int i=0; i<d.getSenetenceSize(); i++){
				debugWriter.format("%d,%d,%d,\"%s\",%d,%d,%d,%f,%f\n", d.getID(),d.getAnnotationType(),i,d.getSentence(i).getRawSentence().toLowerCase().replaceAll("[^a-z0-9.]", " ") ,d.getSentence(i).getSentenceSenitmentLabel(),d.getSentence(i).getSentencePredictedSenitmentLabel(), d.getSentence(i).getSentencePredictedTopic(), d.getSentence(i).getTopicTransition(),d.getSentence(i).getSentimentTransition());
			}
		}
		debugWriter.flush();
		debugWriter.close();
	}
	
	
	
	public void calculatePrecisionRecall(int foldNumber){
		int[][] precision_recall = new int [2][2];
		precision_recall [0][0] = 0; // 0 is for pos
		precision_recall[0][1] = 0; // 1 is neg 
		precision_recall[1][0] = 0;
		precision_recall[1][1] = 0;
		
		int actualLabel, predictedLabel;
		
		for(_Doc d:m_testSet) {
			// if document is from newEgg which is 2 then calculate precision-recall
			if(d.getAnnotationType()==annotationType.ANNOTATED){
				
				for(int i=0; i<d.getSenetenceSize(); i++){
					actualLabel = d.getSentence(i).getSentenceSenitmentLabel();
					predictedLabel = d.getSentence(i).getSentencePredictedSenitmentLabel();
					precision_recall[actualLabel][predictedLabel]++;
				}
			}
			else if(d.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED){
				
				for(int i=0; i<d.getSenetenceSize(); i++){
					actualLabel = d.getSentence(i).getSentenceSenitmentLabel();
					if(actualLabel==-1) continue;
					predictedLabel = d.getSentence(i).getSentencePredictedSenitmentLabel();
					precision_recall[actualLabel][predictedLabel]++;
					if(actualLabel==0 && predictedLabel==1){
						System.out.print("\n"+d.getSentence(i).getRawSentence()+",");
						
					}
						
				}
			}
		}
		
		System.out.println("\nConfusion Matrix");
		infoWriter.println("Confusion Matrix");
		for(int i=0; i<2; i++)
		{
			for(int j=0; j<2; j++)
			{
				System.out.print(precision_recall[i][j]+",");
				infoWriter.print(precision_recall[i][j]+",");
			}
			System.out.println();
			infoWriter.println();
		}
		
		double pros_precision = (double)precision_recall[0][0]/(precision_recall[0][0] + precision_recall[1][0]);
		double cons_precision = (double)precision_recall[1][1]/(precision_recall[0][1] + precision_recall[1][1]);
		
		
		double pros_recall = (double)precision_recall[0][0]/(precision_recall[0][0] + precision_recall[0][1]);
		double cons_recall = (double)precision_recall[1][1]/(precision_recall[1][0] + precision_recall[1][1]);
		
		System.out.println("pros_precision:"+pros_precision+" pros_recall:"+pros_recall);
		infoWriter.println("pros_precision:"+pros_precision+" pros_recall:"+pros_recall);
		System.out.println("cons_precision:"+cons_precision+" cons_recall:"+cons_recall);
		infoWriter.println("cons_precision:"+cons_precision+" cons_recall:"+cons_recall);
		
		
		double pros_f1 = 2/(1/pros_precision + 1/pros_recall);
		double cons_f1 = 2/(1/cons_precision + 1/cons_recall);
		
		prosConsPerFold[foldNumber][0]=pros_f1;
		prosConsPerFold[foldNumber][1]=cons_f1;
		
		System.out.println("F1 measure:pros:"+pros_f1+", cons:"+cons_f1);
		infoWriter.println("F1 measure:pros:"+pros_f1+", cons:"+cons_f1);
	}
	
	// added by Mustafiz for normalizing the feature of topic and sentiment for each doc
	// here we normalize each feature by z-score https://en.wikipedia.org/wiki/Standard_score
	public void normalizeFeature(){
		System.out.println("Normalizing the sentiment and topic features");
		int totalNumberofSentencesInTrainSet = 0;
		
		int sentimentFeatureSize = 1; // because only 0 is continuous
		int topicFeatureSize = 3; // because the first 0,1, and 2 are continuous
		
		double[] sentimentFeatures = new double[sentimentFeatureSize];
		double[] topicFeatures = new double[topicFeatureSize];
		
		double[] sentimentFeaturesMean = new double[sentimentFeatureSize];
		double[] topicFeaturesMean = new double[topicFeatureSize];
		
		double[] sentimentFeaturesStandardDeviation = new double[sentimentFeatureSize];
		double[] topicFeaturesStandardDeviation = new double[topicFeatureSize];
		
		//initialize the vectors to zero
		for(int i=0; i<sentimentFeatureSize;i++)
			sentimentFeatures[i] = 0.0;
		
		for(int i=0; i<topicFeatureSize;i++)
			topicFeatures[i] = 0.0;
		
		// getting the summation of all the values of features across all the documents
		for(_Doc d:m_trainSet){
			_Stn[] sentences = d.getSentences();
			int stnSize = d.getSenetenceSize();
			totalNumberofSentencesInTrainSet+=stnSize;
			for(int s=0; s<stnSize; s++){
				for(int i=0; i<sentimentFeatureSize;i++)
					sentimentFeatures[i] += sentences[s].m_sentiTransitFv[i];
				for(int i=0; i<topicFeatureSize;i++)
					topicFeatures[i] += sentences[s].m_transitFv[i];
			}// sentence loop
		}// doc loop
		
		//taking the mean
		for(int i=0; i<sentimentFeatureSize;i++)
			sentimentFeaturesMean[i] = sentimentFeatures[i]/totalNumberofSentencesInTrainSet;
		
		for(int i=0; i<topicFeatureSize;i++)
			topicFeaturesMean[i] = topicFeatures[i]/totalNumberofSentencesInTrainSet;
		
		//calculating the standard deviation STD
		
		//Again initialize the vectors to zero
		for(int i=0; i<sentimentFeatureSize;i++)
			sentimentFeatures[i] = 0.0;

		for(int i=0; i<topicFeatureSize;i++)
			topicFeatures[i] = 0.0;
		
		// getting the summation of all the values of features across all the documents for STD
		for(_Doc d:m_trainSet){
			_Stn[] sentences = d.getSentences();
			int stnSize = d.getSenetenceSize();
			for(int s=0; s<stnSize; s++){
				for(int i=0; i<sentimentFeatureSize;i++)
					sentimentFeatures[i] += (sentences[s].m_sentiTransitFv[i]-sentimentFeaturesMean[i])*(sentences[s].m_sentiTransitFv[i]-sentimentFeaturesMean[i]);
				for(int i=0; i<topicFeatureSize;i++)
					topicFeatures[i] += (sentences[s].m_transitFv[i]-topicFeaturesMean[i])*(sentences[s].m_transitFv[i]-topicFeaturesMean[i]);
			}// sentence loop
		}// doc loop

		
		//taking the STD
		for(int i=0; i<sentimentFeatureSize;i++)
			sentimentFeaturesStandardDeviation[i] = Math.sqrt(sentimentFeatures[i]/(totalNumberofSentencesInTrainSet-1));

		for(int i=0; i<topicFeatureSize;i++)
			topicFeaturesStandardDeviation[i] = Math.sqrt(topicFeatures[i]/(totalNumberofSentencesInTrainSet-1));

		//Now normalize using z-score both train and test set
		for(_Doc d:m_corpus.getCollection()){
			_Stn[] sentences = d.getSentences();
			int stnSize = d.getSenetenceSize();
			for(int s=0; s<stnSize; s++){
				for(int i=0; i<sentimentFeatureSize;i++)
					sentences[s].m_sentiTransitFv[i] = (sentences[s].m_sentiTransitFv[i] - sentimentFeaturesMean[i])/sentimentFeaturesStandardDeviation[i];
				for(int i=0; i<topicFeatureSize;i++)
					sentences[s].m_transitFv[i] = (sentences[s].m_transitFv[i] - topicFeaturesMean[i])/topicFeaturesStandardDeviation[i]; 
			}// sentence loop
		}// Corpus loop

	}
	
	//k-fold Cross Validation.
	public void crossValidation(int k) {
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		HashMap<Integer, double[]> tpt = new HashMap<Integer, double[]>();
		
		prosConsPerFold = new  double[k][2]; // 0 for pros and 1 cons // k is the number of fold
		for(int a=0; a<k;a++)
			prosConsPerFold[a][0]=prosConsPerFold[a][1]=0.0;
		
		double[] perf;
		int amazonTrainsetRatingCount[] = {0,0,0,0,0};
		int amazonRatingCount[] = {0,0,0,0,0};
		
		int newEggRatingCount[] = {0,0,0,0,0};
		int newEggTrainsetRatingCount[] = {0,0,0,0,0};
		
		/*if(m_randomFold==true){
			perf = new double[k];
			m_corpus.shuffle(k);
			int[] masks = m_corpus.getMasks();
			ArrayList<_Doc> docs = m_corpus.getCollection();
			//Use this loop to iterate all the ten folders, set the train set and test set.
			for (int i = 0; i < k; i++) {
				
				for (int j = 0; j < masks.length; j++) {
					
					if( masks[j]==i ) 
						m_testSet.add(docs.get(j));
					else 
						m_trainSet.add(docs.get(j));
					
					if(masks[j]==i || m_trainSet.size()<=0.8*docs.size())
						m_trainSet.add(docs.get(j));
					else
						m_testSet.add(docs.get(j));
					
				}
				
				System.out.println("Fold number "+i);
				System.out.println("Train Set Size "+m_trainSet.size());
				System.out.println("Test Set Size "+m_testSet.size());

				if(this instanceof HTSM)
					normalizeFeature();
				long start = System.currentTimeMillis();
				EM();
				perf[i] = Evaluation(i);
				System.out.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
				
				m_trainSet.clear();
				m_testSet.clear();
			}
			
			
		} 
		*/if(m_randomFold==true){
			
			perf = new double[k];
			ArrayList<_Doc> neweggDocs= new ArrayList<_Doc>() ;
			
			for(_Doc d:m_corpus.getCollection()){
				if(d.getAnnotationType()==annotationType.ANNOTATED || d.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED)
					neweggDocs.add(d);
			}
			
			// shuffling only newEgg docs
			int[] masks = new int [neweggDocs.size()]; 
			Random rand = new Random();
			for(int i=0; i< masks.length; i++) {
				masks[i] = rand.nextInt(k);
			}
			
			//Use this loop to iterate all the k folders, set the train set and test set.
			for (int i = 0; i < k; i++) {
				
				// adding one fold of train and test from newEgg
				for (int j = 0; j < masks.length; j++) {
					if( masks[j]==i ){ 
						m_testSet.add(neweggDocs.get(j));
					}
					else{ 
						m_trainSet.add(neweggDocs.get(j));
					}
				}
				
				//adding all the data from amazon in trainset
				int index = 0;
				
				for(int a=0; a<=5000;a=a+1000){
					System.out.println("a:"+ a);
					
					if(a!=0){
						int m = 0;
						int l = index;
						for(; l<m_corpus.getCollection().size();l++){
							_Doc d = m_corpus.getCollection().get(l);
							if(m>1000)
								break;
							if(d.getAnnotationType()==annotationType.UNANNOTATED){
								m_trainSet.add(d);
								m++;
							}
							
						}
						index = l;

					}
				
					System.out.println("Fold number "+i);
					System.out.println("Train Set Size "+m_trainSet.size());
					System.out.println("Test Set Size "+m_testSet.size());

					if(this instanceof HTSM)
						normalizeFeature();
					long start = System.currentTimeMillis();
					EM();
					perf[i] = Evaluation(i);
					System.out.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
					
					
					int precision_recall_local [][] = {{0,0},{0,0}};
					for(_Doc d: m_testSet){
						
						for(int s=0; s<d.getSenetenceSize(); s++){
							int actualLabel = d.getSentence(s).getSentenceSenitmentLabel();
							if(actualLabel==-1) continue;
							int predictedLabel = d.getSentence(s).getSentencePredictedSenitmentLabel();
							precision_recall_local[actualLabel][predictedLabel]++;
						}
					}
				    
				    double pros_precision = (double)precision_recall_local[0][0]/(precision_recall_local[0][0] + precision_recall_local[1][0]);
					double cons_precision = (double)precision_recall_local[1][1]/(precision_recall_local[0][1] + precision_recall_local[1][1]);
					
					double pros_recall = (double)precision_recall_local[0][0]/(precision_recall_local[0][0] + precision_recall_local[0][1]);
					double cons_recall = (double)precision_recall_local[1][1]/(precision_recall_local[1][0] + precision_recall_local[1][1]);
					
					double pros_f1 = 2/(1/pros_precision + 1/pros_recall);
					double cons_f1 = 2/(1/cons_precision + 1/cons_recall);
					
					double result [] = {pros_f1,cons_f1};
					
					tpt.put(i+a, result);
				    
					System.out.format("%s Train/Test finished in %.2f seconds with pros F1 %.4f and cons F1 %.4f ..\n", this.toString(), (System.currentTimeMillis()-start)/1000.0, pros_f1, cons_f1);
					infoWriter.format("%s Train/Test finished in %.2f seconds with pros F1 %.4f and cons F1 %.4f ..\n", this.toString(), (System.currentTimeMillis()-start)/1000.0, pros_f1, cons_f1);
					
					
				}// amazon trainSize loop ends
		
				m_trainSet.clear();
				m_testSet.clear();
			}
			
			double trainSizeSum [][] = {{0,0},{0,0},{0,0},{0,0},{0,0},{0,0}};
			for(int size = 0; size<=5; size = size+1){
				for(int fold = 0; fold<k;fold++){
					double tmp [] = tpt.get(fold+size*1000);
					trainSizeSum[size][0] += tmp[0];
					trainSizeSum[size][1] += tmp[1];
				}
			}
			
			infoWriter.println("Pros F1 Folding Matrix");
			
			for(int size = 0; size<=5; size = size+1){
				for(int fold = 0; fold<k;fold++){
					infoWriter.print(tpt.get(fold+size*1000)[0]+", ");
					
				}
			}
			
			
			infoWriter.println("\nCons F1 Folding Matrix");
			
			for(int size = 0; size<=5; size = size+1){
				for(int fold = 0; fold<k;fold++){
					infoWriter.print(tpt.get(fold+size*1000)[1]+", ");
					
				}
			}
			
			infoWriter.println();
			for(int size = 0; size<=5; size = size+1){
				System.out.println("Train size:"+size*1000 +", Pros F1:"+trainSizeSum[size][0]/k+", Cons F1:"+trainSizeSum[size][1]/k);
				infoWriter.println("Train size:"+size*1000 +", Pros F1:"+trainSizeSum[size][0]/k+", Cons F1:"+trainSizeSum[size][1]/k);
				
			}
		}
		
		
		else {
			k = 1;
			perf = new double[k];
		    
			int totalNewqEggDoc = 0;
		    int totalAmazonDoc = 0;
			for(_Doc d:m_corpus.getCollection()){
				if(d.getAnnotationType()==annotationType.ANNOTATED || d.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED ){
					newEggRatingCount[d.getYLabel()]++;
					totalNewqEggDoc++;
					}
				else if(d.getAnnotationType()==annotationType.UNANNOTATED){
					amazonRatingCount[d.getYLabel()]++;
					totalAmazonDoc++;
				}
			}
			System.out.println("Total New Egg Doc:"+totalNewqEggDoc);
			infoWriter.println("Total New Egg Doc:"+totalNewqEggDoc);
			System.out.println("Total Amazon Doc:"+ totalAmazonDoc);
			infoWriter.println("Total Amazon Doc:"+ totalAmazonDoc);
			
			int amazonTrainSize = 0;
			int amazonTestSize = 0;
			int newEggTrainSize = 0;
			int newEggTestSize = 0;
			
			for(_Doc d:m_corpus.getCollection()){
				
				if(d.getAnnotationType()==annotationType.UNANNOTATED){ // from Amazon
					int rating = d.getYLabel();
					
					if(amazonTrainsetRatingCount[rating]<=0.8*amazonRatingCount[rating]){
						m_trainSet.add(d);
						amazonTrainsetRatingCount[rating]++;
						amazonTrainSize++;
					}else{
						m_testSet.add(d);
						amazonTestSize++;
					}
				}
				
				if(m_LoadnewEggInTrain==true && (d.getAnnotationType()==annotationType.ANNOTATED || d.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED)) {
					
					int rating = d.getYLabel();
					if(newEggTrainsetRatingCount[rating]<=0.8*newEggRatingCount[rating]){
						m_trainSet.add(d);
						newEggTrainsetRatingCount[rating]++;
						newEggTrainSize++;
					}else{
						m_testSet.add(d);
						newEggTestSize++;
					}
					
				}
				if(m_LoadnewEggInTrain==false && (d.getAnnotationType()==annotationType.ANNOTATED || d.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED )) {
					int rating = d.getYLabel();
					if(newEggTrainsetRatingCount[rating]<=0.8*newEggRatingCount[rating]){
						// Do nothing simply ignore it make for different set
						//m_trainSet.add(d);
						newEggTrainsetRatingCount[rating]++;
						//newEggTrainSize++;
					}else{
						m_testSet.add(d);
						newEggTestSize++;
					}
				}
			}
			
			if(m_trainSetForASUMJST){
				System.out.println("Starting Generating File for JST & ASUM");
				generateFileForJSTASUM();
				System.out.println("Finished Generating File for JST & ASUM ");
				System.exit(-1);
			}
			
			
			
			System.out.println("Neweeg Train Size: "+newEggTrainSize+" test Size: "+newEggTestSize);
			infoWriter.println("Neweeg Train Size: "+newEggTrainSize+" test Size: "+newEggTestSize);
			
			System.out.println("Amazon Train Size: "+amazonTrainSize+" test Size: "+amazonTestSize);
			infoWriter.println("Amazon Train Size: "+amazonTrainSize+" test Size: "+amazonTestSize);
			
			for(int i=0; i<amazonTrainsetRatingCount.length; i++){
				System.out.println("Rating ["+i+"] and Amazon TrainSize:"+amazonTrainsetRatingCount[i]+" and newEgg TrainSize:"+newEggTrainsetRatingCount[i]);
				infoWriter.println("Rating ["+i+"] and Amazon TrainSize:"+amazonTrainsetRatingCount[i]+" and newEgg TrainSize:"+newEggTrainsetRatingCount[i]);
			}
	
			System.out.println("Combined Train Set Size "+m_trainSet.size());
			infoWriter.println("Combined Train Set Size "+m_trainSet.size());
			System.out.println("Combined Test Set Size "+m_testSet.size());
			infoWriter.println("Combined Test Set Size "+m_testSet.size());
			
			if(this instanceof HTSM)
				normalizeFeature();
			long start = System.currentTimeMillis();
			EM();
			perf[0] = Evaluation(k);
			System.out.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
			infoWriter.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
			
		}
		//output the performance statistics
		double mean = Utils.sumOfArray(perf)/k, var = 0;
		for(int i=0; i<perf.length; i++)
			var += (perf[i]-mean) * (perf[i]-mean);
		var = Math.sqrt(var/k);
		System.out.format("Perplexity %.3f+/-%.3f\n", mean, var);
		infoWriter.format("Perplexity %.3f+/-%.3f\n", mean, var);
		
		double prosSum = 0.0;
		double consSum = 0.0;
		
		for(int a=0; a<k;a++){
			prosSum+=prosConsPerFold[a][0];
			consSum+=prosConsPerFold[a][1];
		}
		
		double prosMean = prosSum/k;
		double consMean = consSum/k;
		
		double prosVar = 0.0;
		double consVar = 0.0;
		
		for(int a=0; a<k;a++){
			prosVar+= (prosConsPerFold[a][0] - prosMean)*(prosConsPerFold[a][0] - prosMean);
			consVar+= (prosConsPerFold[a][1] - consMean)*(prosConsPerFold[a][1] - consMean);
		}
		
		prosVar = Math.sqrt(prosVar/k);
		consVar = Math.sqrt(consVar/k);
		
		infoWriter.format("Cross Validated for %d fold  Pros:%f, Cons:%f\n", k, prosSum/k, consSum/k);
		infoWriter.format("Cross Validated foe %d fold Pros Var:%f, Cons Var:%f\n",k, prosVar, consVar);
		
		System.out.format("Cross Validated for %d fold  Pros:%f, Cons:%f\n", k, prosSum/k, consSum/k);
		System.out.format("Cross Validated foe %d fold Pros Var:%f, Cons Var:%f\n",k, prosVar, consVar);
		
		
		infoWriter.flush();
		infoWriter.close();
	}	
}
