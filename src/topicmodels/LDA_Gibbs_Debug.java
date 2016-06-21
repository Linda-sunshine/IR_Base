package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import structures.MyPriorityQueue;
import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._RankItem;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class LDA_Gibbs_Debug extends LDA_Gibbs{
	Random m_rand;
	int m_burnIn; // discard the samples within burn in period
	int m_lag; // lag in accumulating the samples
	
	double[] m_topicProbCache;
	
	//used to compute loglikelihood
	languageModelBaseLine m_LM; 
	
	double m_tau;
	
	//all computation here is not in log-space!!!
	public LDA_Gibbs_Debug(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, 
			int number_of_topics, double alpha, double burnIn, int lag, double ksi, double tau) {
		super( number_of_iteration,  converge,  beta,
			 c,  lambda, number_of_topics,  alpha,  burnIn,  lag);
		
		m_rand = new Random();
		m_burnIn = (int) (burnIn * number_of_iteration);
		m_lag = lag;
		
		m_topicProbCache = new double[number_of_topics];
		m_LM = new languageModelBaseLine(c, ksi);
		m_tau = tau;
	}
	
	protected void initialize_probability(Collection<_Doc> collection) {
		createSpace();
		for(int i=0; i< number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		Arrays.fill(m_sstat, d_beta*vocabulary_size);
		
		for(_Doc d: collection){
			if(d instanceof _ParentDoc) {
				for(_Stn stnObj: d.getSentences()){
					stnObj.setTopicsVct(number_of_topics);
				}	
				d.setTopics4Gibbs(number_of_topics, d_alpha);
			}else if(d instanceof _ChildDoc){
				((_ChildDoc) d).setTopics4Gibbs_LDA(number_of_topics, d_alpha);
			}
			
			
			for(_Word w:d.getWords()) {
				word_topic_sstat[w.getTopic()][w.getIndex()] ++;
				m_sstat[w.getTopic()] ++;
			}
		}
		
		imposePrior();
	}
	
	protected void collectStats(_Doc d) {
		for(int k=0; k<this.number_of_topics; k++)
			d.m_topics[k] += d.m_sstat[k];
		if(d instanceof _ParentDoc){
			((_ParentDoc) d).collectTopicWordStat();
		}
	}
	
	protected double wordByTopicProb(int tid, int wid){
		return word_topic_sstat[tid][wid]/m_sstat[tid];
	}
	
	protected double topicInDocProb(int tid, _Doc d){
		return (d.m_sstat[tid]);
	}
	
	protected void finalEst(){
		super.finalEst();
	}

	protected void estThetaInDoc(_Doc d) {
		super.estThetaInDoc(d);
		if(d instanceof _ParentDoc){
//			estParentStnTopicProportion((_ParentDoc)d);
			((_ParentDoc)d).estStnTheta();
		}
	}
	
	protected void estParentStnTopicProportion(_ParentDoc pDoc) {
//		 pDoc.estStnTheta();
		for (_Stn stnObj : pDoc.getSentences()) {
			estStn(stnObj, pDoc);
		}
	}
	
	public void estStn(_Stn stnObj,  _ParentDoc d){
		int i=0;
		initStn(stnObj);
		do{
			calculateStn_E_step(stnObj, d);
			if(i>m_burnIn && i%m_lag == 0){
				collectStnStats(stnObj, d);
			}
			
		}while(++i<number_of_iteration);
		
		Utils.L1Normalization(stnObj.m_topics);
	}
	
	public void initStn(_Stn stnObj){
		stnObj.setTopicsVct(number_of_topics);
	}
	
	public void calculateStn_E_step( _Stn stnObj, _ParentDoc d){
		stnObj.permuteStn();
		
		double normalizedProb = 0;
		int wid, tid;
		for(_Word w: stnObj.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
	
			stnObj.m_topicSstat[tid] --;
		
			normalizedProb = 0;
			
			double pWordTopic = 0;
			
			for(tid=0; tid<number_of_topics; tid++){
				pWordTopic = wordByTopicProb(tid, wid);
				double pTopic = parentTopicInStnProb(tid, stnObj, d);
				
				m_topicProbCache[tid] = pWordTopic*pTopic;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<m_topicProbCache.length; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb <= 0)
					break;
			}
			
			if(tid==m_topicProbCache.length)
				tid --;
			

			w.setTopic(tid);
			stnObj.m_topicSstat[tid] ++;
			
		}
		
	}
	
	public double parentTopicInStnProb(int tid, _Stn stnObj, _ParentDoc d){
//		return (d_alpha+stnObj.m_topicSstat[tid])/(number_of_topics*d_alpha+stnObj.getLength());
		return (d_alpha + d.m_topics[tid]+stnObj.m_topicSstat[tid])/(number_of_topics*d_alpha+1+stnObj.getLength());
	}
	
	public void collectStnStats(_Stn stnObj, _ParentDoc d){
		for(int k=0; k<number_of_topics; k++){
//			stnObj.m_topics[k] += stnObj.m_topicSstat[k]+d_alpha;
			stnObj.m_topics[k] += stnObj.m_topicSstat[k]+d_alpha+d.m_topics[k];
		}
	}
	
	public void crossValidation(int k) {
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		
		double[] perf = null;
		
		_Corpus parentCorpus = new _Corpus();
		ArrayList<_Doc> docs = m_corpus.getCollection();
		ArrayList<_ParentDoc> parentDocs = new ArrayList<_ParentDoc>();
		for(_Doc d: docs){
			if(d instanceof _ParentDoc){
				parentCorpus.addDoc(d);
				parentDocs.add((_ParentDoc) d);
			}
		}
		
		System.out.println("size of parent docs\t"+parentDocs.size());
		
		parentCorpus.setMasks();
		if(m_randomFold==true){
			perf = new double[k];
			parentCorpus.shuffle(k);
			int[] masks = parentCorpus.getMasks();
			
			for(int i=0; i<k; i++){
				for(int j=0; j<masks.length; j++){
					if(masks[j] == i){
						m_testSet.add(parentDocs.get(j));
					}else {
						m_trainSet.add(parentDocs.get(j));
						for(_ChildDoc d: parentDocs.get(j).m_childDocs){
							m_trainSet.add(d);
						}
					}
					
				}
				
//				writeFile(i, m_trainSet, m_testSet);
				System.out.println("Fold number "+i);
				infoWriter.println("Fold number "+i);
				
				System.out.println("Train Set Size "+m_trainSet.size());
				infoWriter.println("Train Set Size "+m_trainSet.size());
				
				System.out.println("Test Set Size "+m_testSet.size());
				infoWriter.println("Test Set Size "+m_testSet.size());

				long start = System.currentTimeMillis();
				EM();
				perf[i] = Evaluation(i);
				
				System.out.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
				infoWriter.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
				
				if(i<k-1){
					m_trainSet.clear();
					m_testSet.clear();	
				}
			}
			
		}
		double mean = Utils.sumOfArray(perf)/k, var = 0;
		for(int i=0; i<perf.length; i++)
			var += (perf[i]-mean) * (perf[i]-mean);
		var = Math.sqrt(var/k);
		System.out.format("Perplexity %.3f+/-%.3f\n", mean, var);
		infoWriter.format("Perplexity %.3f+/-%.3f\n", mean, var);
	}
	
	public double Evaluation(int i) {
		m_collectCorpusStats = false;
		double perplexity = 0, loglikelihood, totalWords=0, sumLikelihood = 0;
		
		System.out.println("In Normal");
		
		for(_Doc d:m_testSet) {				
			loglikelihood = inference(d);
			sumLikelihood += loglikelihood;
			perplexity += loglikelihood;
			totalWords += d.getDocTestLength();
			for(_ChildDoc cDoc: ((_ParentDoc)d).m_childDocs){
				totalWords += cDoc.getDocTestLength();
			}
		}
		System.out.println("total Words\t"+totalWords+"perplexity\t"+perplexity);
		infoWriter.println("total Words\t"+totalWords+"perplexity\t"+perplexity);
		perplexity /= totalWords;
		perplexity = Math.exp(-perplexity);
		sumLikelihood /= m_testSet.size();

		System.out.format("Test set perplexity is %.3f and log-likelihood is %.3f\n", perplexity, sumLikelihood);
		infoWriter.format("Test set perplexity is %.3f and log-likelihood is %.3f\n", perplexity, sumLikelihood);
		return perplexity;		
	}
	
	@Override
//	public double inference(_Doc pDoc){
//		ArrayList<_Doc> sampleTestSet = new ArrayList<_Doc>();
//		
//		initTest(sampleTestSet, pDoc);
//	
//		double logLikelihood = 0.0, count = 0;
//		int  iter = 0;
//		do {
//			int t;
//			_Doc tmpDoc;
//			for(int i=sampleTestSet.size()-1; i>1; i--) {
//				t = m_rand.nextInt(i);
//				
//				tmpDoc = sampleTestSet.get(i);
//				sampleTestSet.set(i, sampleTestSet.get(t));
//				sampleTestSet.set(t, tmpDoc);			
//			}
//			
//			for(_Doc doc: sampleTestSet)
//				calculate_E_step(doc);
//			
//			if (iter>m_burnIn && iter%m_lag==0){
//				double tempLogLikelihood = 0;
//				for(_Doc doc: sampleTestSet){
//					collectStats(doc);
//					// tempLogLikelihood += calculate_log_likelihood(doc);
//				}
//				count ++;
//				// if (logLikelihood == 0)
//				// logLikelihood = tempLogLikelihood;
//				// else {
//				//
//				// logLikelihood = Utils.logSum(logLikelihood,
//				// tempLogLikelihood);
//				// }
//			}
//		} while (++iter<this.number_of_iteration);
//
//		for(_Doc doc: sampleTestSet){
//			estThetaInDoc(doc);
//			logLikelihood += calculate_test_log_likelihood(doc);
//		}
//		
//		return logLikelihood;
//	}

	public double inference(_Doc pDoc){
		ArrayList<_Doc> sampleTestSet = new ArrayList<_Doc>();
		
		initTest(sampleTestSet, pDoc);
	
		double logLikelihood = 0.0, count = 0;
		inferenceParentDoc((_ParentDoc)pDoc);
		logLikelihood = inferenceChildDoc((_ParentDoc)pDoc);
		
		return logLikelihood;
	}

	protected double inferenceParentDoc(_ParentDoc pDoc){
		double likelihood = 0;
		int iter = 0;
		do{
			
			calculate_E_step(pDoc);
			
			if(iter>m_burnIn && iter%m_lag==0){
				collectStats(pDoc);
			}
			
		}while(++iter<number_of_iteration);
		
		return likelihood;
	}
	
	protected double inferenceChildDoc(_ParentDoc pDoc){
		double likelihood = 0;
		int iter = 0;
		
		do{
			int t;
			_ChildDoc tmpDoc;
			for(int i=pDoc.m_childDocs.size()-1; i>1; i--){
				t = m_rand.nextInt(i);
				
				tmpDoc = pDoc.m_childDocs.get(i);
				pDoc.m_childDocs.set(i, pDoc.m_childDocs.get(t));
				pDoc.m_childDocs.set(t, tmpDoc);
			}
			
			for(_Doc doc:pDoc.m_childDocs){
				calculate_E_step(doc);
			}
			
			if(iter>m_burnIn && iter%m_lag==0){
				for(_Doc doc:pDoc.m_childDocs){
					collectStats(doc);
				}
			}
			
		}while(++iter<number_of_iteration);
		
		for(_Doc doc:pDoc.m_childDocs){
			likelihood += calculate_test_log_likelihood(doc);
		}
		
		return likelihood;
	}
	
	protected double calculate_test_log_likelihood(_Doc d){
		if(d instanceof _ParentDoc)
			return testLogLikelihoodByIntegrateTopics((_ParentDoc)d);
		else
			return testLogLikelihoodByIntegrateTopics((_ChildDoc)d);
	}
	
	protected double testLogLikelihoodByIntegrateTopics(_ParentDoc d){
		double docLogLikelihood = 0.0;
		
		double docInferLen = d.getDocInferLength();
		
		for(_Word w:d.getTestWords()){
			int wid = w.getIndex();
	
			double wordLogLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = wordByTopicProb(k, wid)
						* topicInDocProb(k, d)
						/ (docInferLen + number_of_topics * d_alpha);
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			docLogLikelihood += Math.log(wordLogLikelihood);
		}
		
		return docLogLikelihood;
	}

	protected double testLogLikelihoodByIntegrateTopics(_ChildDoc d){
		double docLogLikelihood = 0.0;
		
		double docInferLen = d.getDocInferLength();
		for(_Word w:d.getTestWords()){
			int wid = w.getIndex();
	
			double wordLogLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = wordByTopicProb(k, wid)
						* topicInDocProb(k, d)
						/ (docInferLen + number_of_topics * d_alpha);
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			docLogLikelihood += Math.log(wordLogLikelihood);
		}
		
		return docLogLikelihood;
	}
	
	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d){
			
		_ParentDoc pDoc = (_ParentDoc)d;
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}
		
		int testLength = 0;
//		int testLength = (int)(m_testWord4PerplexityProportion*d.getTotalDocLength());
		pDoc.setTopics4GibbsTest(number_of_topics, d_alpha, testLength);
		sampleTestSet.add(pDoc);
		
		pDoc.createSparseVct4Infer();
	
		for(_ChildDoc cDoc: pDoc.m_childDocs){
			testLength = (int)(m_testWord4PerplexityProportion*cDoc.getTotalDocLength());
			cDoc.setTopics4GibbsTest(number_of_topics, d_alpha, testLength);
			sampleTestSet.add(cDoc);
			cDoc.createSparseVct4Infer();
			
		}
	}
	
	protected double calculate_log_likelihood(){
		double logLikelihood = 0.0;
		
		for(_Doc d: m_trainSet){
			logLikelihood += calculate_log_likelihood(d);
		}
		
		return logLikelihood;
	}
	
	public double calculate_log_likelihood(_Doc d){
		double docLogLikelihood = 0.0;
		_SparseFeature[] fv = d.getSparse();
		
		for(int j=0; j<fv.length; j++){
			int wid = fv[j].getIndex();
			double value = fv[j].getValue();
			
			double wordLogLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){

				double wordPerTopicLikelihood = wordByTopicProb(k, wid)
						* topicInDocProb(k, d)
						/ (d.getTotalDocLength() + number_of_topics * d_alpha);
				wordLogLikelihood += wordPerTopicLikelihood;

			}
			if(wordLogLikelihood < 1e-10){
				wordLogLikelihood += 1e-10;
				System.out.println("small log likelihood per word");
			}

			wordLogLikelihood = Math.log(wordLogLikelihood);

			docLogLikelihood += value*wordLogLikelihood;
		}

		return docLogLikelihood;
	}
	
	@Override
 	public void printTopWords(int k, String betaFile) {

		double loglikelihood = calculate_log_likelihood();
		System.out.format("Final Log Likelihood %.3f\t", loglikelihood);
		
		String filePrefix = betaFile.replace("topWords.txt", "");
		debugOutput(filePrefix);
		
		Arrays.fill(m_sstat, 0);

		System.out.println("print top words");
		for (_Doc d : m_trainSet) {
			for (int i = 0; i < number_of_topics; i++)
				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
						: d.m_topics[i];	
		}

		Utils.L1Normalization(m_sstat);
		
		try {
			System.out.println("beta file");
			PrintWriter betaOut = new PrintWriter(new File(betaFile));
			for (int i = 0; i < topic_term_probabilty.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j),
							topic_term_probabilty[i][j]));

				betaOut.format("Topic %d(%.3f):\t", i, m_sstat[i]);
				for (_RankItem it : fVector) {
					betaOut.format("%s(%.3f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
					System.out.format("%s(%.3f)\t", it.m_name,
						m_logSpace ? Math.exp(it.m_value) : it.m_value);
				}
				betaOut.println();
				System.out.println();
			}
	
			betaOut.flush();
			betaOut.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}
	
	public void debugOutput(String filePrefix){

		File topicFolder = new File(filePrefix + "topicAssignment");
	
		if (!topicFolder.exists()) {
			System.out.println("creating directory" + topicFolder);
			topicFolder.mkdir();
		}

		File childTopKStnFolder = new File(filePrefix+"topKStn");
		if(!childTopKStnFolder.exists()){
			System.out.println("creating top K stn directory\t"+childTopKStnFolder);
			childTopKStnFolder.mkdir();
		}
		
		File stnTopKChildFolder = new File(filePrefix+"topKChild");
		if(!stnTopKChildFolder.exists()){
			System.out.println("creating top K child directory\t"+stnTopKChildFolder);
			stnTopKChildFolder.mkdir();
		}
		
		int topKStn = 10;
		int topKChild = 10;
		for (_Doc d : m_trainSet) {
			if(d instanceof _ParentDoc){
				printParentTopicAssignment(d, topicFolder);
			}else if(d instanceof _ChildDoc){
				printChildTopicAssignment(d, topicFolder);
			}
//			if(d instanceof _ParentDoc){
//				printTopKChild4Stn(topKChild, (_ParentDoc)d, stnTopKChildFolder);
//				printTopKStn4Child(topKStn, (_ParentDoc)d, childTopKStnFolder);
//			}
		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";
	
		printParameter(parentParameterFile, childParameterFile, m_trainSet);
		printTestParameter4Spam(filePrefix);

		String similarityFile = filePrefix+"topicSimilarity.txt";
		discoverSpecificComments(similarityFile);
		printEntropy(filePrefix);
		printTopKChild4Parent(filePrefix, topKChild);
		printTopKChild4Stn(filePrefix, topKChild);
		printTopKChild4StnWithHybrid(filePrefix, topKChild);
		printTopKChild4StnWithHybridPro(filePrefix, topKChild);
		printTopKStn4Child(filePrefix, topKStn);
	}

	protected void discoverSpecificComments(String similarityFile) {
		System.out.println("topic similarity");
	
		try {
			PrintWriter pw = new PrintWriter(new File(similarityFile));

			for (_Doc doc : m_trainSet) {
				if (doc instanceof _ParentDoc) {
					pw.print(doc.getName() + "\t");
					double stnTopicSimilarity = 0.0;
					double docTopicSimilarity = 0.0;
					for (_ChildDoc cDoc : ((_ParentDoc) doc).m_childDocs) {
						pw.print(cDoc.getName() + ":");

						docTopicSimilarity = computeSimilarity(((_ParentDoc) doc).m_topics, cDoc.m_topics);
						pw.print(docTopicSimilarity);
						for (_Stn stnObj:doc.getSentences()) {
						
							stnTopicSimilarity = computeSimilarity(stnObj.m_topics, cDoc.m_topics);
							
							pw.print(":"+(stnObj.getIndex()+1) + ":" + stnTopicSimilarity);
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
	
	double computeSimilarity(double[] topic1, double[] topic2) {
		return Utils.cosine(topic1, topic2);
	}
	
	public void printParentTopicAssignment(_Doc d, File topicFolder) {
		//	System.out.println("printing topic assignment parent documents");
			
		String topicAssignmentFile = d.getName() + ".txt";
		try {

			PrintWriter pw = new PrintWriter(new File(topicFolder,
					topicAssignmentFile));

			for (_Stn stnObj : d.getSentences()) {
				pw.print(stnObj.getIndex() + "\t");
				for (_Word w : stnObj.getWords()) {
					int index = w.getIndex();
					int topic = w.getTopic();
					String featureName = m_corpus.getFeature(index);
//							System.out.println("test\t"+featureName+"\tdocName\t"+d.getName());			
					pw.print(featureName + ":" + topic + "\t");
				}
				pw.println();
			}
		
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public void printChildTopicAssignment(_Doc d, File topicFolder) {
		//	System.out.println("printing topic assignment parent documents");
			
		String topicAssignmentFile = d.getName() + ".txt";
		try {

			PrintWriter pw = new PrintWriter(new File(topicFolder,
					topicAssignmentFile));

			for (_Word w : d.getWords()) {
				int index = w.getIndex();
				int topic = w.getTopic();
				String featureName = m_corpus.getFeature(index);
				pw.print(featureName + ":" + topic + "\t");
				if(featureName=="brain")
					System.out.println("hole\t"+d.getName());
				if(featureName=="moon")
					System.out.println("moon\t"+d.getName());
				if(featureName=="batteri")
					System.out.println("batteri\t"+d.getName());
			}
		
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	protected void printParameter(String parentParameterFile, String childParameterFile, ArrayList<_Doc> docList){
		System.out.println("printing parameter");
		try{
			System.out.println(parentParameterFile);
			System.out.println(childParameterFile);
			
			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(childParameterFile));
			
			for(_Doc d: docList){
				if(d instanceof _ParentDoc){
					parentParaOut.print(d.getName()+"\t");
					parentParaOut.print("topicProportion\t");
					for(int k=0; k<number_of_topics; k++){
						parentParaOut.print(d.m_topics[k]+"\t");
					}
					
					for(_Stn stnObj:d.getSentences()){							
						parentParaOut.print("sentence"+(stnObj.getIndex()+1)+"\t");
						for(int k=0; k<number_of_topics;k++){
							parentParaOut.print(stnObj.m_topics[k]+"\t");
						}
					}
					
					parentParaOut.println();
					
					for (_ChildDoc cDoc : ((_ParentDoc) d).m_childDocs) {
						childParaOut.print(cDoc.getName()+"\t");
	
						childParaOut.print("topicProportion\t");
						for (int k = 0; k < number_of_topics; k++) {
							childParaOut.print(d.m_topics[k] + "\t");
						}
						
						childParaOut.println();		
					}
				}
			}
			
			parentParaOut.flush();
			parentParaOut.close();
			
			childParaOut.flush();
			childParaOut.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	protected void printEntropy(String filePrefix){
		String entropyFile = filePrefix+"entropy.txt";
		boolean logScale = true;
		
		try{
			PrintWriter entropyPW = new PrintWriter(new File(entropyFile));
			
			for(_Doc d: m_trainSet){
				double entropyValue = 0.0;
				entropyValue = Utils.entropy(d.m_topics, logScale);
				entropyPW.print(d.getName()+"\t"+entropyValue);
				entropyPW.println();
			}
			entropyPW.flush();
			entropyPW.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		
	} 
		
	//comment is a query, retrieve stn by topical similarity
	protected HashMap<Integer, Double> rankStn4ChildBySim( _ParentDoc pDoc, _ChildDoc cDoc){

		HashMap<Integer, Double> stnSimMap = new HashMap<Integer, Double>();
		
		for(_Stn stnObj:pDoc.getSentences()){
//			double stnSim = computeSimilarity(cDoc.m_topics, stnObj.m_topics);
//			stnSimMap.put(stnObj.getIndex()+1, stnSim);
//			
			double stnKL = Utils.klDivergence(cDoc.m_topics, stnObj.m_topics);
//			double stnKL = Utils.KLsymmetric(cDoc.m_topics, stnObj.m_topics);
//			double stnKL = Utils.klDivergence(stnObj.m_topics, cDoc.m_topics);
			stnSimMap.put(stnObj.getIndex()+1, -stnKL);
		}
		
		return stnSimMap;
	}
	
	protected HashMap<String, Double> rankChild4StnByHybrid(_Stn stnObj, _ParentDoc pDoc){
		HashMap<String, Double> childLikelihoodMap = new HashMap<String, Double>();
		
		double smoothingMu = m_LM.m_smoothingMu;
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			double cDocLen = cDoc.getTotalDocLength();
			_SparseFeature[] fv = cDoc.getSparse();
			
			double stnLogLikelihood = 0;
			double alphaDoc = smoothingMu/(smoothingMu+cDocLen);
			
			_SparseFeature[] sv = stnObj.getFv();
			for(_SparseFeature svWord:sv){
				double featureLikelihood = 0;
				
				int wid = svWord.getIndex();
				double stnVal = svWord.getValue();
				
				int featureIndex = Utils.indexOf(fv, wid);
				double docVal = 0;
				if(featureIndex!=-1){
					docVal = fv[featureIndex].getValue();
				}
				
				double LMLikelihood = (1-alphaDoc)*docVal/(cDocLen);
				
				LMLikelihood += alphaDoc*m_LM.getReferenceProb(wid);
				
				double TMLikelihood = 0;
				for(int k=0; k<number_of_topics; k++){
//					double likelihoodPerTopic = topic_term_probabilty[k][wid];
//					System.out.println("likelihoodPerTopic1-----\t"+likelihoodPerTopic);
//					
//					likelihoodPerTopic *= cDoc.m_topics[k];
//					System.out.println("likelihoodPerTopic2-----\t"+likelihoodPerTopic);
					TMLikelihood += (word_topic_sstat[k][wid]/m_sstat[k])*(topicInDocProb(k, cDoc)/(d_alpha*number_of_topics+cDocLen));

//					TMLikelihood += topic_term_probabilty[k][wid]*cDoc.m_topics[k];
//					System.out.println("TMLikelihood\t"+TMLikelihood);
				}
				
				featureLikelihood = m_tau*LMLikelihood+(1-m_tau)*TMLikelihood;
//				featureLikelihood = TMLikelihood;
				featureLikelihood = Math.log(featureLikelihood);
				stnLogLikelihood += stnVal*featureLikelihood;
				
			}
			
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}
		
	protected HashMap<String, Double> rankChild4StnByHybridPro(_Stn stnObj, _ParentDoc pDoc){
		HashMap<String, Double> childLikelihoodMap = new HashMap<String, Double>();
		
		double smoothingMu = m_LM.m_smoothingMu;
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			double cDocLen = cDoc.getTotalDocLength();
			
			double stnLogLikelihood = 0;
			double alphaDoc = smoothingMu/(smoothingMu+cDocLen);
			
			_SparseFeature[] fv = cDoc.getSparse();
			_SparseFeature[] sv = stnObj.getFv();
			for(_SparseFeature svWord: sv){
				double wordLikelihood = 0;
				int wid = svWord.getIndex();
				double stnVal = svWord.getValue();
				
				int featureIndex = Utils.indexOf(fv, wid);
				double docVal = 0;
				if(featureIndex!=-1){
					docVal = fv[featureIndex].getValue();
				}
				
				double LMLikelihood = (1-alphaDoc)*docVal/cDocLen;
				LMLikelihood += alphaDoc*m_LM.getReferenceProb(wid);
				
				double TMLikelihood = 0;
				
				for(int k=0; k<number_of_topics; k++){
					double wordPerTopicLikelihood = (word_topic_sstat[k][wid]/m_sstat[k])*(topicInDocProb(k, cDoc)/(d_alpha*number_of_topics+cDocLen));
					TMLikelihood += wordPerTopicLikelihood;
				}
				
				wordLikelihood = m_tau*LMLikelihood+(1-m_tau)*TMLikelihood;
				wordLikelihood = Math.log(wordLikelihood);
				stnLogLikelihood += stnVal*wordLikelihood;
			}
			
			double cosineSim = computeSimilarity(stnObj.m_topics, cDoc.m_topics);
			stnLogLikelihood = m_tau*stnLogLikelihood + (1-m_tau)*cosineSim;

			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		return childLikelihoodMap;
	}
	
	//stn is a query, retrieve comment by likelihood
	protected HashMap<String, Double> rankChild4StnByLikelihood(_Stn stnObj, _ParentDoc pDoc){
		
		HashMap<String, Double>childLikelihoodMap = new HashMap<String, Double>();

		for(_ChildDoc cDoc:pDoc.m_childDocs){
			int cDocLen = cDoc.getTotalDocLength();
			
			double stnLogLikelihood = 0;
			for(_Word w: stnObj.getWords()){
				double wordLikelihood = 0;
				int wid = w.getIndex();
			
				for(int k=0; k<number_of_topics; k++){
					wordLikelihood += (word_topic_sstat[k][wid]/m_sstat[k])*(topicInDocProb(k, cDoc)/(d_alpha*number_of_topics+cDocLen));
//					wordLikelihood += topic_term_probabilty[k][wid]*cDoc.m_topics[k];
				}
				
				stnLogLikelihood += Math.log(wordLikelihood);
			}
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
}

	protected HashMap<String, Double> rankChild4StnByLanguageModel(_Stn stnObj, _ParentDoc pDoc){
		HashMap<String, Double> childLikelihoodMap = new HashMap<String, Double>();
		
		double smoothingMu = m_LM.m_smoothingMu;
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			int cDocLen = cDoc.getTotalDocLength();
			_SparseFeature[] fv = cDoc.getSparse();
			
			double stnLogLikelihood = 0;
			double alphaDoc = smoothingMu/(smoothingMu+cDocLen);
			
			_SparseFeature[] sv = stnObj.getFv();
			for(_SparseFeature svWord:sv){
				double featureLikelihood = 0;
				
				int wid = svWord.getIndex();
				double stnVal = svWord.getValue();
				
				int featureIndex = Utils.indexOf(fv, wid);
				double docVal = 0;
				if(featureIndex!=-1){
					docVal = fv[featureIndex].getValue();
				}
				
				double smoothingProb = (1-alphaDoc)*docVal/(cDocLen);
				
				smoothingProb += alphaDoc*m_LM.getReferenceProb(wid);
				featureLikelihood = Math.log(smoothingProb);
				stnLogLikelihood += stnVal*featureLikelihood;
			}
			
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}
	
	protected HashMap<String, Double> hybridRankChild4Stn(_Stn stnObj, _ParentDoc pDoc){
		HashMap<String, Double> childLikelihoodMapByTM = new HashMap<String, Double>();
		childLikelihoodMapByTM = rankChild4StnByLikelihood(stnObj, pDoc);
		
		HashMap<String, Double> childLikelihoodMapByLM = new HashMap<String, Double>();
		childLikelihoodMapByLM = rankChild4StnByLanguageModel(stnObj, pDoc);
		
		for(String cDocName:childLikelihoodMapByTM.keySet()){
			double TMVal = childLikelihoodMapByTM.get(cDocName);
			double LMVal = childLikelihoodMapByLM.get(cDocName);
			double retrievalScore = m_tau*TMVal+(1-m_tau)*LMVal;
			
			childLikelihoodMapByTM.put(cDocName, retrievalScore);
		}
		
		return childLikelihoodMapByTM;
	}
	
	protected List<Map.Entry<Integer, Double>> sortHashMap4Integer(HashMap<Integer, Double> stnLikelihoodMap, boolean descendOrder){
		List<Map.Entry<Integer, Double>> sortList = new ArrayList<Map.Entry<Integer, Double>>(stnLikelihoodMap.entrySet());
		
		if(descendOrder == true){
			Collections.sort(sortList, new Comparator<Map.Entry<Integer, Double>>() {
				public int compare(Entry<Integer, Double> e1, Entry<Integer, Double> e2){
					return e2.getValue().compareTo(e1.getValue());
				}
			});
		}else{
			Collections.sort(sortList, new Comparator<Map.Entry<Integer, Double>>() {
				public int compare(Entry<Integer, Double> e1, Entry<Integer, Double> e2){
					return e2.getValue().compareTo(e1.getValue());
				}
			});
		}
		
		return sortList;

	}
	
	protected List<Map.Entry<String, Double>> sortHashMap4String(HashMap<String, Double> stnLikelihoodMap, boolean descendOrder){
		List<Map.Entry<String, Double>> sortList = new ArrayList<Map.Entry<String, Double>>(stnLikelihoodMap.entrySet());
		
		if(descendOrder == true){
			Collections.sort(sortList, new Comparator<Map.Entry<String, Double>>() {
				public int compare(Entry<String, Double> e1, Entry<String, Double> e2){
					return e2.getValue().compareTo(e1.getValue());
				}
			});
		}else{
			Collections.sort(sortList, new Comparator<Map.Entry<String, Double>>() {
				public int compare(Entry<String, Double> e1, Entry<String, Double> e2){
					return e2.getValue().compareTo(e1.getValue());
				}
			});
		}
		
		return sortList;

	}

	protected void printTopKChild4Stn(int topK, _ParentDoc pDoc, File topKChildFolder){
		File topKChild4PDocFolder = new File(topKChildFolder, pDoc.getName());
		if(!topKChild4PDocFolder.exists()){
//			System.out.println("creating top K stn directory\t"+topKChild4PDocFolder);
			topKChild4PDocFolder.mkdir();
		}
		
		for(_Stn stnObj:pDoc.getSentences()){
			HashMap<String, Double> likelihoodMap = rankChild4StnByLikelihood(stnObj, pDoc);
			String topChild4StnFile =  (stnObj.getIndex()+1)+".txt";
				
			try{
				int i=0;
				
				PrintWriter pw = new PrintWriter(new File(topKChild4PDocFolder, topChild4StnFile));
				
				for(Map.Entry<String, Double> e: sortHashMap4String(likelihoodMap, true)){
					if(i==topK)
						break;
					pw.print(e.getKey());
					pw.print("\t"+e.getValue());
					pw.println();
					
					i++;
				}
				
				pw.flush();
				pw.close();
			}catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	protected void printTopKStn4Child(int topK, _ParentDoc pDoc, File topKStnFolder){
		File topKStn4PDocFolder = new File(topKStnFolder, pDoc.getName());
		if(!topKStn4PDocFolder.exists()){
//			System.out.println("creating top K stn directory\t"+topKStn4PDocFolder);
			topKStn4PDocFolder.mkdir();
		}
		
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			String topKStn4ChildFile = cDoc.getName()+".txt";
			HashMap<Integer, Double> stnSimMap = rankStn4ChildBySim(pDoc, cDoc);

			try{
				int i=0;
				
				PrintWriter pw = new PrintWriter(new File(topKStn4PDocFolder, topKStn4ChildFile));
				
				for(Map.Entry<Integer, Double> e: sortHashMap4Integer(stnSimMap, true)){
					if(i==topK)
						break;
					pw.print(e.getKey());
					pw.print("\t"+e.getValue());
					pw.println();
					
					i++;
				}
				
				pw.flush();
				pw.close();
			}catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	protected double rankChild4ParentByLikelihood(_ChildDoc cDoc, _ParentDoc pDoc){

		int cDocLen = cDoc.getTotalDocLength();
		_SparseFeature[] fv = pDoc.getSparse();
		
		double docLogLikelihood = 0;
		for(_SparseFeature i: fv){
			int wid = i.getIndex();
			double value = i.getValue();
			
			double wordLogLikelihood = 0;
			
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = (word_topic_sstat[k][wid]/m_sstat[k])*((cDoc.m_sstat[k]+d_alpha)/(d_alpha*number_of_topics+cDocLen));
				wordLogLikelihood += wordPerTopicLikelihood;
			}
						
			docLogLikelihood += value*Math.log(wordLogLikelihood);
		}
	
		return docLogLikelihood;	
	}
	
	protected double rankChild4ParentBySim(_ChildDoc cDoc, _ParentDoc pDoc) {
		double childSim = computeSimilarity(cDoc.m_topics, pDoc.m_topics);

		return childSim;
	}

	protected void printTopKChild4Parent(String filePrefix, int topK) {
		String topKChild4StnFile = filePrefix+"topChild4Parent.txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4StnFile));
			
			for(_Doc d: m_trainSet){
				if(d instanceof _ParentDoc){
					_ParentDoc pDoc = (_ParentDoc)d;
					
					pw.print(pDoc.getName()+"\t");
					
					for(_ChildDoc cDoc:pDoc.m_childDocs){
						double docScore = rankChild4ParentBySim(cDoc,
								pDoc);
				
						pw.print(cDoc.getName() + ":" + docScore + "\t");
						
					}
					
					pw.println();
				}
			}
			pw.flush();
			pw.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	protected void printTopKChild4StnWithHybridPro(String filePrefix, int topK){
		String topKChild4StnFile = filePrefix+"topChild4Stn_hybridPro.txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4StnFile));
			
			m_LM.generateReferenceModel();
			
			for(_Doc d: m_trainSet){
				if(d instanceof _ParentDoc){
					_ParentDoc pDoc = (_ParentDoc)d;
					
					pw.println(pDoc.getName()+"\t"+pDoc.getSenetenceSize());
					
					for(_Stn stnObj:pDoc.getSentences()){
						HashMap<String, Double> likelihoodMap = rankChild4StnByHybridPro(stnObj, pDoc);
						
						pw.print((stnObj.getIndex()+1)+"\t");
						
						for(Map.Entry<String, Double> e: sortHashMap4String(likelihoodMap, true)){
							pw.print(e.getKey());
							pw.print(":"+e.getValue());
							pw.print("\t");
							
						}
						pw.println();		
				
					}
				}
			}
			pw.flush();
			pw.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	protected void printTopKChild4StnWithHybrid(String filePrefix, int topK){
		String topKChild4StnFile = filePrefix+"topChild4Stn_hybrid.txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4StnFile));
			
			m_LM.generateReferenceModel();
			
			for(_Doc d: m_trainSet){
				if(d instanceof _ParentDoc){
					_ParentDoc pDoc = (_ParentDoc)d;
					
					pw.println(pDoc.getName()+"\t"+pDoc.getSenetenceSize());
					
					for(_Stn stnObj:pDoc.getSentences()){
//						HashMap<String, Double> likelihoodMap = rankChild4StnByLikelihood(stnObj, pDoc);
						HashMap<String, Double> likelihoodMap = rankChild4StnByHybrid(stnObj, pDoc);
//						HashMap<String, Double> likelihoodMap = rankChild4StnByLanguageModel(stnObj, pDoc);

						
						int i=0;
						pw.print((stnObj.getIndex()+1)+"\t");
						
						for(Map.Entry<String, Double> e: sortHashMap4String(likelihoodMap, true)){
//							if(i==topK)
//								break;
							pw.print(e.getKey());
							pw.print(":"+e.getValue());
							pw.print("\t");
							
							i++;
						}
						pw.println();		
				
					}
				}
			}
			pw.flush();
			pw.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	protected void printTopKChild4Stn(String filePrefix, int topK){
		String topKChild4StnFile = filePrefix+"topChild4Stn.txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4StnFile));
			
//			m_LM.generateReferenceModel();
			
			for(_Doc d: m_trainSet){
				if(d instanceof _ParentDoc){
					_ParentDoc pDoc = (_ParentDoc)d;
					
					pw.println(pDoc.getName()+"\t"+pDoc.getSenetenceSize());
					
					for(_Stn stnObj:pDoc.getSentences()){
						HashMap<String, Double> likelihoodMap = rankChild4StnByLikelihood(stnObj, pDoc);
//						HashMap<String, Double> likelihoodMap = rankChild4StnByHybrid(stnObj, pDoc);
//						HashMap<String, Double> likelihoodMap = rankChild4StnByLanguageModel(stnObj, pDoc);

						
						int i=0;
						pw.print((stnObj.getIndex()+1)+"\t");
						
						for(Map.Entry<String, Double> e: sortHashMap4String(likelihoodMap, true)){
//							if(i==topK)
//								break;
							pw.print(e.getKey());
							pw.print(":"+e.getValue());
							pw.print("\t");
							
							i++;
						}
						pw.println();		
				
					}
				}
			}
			pw.flush();
			pw.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}

	}

	protected void printTopKStn4Child(String filePrefix, int topK){
		String topKStn4ChildFile = filePrefix+"topStn4Child.txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topKStn4ChildFile));
			
			for(_Doc d: m_trainSet){
				if(d instanceof _ParentDoc){
					_ParentDoc pDoc = (_ParentDoc)d;
					
					pw.println(pDoc.getName()+"\t"+pDoc.m_childDocs.size());
					
					for(_ChildDoc cDoc: pDoc.m_childDocs){
						HashMap<Integer, Double>stnSimMap = rankStn4ChildBySim(pDoc, cDoc);
						int i = 0;
						
						pw.print(cDoc.getName()+"\t");
						for(Map.Entry<Integer, Double> e: sortHashMap4Integer(stnSimMap, true)){
//							if(i==topK)
//								break;
							pw.print(e.getKey());
							pw.print(":"+e.getValue());
							pw.print("\t");
							
							i++;
						}
						pw.println();
					}
				}
			}
			pw.flush();
			pw.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	public void EMonCorpus(){
		separateTrainTest4Spam();
		EM();
		mixTest4Spam();
		inferenceTest4Spam();
	}
	
	public void separateTrainTest4Dynamic() {
		int cvFold = 10;
		ArrayList<_Doc> parentTrainSet = new ArrayList<_Doc>();
		double avgCommentNum = 0;
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		for(_Doc d:m_corpus.getCollection()){
			if(d instanceof _ParentDoc){
				int childSize = ((_ParentDoc)d).m_childDocs.size();
				if(childSize<10){
					parentTrainSet.add(d);
				}else{
					m_testSet.add(d);
					avgCommentNum += childSize;
				}
			}
		}
		
		System.out.println("avg comments for parent doc in testSet\t"+avgCommentNum*1.0/m_testSet.size());
		
		for(_Doc d:parentTrainSet){
			_ParentDoc pDoc = (_ParentDoc) d;
			m_trainSet.add(d);
			pDoc.m_childDocs4Dynamic = new ArrayList<_ChildDoc>();
			for(_ChildDoc cDoc:pDoc.m_childDocs){
				m_trainSet.add(cDoc);
				pDoc.addChildDoc4Dynamics(cDoc);
			}
		}
		System.out.println("m_testSet size\t"+m_testSet.size());
		System.out.println("m_trainSet size\t"+m_trainSet.size());
	}
	
	public void inferenceTest4Dynamical(int commentNum){
		m_collectCorpusStats = false;

		for(_Doc d:m_testSet){
			inferenceDoc4Dynamical(d, commentNum);
		}
	}
	
	public void printTestParameter4Dynamic(int commentNum){
		String xProportionFile = "./data/results/dynamic/testChildXProportion_"+commentNum+".txt";
		
		String parentParameterFile = "./data/results/dynamic/testParentParameter_"+commentNum+".txt";
		String childParameterFile = "./data/results/dynamic/testChildParameter_"+commentNum+".txt";
	
		printParameter(parentParameterFile, childParameterFile, m_testSet);
	}
	
	public void inferenceDoc4Dynamical(_Doc d, int commentNum){
		ArrayList<_Doc> sampleTestSet = new ArrayList<_Doc>();
		initTest4Dynamical(sampleTestSet, d, commentNum);
		
		double tempLikelihood = inference4Doc(sampleTestSet);
	}
	
	//dynamical add comments to sampleTest
	public void initTest4Dynamical(ArrayList<_Doc> sampleTestSet, _Doc d, int commentNum){
		_ParentDoc pDoc = (_ParentDoc)d;
		pDoc.m_childDocs4Dynamic = new ArrayList<_ChildDoc>();
		pDoc.setTopics4Gibbs(number_of_topics, d_alpha);
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}

		sampleTestSet.add(pDoc);
		int count = 0;
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			if(count>=commentNum){
				break;
			}
			count ++;
			cDoc.setTopics4Gibbs_LDA(number_of_topics, d_alpha);
			sampleTestSet.add(cDoc);
			pDoc.addChildDoc4Dynamics(cDoc);
		}
	}
	
	public double inference4Doc(ArrayList<_Doc> sampleTestSet){
		double logLikelihood = 0.0, count = 0;
		int  iter = 0;
		do {
			int t;
			_Doc tmpDoc;
			for(int i=sampleTestSet.size()-1; i>1; i--) {
				t = m_rand.nextInt(i);
				
				tmpDoc = sampleTestSet.get(i);
				sampleTestSet.set(i, sampleTestSet.get(t));
				sampleTestSet.set(t, tmpDoc);			
			}
			
			for(_Doc doc: sampleTestSet)
				calculate_E_step(doc);
			
			if (iter>m_burnIn && iter%m_lag==0){
				for(_Doc doc: sampleTestSet){
					collectStats(doc);
				}
			}
		} while (++iter<this.number_of_iteration);

		for(_Doc doc: sampleTestSet){
			estThetaInDoc(doc);
		}
		
		return logLikelihood;
	}
	
	public void mixTest4Spam() {
		int t = 0, j1 = 0, j2 = 0;
		_ChildDoc tmpDoc1;
		_ChildDoc tmpDoc2;
		for (int i = m_testSet.size() - 1; i > 1; i--) {
			t = m_rand.nextInt(i);

			_ParentDoc pDoc1 = (_ParentDoc) m_testSet.get(i);
			int pDocCDocSize1 = pDoc1.m_childDocs.size();

			j1 = m_rand.nextInt(pDocCDocSize1);
			tmpDoc1 = (_ChildDoc) pDoc1.m_childDocs.get(j1);

			_ParentDoc pDoc2 = (_ParentDoc) m_testSet.get(t);
			int pDocCDocSize2 = pDoc2.m_childDocs.size();

			j2 = m_rand.nextInt(pDocCDocSize2);
			tmpDoc2 = (_ChildDoc) pDoc2.m_childDocs.get(j2);

			pDoc1.m_childDocs.set(j1, tmpDoc2);
			tmpDoc2.setParentDoc(pDoc1);
			pDoc2.m_childDocs.set(j2, tmpDoc1);
			tmpDoc1.setParentDoc(pDoc2);
		}
	}

	public void separateTrainTest4Spam() {
		int cvFold = 10;
		ArrayList<_Doc> parentTrainSet = new ArrayList<_Doc>();
		double avgCommentNum = 0;
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		for (_Doc d : m_corpus.getCollection()) {
			if (d instanceof _ParentDoc) {
				if (m_rand.nextInt(cvFold) != 5) {
					parentTrainSet.add(d);
				} else {
					m_testSet.add(d);
					avgCommentNum += ((_ParentDoc) d).m_childDocs.size();
				}
			}
		}

		System.out.println("avg comments for parent doc in testSet\t"
				+ avgCommentNum * 1.0 / m_testSet.size());

		for (_Doc d : parentTrainSet) {
			_ParentDoc pDoc = (_ParentDoc) d;
			m_trainSet.add(d);
			for (_ChildDoc cDoc : pDoc.m_childDocs) {
				m_trainSet.add(cDoc);
			}
		}
		System.out.println("m_testSet size\t" + m_testSet.size());
		System.out.println("m_trainSet size\t" + m_trainSet.size());
	}

	public void inferenceTest4Spam(){
		m_collectCorpusStats = false;
		
		for(_Doc d:m_testSet){
			inferenceDoc4Spam(d);
		}
	}
	
	public void inferenceDoc4Spam(_Doc d){
		ArrayList<_Doc> sampleTestSet = new ArrayList<_Doc>();
		initTest4Spam(sampleTestSet, d);
		double tempLikelihood = inference4Doc(sampleTestSet);
	}
	
	public void initTest4Spam(ArrayList<_Doc> sampleTestSet, _Doc d){
		_ParentDoc pDoc = (_ParentDoc)d;
		pDoc.setTopics4Gibbs(number_of_topics, 0);
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}
		sampleTestSet.add(pDoc);
		
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			cDoc.setTopics4Gibbs_LDA(number_of_topics, d_alpha);
			sampleTestSet.add(cDoc);
		}
	}

	public void printTestParameter4Spam(String filePrefix){		
		String parentParameterFile = filePrefix+"testParentParameter.txt";
		String childParameterFile =  filePrefix+"testChildParameter.txt";
	
		printParameter(parentParameterFile, childParameterFile, m_testSet);
	}
}
