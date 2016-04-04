package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

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

public class ParentChild_Gibbs extends LDA_Gibbs_Debug {
	enum MatchPair {
		MP_ChildDoc,
		MP_ChildGlobal,
		MP_ChildLocal
	}
	
	double[] m_gamma, m_topicProbCache;
	double[][] m_xTopicProbCache;
	double m_mu, m_kAlpha;
	
	boolean m_statisticsNormalized = false;//a warning sign of normalizing statistics before collecting new ones
	
	public ParentChild_Gibbs(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double mu) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag);

		m_mu = mu;
		m_kAlpha = d_alpha * number_of_topics;
		
		if (gamma!=null) {
			m_gamma = new double[gamma.length];
			System.arraycopy(gamma, 0, m_gamma, 0, gamma.length);
		}
		m_topicProbCache = new double[number_of_topics];
		m_xTopicProbCache = new double[gamma.length][number_of_topics];
	}
	
	@Override
	public String toString(){
		return String.format("Parent Child topic model [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta, m_gamma[0], m_gamma[1]);
	}
	
	protected void computeMu4Doc(_ChildDoc d){
		_ParentDoc tempParent = d.m_parentDoc;
		double mu = Utils.cosine_values(tempParent.getSparse(), d.getSparse());
		d.setMu(mu);
	}
	
	//will be called before entering EM iterations
	@Override
	protected void initialize_probability(Collection<_Doc> collection){
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		Arrays.fill(m_sstat, d_beta*vocabulary_size); // avoid adding such prior later on
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc){
				for(_Stn stnObj: d.getSentences())
					stnObj.setTopicsVct(number_of_topics);				
			} else if(d instanceof _ChildDoc)
				((_ChildDoc) d).createXSpace(number_of_topics, m_gamma.length);
			
			d.setTopics4Gibbs(number_of_topics, 0);
			for (_Word w:d.getWords()) {
				word_topic_sstat[w.getTopic()][w.getIndex()]++;
				m_sstat[w.getTopic()]++;
			}			
		}
		
		imposePrior();
		
		m_statisticsNormalized = false;
	}
	
	@Override
	public double calculate_E_step(_Doc d){
		d.permutation();
		
		if(d instanceof _ParentDoc)
			sampleInParentDoc((_ParentDoc)d);
		else if(d instanceof _ChildDoc)
			sampleInChildDoc((_ChildDoc)d);
		
		return 0;
	}
	
	protected void sampleInParentDoc(_ParentDoc d){
		int wid, tid;
		double normalizedProb;		
		
		for(_Word w:d.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			
			d.m_sstat[tid] --;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}

			normalizedProb = 0;
			for(tid=0; tid<number_of_topics; tid++){
				double pWordTopic = parentWordByTopicProb(tid, wid);
				double pTopicPdoc = parentTopicInDocProb(tid, d);
				double pTopicCdoc = parentChildInfluenceProb(tid, d);
					
				m_topicProbCache[tid] = pWordTopic * pTopicPdoc * pTopicCdoc;
				normalizedProb += m_topicProbCache[tid];
			}

			normalizedProb *= m_rand.nextDouble();			
			for(tid=0; tid<number_of_topics; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb <= 0)
					break;
			}
			
			if(tid == number_of_topics)
				tid --;
			
			w.setTopic(tid);
			d.m_sstat[tid] ++;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] ++;
				m_sstat[tid] ++;
			}
		}
	}

	//probability of word given topic p(w|z, phi^p, beta)
	protected double parentWordByTopicProb(int tid, int wid){
		return word_topic_sstat[tid][wid] / m_sstat[tid];
	}

	//probability of topic given doc p(z|d, alpha)
	protected double parentTopicInDocProb(int tid, _ParentDoc d){
		return d_alpha + d.m_sstat[tid];
	}
	
	protected double parentChildInfluenceProb(int tid, _ParentDoc pDoc){
		double term = 1.0;
		
		if (tid==0)
			return term;//reference point
		
		double muDp = m_mu / pDoc.getTotalDocLength();
		for (_ChildDoc cDoc : pDoc.m_childDocs) {
			term *= gammaFuncRatio((int)cDoc.m_xTopicSstat[0][tid], muDp, d_alpha+pDoc.m_sstat[tid]*muDp) 
					/ gammaFuncRatio((int)cDoc.m_xTopicSstat[0][0], muDp, d_alpha+pDoc.m_sstat[0]*muDp);		
		} 

		return term;
	}
	
	double gammaFuncRatio(int nc, double muDp, double alphaMuNp) {
		if (nc==0)
			return 1.0;
		
		double result = 1.0;
		for(int n=1; n<=nc; n++) 
			result *= 1 + muDp / (alphaMuNp + n);
		return result;
	}

	protected void sampleInChildDoc(_ChildDoc d){
		int wid, tid, xid;		
		double normalizedProb;
		
		for(_Word w:d.getWords()){			
			wid = w.getIndex();
			tid = w.getTopic();
			xid = w.getX();
			
			d.m_xTopicSstat[xid][tid] --;
			d.m_xSstat[xid] --;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid]--;
				m_sstat[tid]--;
			}
			
			normalizedProb = 0;
			double pLambdaOne = childXInDocProb(1, d);
			double pLambdaZero = childXInDocProb(0, d);
			
			for(tid=0; tid<number_of_topics; tid++){
				double pWordTopic = childWordByTopicProb(tid, wid);
				
				//p(z=tid,x=1) from specific
				double pTopicLocal = childTopicInDocProb(tid, 1, d);				
				
				//p(z=tid,x=0) from background
				double pTopicGlobal = childTopicInDocProb(tid, 0, d);				
				
				m_xTopicProbCache[1][tid] = pWordTopic * pTopicLocal * pLambdaOne;
				normalizedProb += m_xTopicProbCache[1][tid];
				
				m_xTopicProbCache[0][tid] = pWordTopic * pTopicGlobal * pLambdaZero;
				normalizedProb += m_xTopicProbCache[0][tid];
			}
			
			boolean finishLoop = false;
			normalizedProb *= m_rand.nextDouble();
			for(xid=0; xid<m_gamma.length; xid++){
				for(tid=0; tid<number_of_topics; tid++){
					normalizedProb -= m_xTopicProbCache[xid][tid];
					if(normalizedProb<=0){
						finishLoop = true;
						break;
					}
				}
				if (finishLoop)
					break;
			}

			if (xid == m_gamma.length)
				xid--;
			
			if (tid == number_of_topics)
				tid--;
			
			w.setTopic(tid);
			w.setX(xid);

			d.m_xTopicSstat[xid][tid] ++;
			d.m_xSstat[xid] ++;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid]++;
				m_sstat[tid]++;
			}			
		}
	}

	//probability of word given topic p(w|z, phi^c, beta)
	protected double childWordByTopicProb(int tid, int wid){
		return word_topic_sstat[tid][wid] / m_sstat[tid];
	}

	//probability of topic in given child doc p(z^c|d, alpha, z^p)
	protected double childTopicInDocProb(int tid, int xid, _ChildDoc d){
		double docLength = d.m_parentDoc.getTotalDocLength();

		if(xid == 1){//local topics
			return (d_alpha + d.m_xTopicSstat[1][tid])
					/(m_kAlpha + d.m_xSstat[1]);
		} else if(xid == 0){//global topics
			return (d_alpha + m_mu*d.m_parentDoc.m_sstat[tid]/docLength + d.m_xTopicSstat[0][tid])
					/(m_kAlpha + m_mu + d.m_xSstat[0]);
		} else
			return Double.NaN;//this branch is impossible
	}

	protected double childXInDocProb(int xid, _ChildDoc d){
		return m_gamma[xid] + d.m_xSstat[xid];
	}	

	@Override
	public void calculate_M_step(int iter){

		if(iter>m_burnIn && iter%m_lag==0){
			if (m_statisticsNormalized) {
				System.err.println("The statistics collector has been normlaized before, cannot further accumulate the samples!");
				System.exit(-1);
			}
			
			for(int i=0; i<this.number_of_topics; i++){
				for(int v=0; v<this.vocabulary_size; v++){
					topic_term_probabilty[i][v] += word_topic_sstat[i][v];//collect the current sample
				}
			}
			
			// used to estimate final theta for each document
			for(_Doc d:m_trainSet){
				if(d instanceof _ParentDoc)
					collectParentStats((_ParentDoc)d);
				else if(d instanceof _ChildDoc)
					collectChildStats((_ChildDoc)d);
			}
		}
	}	
	
	//such statistic collection mechanism makes us unable to normalize the corresponding structure for efficient likelihood computation
	protected void collectParentStats(_ParentDoc d) {
		for (int k = 0; k < this.number_of_topics; k++) 
			d.m_topics[k] += d.m_sstat[k] + d_alpha;
		d.collectTopicWordStat();
	}
	
	protected void collectChildStats(_ChildDoc d) {
		for (int j = 0; j < m_gamma.length; j++)
			d.m_xProportion[j] += d.m_xSstat[j] + m_gamma[j];
		
		double parentDocLength = d.m_parentDoc.getTotalDocLength()/m_mu, gTopic, lTopic;
		// used to output the topK words and parameters
		for (int k = 0; k < number_of_topics; k++) {		
			lTopic = d.m_xTopicSstat[1][k] + d_alpha;
			gTopic = d.m_xTopicSstat[0][k] + d_alpha + d.m_parentDoc.m_sstat[k] / parentDocLength;
			d.m_xTopics[1][k] += lTopic;
			d.m_xTopics[0][k] += gTopic;
			d.m_topics[k] += gTopic + lTopic; // this is just an approximation
		}
		
		for(_Word w:d.getWords())
			w.collectXStats();
	}
	
	//once this function is called, we can no longer accumulate statistics
	@Override
	protected void estThetaInDoc(_Doc d) {
		super.estThetaInDoc(d);
		if (d instanceof _ParentDoc){
			// estimate topic proportion of sentences in parent documents
//			((_ParentDoc) d).estStnTheta();
			estParentStnTopicProportion((_ParentDoc) d);
		} else if (d instanceof _ChildDoc) {
			((_ChildDoc) d).estGlobalLocalTheta();
		}
		m_statisticsNormalized = true;
	}
	
	// protected void estParentStnTopicProportion(_ParentDoc pDoc) {
	// // pDoc.estStnTheta();
	// return;
	// }
	
	// public void estStnByPhi(_Stn stnObj, _ParentDoc pDoc){
	// int stnTopicNum = pDoc.m_topics.length;
	// System.out.println("stnTopicNum\t"+stnTopicNum);
	// stnObj.setTopicsVct(stnTopicNum);
	// _SparseFeature[] fv = stnObj.getFv();
	//
	// for(int i=0; i<fv.length; i++){
	// int wid = fv[i].getIndex();
	// double val = fv[i].getValue();
	//
	// for(int k=0; k<stnTopicNum;k++){
	// System.out.println("k\t"+k);
	// System.out.println("wordIndex\t"+wid);
	// System.out.println("m_phi\t"+pDoc.m_phi[wid][k]);
	// stnObj.m_topics[k] += val*pDoc.m_phi[wid][k];
	// }
	// }
	//
	// Utils.L1Normalization(stnObj.m_topics);
	// }
	
	//to make it consistent, we will not assume the statistic collector has been normalized before calling this function
	@Override
	protected double calculate_log_likelihood(){
		double corpusLogLikelihood = 0;//how could we get the corpus-level likelihood?
		
		for(_Doc d: m_trainSet)
			corpusLogLikelihood += calculate_log_likelihood(d);
		
		return corpusLogLikelihood;
	}
	
	void discoverSpecificComments(MatchPair matchType, String similarityFile) {
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
							if (matchType == MatchPair.MP_ChildDoc)
								stnTopicSimilarity = computeSimilarity(stnObj.m_topics, cDoc.m_topics);
							else if (matchType == MatchPair.MP_ChildGlobal)
								stnTopicSimilarity = computeSimilarity(stnObj.m_topics, cDoc.m_xTopics[0]);
							else if (matchType == MatchPair.MP_ChildLocal)
								stnTopicSimilarity = computeSimilarity(stnObj.m_topics, cDoc.m_xTopics[1]);
							
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
				
				// writeFile(i, m_trainSet, m_testSet);
				System.out.println("Fold number "+i);
				System.out.println("Train Set Size "+m_trainSet.size());
				System.out.println("Test Set Size "+m_testSet.size());

				long start = System.currentTimeMillis();
				EM();
				perf[i] = Evaluation(i);
				
				System.out.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
				m_trainSet.clear();
				m_testSet.clear();			
			}
			
		}
		double mean = Utils.sumOfArray(perf)/k, var = 0;
		for(int i=0; i<perf.length; i++)
			var += (perf[i]-mean) * (perf[i]-mean);
		var = Math.sqrt(var/k);
		System.out.format("Perplexity %.3f+/-%.3f\n", mean, var);
	}
	
	public double Evaluation(int i) {
		m_collectCorpusStats = false;
		double perplexity = 0, loglikelihood, totalWords=0, sumLikelihood = 0;
		
		System.out.println("In Normal");
		
		for(_Doc d:m_testSet) {				
			loglikelihood = inference(d);
			sumLikelihood += loglikelihood;
			perplexity += loglikelihood;
			totalWords += d.getTotalDocLength();
			for(_ChildDoc cDoc: ((_ParentDoc)d).m_childDocs){
				totalWords += cDoc.getTotalDocLength();
			}
		}
		System.out.println("total Words\t"+totalWords+"perplexity\t"+perplexity);
		perplexity /= totalWords;
		perplexity = Math.exp(-perplexity);
		sumLikelihood /= m_testSet.size();

		System.out.format("Test set perplexity is %.3f and log-likelihood is %.3f\n", perplexity, sumLikelihood);
		
		return perplexity;		
	}
	
	@Override
	public double inference(_Doc pDoc){
		ArrayList<_Doc> sampleTestSet = new ArrayList<_Doc>();
		
		initTest(sampleTestSet, pDoc);
	
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
				double tempLogLikelihood = 0;
				for(_Doc doc: sampleTestSet){
					if(doc instanceof _ParentDoc){
						collectParentStats((_ParentDoc) doc);
						tempLogLikelihood += calculate_log_likelihood((_ParentDoc) doc);
					}
					else if(doc instanceof _ChildDoc){
						collectChildStats((_ChildDoc) doc);
						tempLogLikelihood += calculate_log_likelihood((_ChildDoc) doc);
					}
					
				}
				count ++;
				if(logLikelihood == 0)
					logLikelihood = tempLogLikelihood;
				else{

					logLikelihood = Utils.logSum(logLikelihood, tempLogLikelihood);
				}
			}
		} while (++iter<this.number_of_iteration);

		for(_Doc doc: sampleTestSet){
			estThetaInDoc(doc);
		}
		
		return logLikelihood - Math.log(count); 	
	}

	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d){
		_ParentDoc pDoc = (_ParentDoc)d;
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}
		pDoc.setTopics4Gibbs(number_of_topics, 0);		
		sampleTestSet.add(pDoc);
		
		for(_ChildDoc cDoc: pDoc.m_childDocs){
			((_ChildDoc) cDoc).createXSpace(number_of_topics, m_gamma.length);
			cDoc.setTopics4Gibbs(number_of_topics, 0);
			sampleTestSet.add(cDoc);
		}
	}

	// used to generate train and test data set 
	public void writeFile(int k, ArrayList<_Doc> trainSet,
			ArrayList<_Doc> testSet) {
		System.out.println("creating folder train test");
		String trainFilePrefix = "YahooTrainFolder" + k;
		String testFilePrefix = "YahooTestFolder" + k;

		File trainFolder = new File(trainFilePrefix);
		File testFolder = new File(testFilePrefix);
		if (!trainFolder.exists()) {
			System.out.println("creating root train directory" + trainFolder);
			trainFolder.mkdir();
		}
		if (!testFolder.exists()) {
			System.out.println("creating root test directory" + testFolder);
			testFolder.mkdir();
		}

		_Corpus trainCorpus = new _Corpus();
		_Corpus testCorpus = new _Corpus();

		for (_Doc d : trainSet)
			trainCorpus.addDoc(d);
		for (_Doc d : testSet)
			testCorpus.addDoc(d);
		outputFile.outputFiles(trainFilePrefix, trainCorpus);
		outputFile.outputFiles(testFilePrefix, testCorpus);
	}
//
	//used to print test parameter
	public void printTestParameter(String parentParameterFile, String childParameterFile){
		System.out.println("printing parameter");
		try{
			System.out.println(parentParameterFile);
			System.out.println(childParameterFile);
			
			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(childParameterFile));
			for(_Doc d: m_testSet){
				
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
				
				for(_ChildDoc cDoc: ((_ParentDoc)d).m_childDocs){
					childParaOut.print(cDoc.getName()+"\t");

					childParaOut.print("topicProportion\t");
					for (int k = 0; k < number_of_topics; k++) {
						childParaOut.print(cDoc.m_topics[k] + "\t");
					}
					
					childParaOut.print("general\t");
					for(int k=0; k<number_of_topics; k++){
						childParaOut.print(cDoc.m_xTopics[0][k]
								+ "\t");
					}

					childParaOut.print("specific\t");
					for (int k = 0; k < number_of_topics; k++) {
						childParaOut.print(cDoc.m_xTopics[1][k]
								+ "\t");
					}

					childParaOut.print("xProportion\t");
					for(int x=0; x<m_gamma.length; x++){
						childParaOut.print(cDoc.m_xProportion[x]+"\t");
					}
					
					childParaOut.println();
					
				}
			}
			
			parentParaOut.flush();
			parentParaOut.close();
			
			childParaOut.flush();
			childParaOut.close();
		}
		catch (Exception e) {
			e.printStackTrace();
//			e.printStackTrace();
//			System.err.print("para File Not Found");
		}
	}
	
	@Override
	public void printTopWords(int k, String betaFile) {
		double loglikelihood = 0.0;
		Arrays.fill(m_sstat, 0);

		System.out.println("print top words");
		for (_Doc d : m_trainSet) {
//			loglikelihood += calculate_log_likelihood(d);
			for (int i = 0; i < m_sstat.length; i++) {
				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
						: d.m_topics[i];	
				if (Double.isNaN(d.m_topics[i]))
					System.out.println("nan name\t" + d.getName());
			}
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

		System.out.format("Final Log Likelihood %.3f\t", loglikelihood);
		
		String filePrefix = betaFile.replace("topWords.txt", "");
		debugOutput(filePrefix);
		
	}
	
	public void debugOutput(String filePrefix){

		File parentTopicFolder = new File(filePrefix + "parentTopicAssignment");
		File childTopicFolder = new File(filePrefix + "childTopicAssignment");
		if (!parentTopicFolder.exists()) {
			System.out.println("creating directory" + parentTopicFolder);
			parentTopicFolder.mkdir();
		}
		if (!childTopicFolder.exists()) {
			System.out.println("creating directory" + childTopicFolder);
			childTopicFolder.mkdir();
		}
		
		File parentPhiFolder = new File(filePrefix + "parentPhi");
		File childPhiFolder = new File(filePrefix + "childPhi");
		if (!parentPhiFolder.exists()) {
			System.out.println("creating directory" + parentPhiFolder);
			parentPhiFolder.mkdir();
		}
		if (!childPhiFolder.exists()) {
			System.out.println("creating directory" + childPhiFolder);
			childPhiFolder.mkdir();
		}
		
		File childXFolder = new File(filePrefix+"xValue");
		if(!childXFolder.exists()){
			System.out.println("creating x Value directory" + childXFolder);
			childXFolder.mkdir();
		}

		for (_Doc d : m_corpus.getCollection()) {
		if (d instanceof _ParentDoc) {
				printTopicAssignment((_ParentDoc)d, parentTopicFolder);
				printParentPhi((_ParentDoc)d, parentPhiFolder);
			} else if (d instanceof _ChildDoc) {
				printTopicAssignment(d, childTopicFolder);
				printXValue(d, childXFolder);
			}

		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";
		printParameter(parentParameterFile, childParameterFile);

		String similarityFile = filePrefix+"topicSimilarity.txt";
		discoverSpecificComments(MatchPair.MP_ChildDoc, similarityFile);
		
		printEntropy(filePrefix);
		
		int topKStn = 10;
		int topKChild = 10;
		printTopKChild4Stn(filePrefix, topKChild);
		
		printTopKStn4Child(filePrefix, topKStn);
		
		printTopKChild4Parent(filePrefix, topKChild);
	}

	public void printTopicAssignment(_ParentDoc d, File parentFolder) {
	//	System.out.println("printing topic assignment parent documents");
		
		String topicAssignmentFile = d.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(parentFolder,
					topicAssignmentFile));
			
			for(_Stn stnObj: d.getSentences()){
				for(_Word w: stnObj.getWords()){
					int index = w.getIndex();
					int topic = w.getTopic();
					String featureName = m_corpus.getFeature(index);
					pw.print(featureName + ":" + topic + "\t");
				}
				pw.println();
			}
				
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}

	protected void printXValue(_Doc d, File childXFolder){
		String XValueFile = d.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(childXFolder,
					XValueFile));
	
			for(_Word w:d.getWords()){
				int index = w.getIndex();
				int x = w.getX();
				double xProb = w.getXProb();
				String featureName = m_corpus.getFeature(index);
				pw.print(featureName + ":" + x + ":" + xProb + "\t");
			}
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public void printParameter(String parentParameterFile, String childParameterFile){
		System.out.println("printing parameter");
		try{
			System.out.println(parentParameterFile);
			System.out.println(childParameterFile);
			
			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(childParameterFile));
			for(_Doc d: m_corpus.getCollection()){
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
					
				}else{
					if(d instanceof _ChildDoc){
						childParaOut.print(d.getName()+"\t");

						childParaOut.print("topicProportion\t");
						for (int k = 0; k < number_of_topics; k++) {
							childParaOut.print(d.m_topics[k] + "\t");
						}
						
						childParaOut.print("general\t");
						for(int k=0; k<number_of_topics; k++){
							childParaOut.print(((_ChildDoc) d).m_xTopics[0][k]
									+ "\t");
						}

						childParaOut.print("specific\t");
						for (int k = 0; k < number_of_topics; k++) {
							childParaOut.print(((_ChildDoc) d).m_xTopics[1][k]
									+ "\t");
						}

						childParaOut.print("xProportion\t");
						for(int x=0; x<m_gamma.length; x++){
							childParaOut.print(((_ChildDoc)d).m_xProportion[x]+"\t");
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

	public void printParentPhi(_ParentDoc d, File phiFolder){
		String parentPhiFileName = d.getName()+".txt";
		_SparseFeature[] fv = d.getSparse();
		
		try{
			PrintWriter parentPW = new PrintWriter(new File(phiFolder, parentPhiFileName));
		
			for(int n=0; n<fv.length; n++){
				int index = fv[n].getIndex();
				String featureName = m_corpus.getFeature(index);
				parentPW.print(featureName + ":\t");
				for(int k=0; k<number_of_topics; k++)
					parentPW.print(d.m_phi[n][k]+"\t");
				parentPW.println();
			}
			parentPW.flush();
			parentPW.close();
		}catch(Exception ex){
			ex.printStackTrace();
		}
	}
	
	protected void printEntropy(String filePrefix){
		String entropyFile = filePrefix+"entropy.txt";
		boolean logScale = true;
		
		try{
			PrintWriter entropyPW = new PrintWriter(new File(entropyFile));
			
			for(_Doc d: m_corpus.getCollection()){
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
	
	protected void printTopKChild4Stn(String filePrefix, int topK){
		String topKChild4StnFile = filePrefix+"topChild4Stn.txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4StnFile));
			
			for(_Doc d: m_corpus.getCollection()){
				if(d instanceof _ParentDoc){
					_ParentDoc pDoc = (_ParentDoc)d;
					
					pw.println(pDoc.getName()+"\t"+pDoc.getSenetenceSize());
					
					for(_Stn stnObj:pDoc.getSentences()){
						HashMap<String, Double> likelihoodMap = rankChild4StnByLikelihood(stnObj, pDoc);
						
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
			
			for(_Doc d: m_corpus.getCollection()){
				if(d instanceof _ParentDoc){
					_ParentDoc pDoc = (_ParentDoc)d;
					
					pw.println(pDoc.getName()+"\t"+pDoc.m_childDocs.size());
					
					for(_ChildDoc childDoc: pDoc.m_childDocs){
//						_ChildDoc4ThreePhi cDoc = (_ChildDoc4ThreePhi)childDoc;
						HashMap<Integer, Double>stnSimMap = rankStn4ChildBySim(pDoc, childDoc);
						int i = 0;
						
						pw.print(childDoc.getName()+"\t");
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
	
	protected void printTopKChild4Parent(String filePrefix, int topK) {
		String topKChild4StnFile = filePrefix+"topChild4Parent.txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4StnFile));
			
			for(_Doc d: m_corpus.getCollection()){
				if(d instanceof _ParentDoc){
					_ParentDoc pDoc = (_ParentDoc)d;
					
					pw.print(pDoc.getName()+"\t");
					
					for(_ChildDoc childDoc:pDoc.m_childDocs){
//						_ChildDoc4ThreePhi cDoc = (_ChildDoc4ThreePhi) childDoc;
						double docLogLikelihood = rankChild4ParentByLikelihood(childDoc, pDoc);
				
						pw.print(childDoc.getName()+":"+docLogLikelihood+"\t");	
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
	
	@Override
	public double calculate_log_likelihood(_Doc d){
		if (d instanceof _ParentDoc)
			return logLikelihoodByIntegrateTopics((_ParentDoc)d);
		else
			return logLikelihoodByIntegrateTopics((_ChildDoc)d);
	}
	
	protected double logLikelihoodByIntegrateTopics(_ParentDoc d) {
		double docLogLikelihood = 0.0;
		_SparseFeature[] fv = d.getSparse();

		for (int j = 0; j < fv.length; j++) {
			int index = fv[j].getIndex();
			double value = fv[j].getValue();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = parentWordByTopicProb(k, index)*parentTopicInDocProb(k, d)/(d.getTotalDocLength()+number_of_topics*d_alpha);
				wordLogLikelihood += wordPerTopicLikelihood;
//				double wordPerTopicLikelihood = Math.log(parentWordByTopicProb(k, index))+Math.log(parentTopicInDocProb(k, d)/(d.getTotalDocLength()+number_of_topics*d_alpha));
				
//				if(wordLogLikelihood == 0)
//					wordLogLikelihood = wordPerTopicLikelihood;
//				else
//					wordLogLikelihood = Utils.logSum(wordLogLikelihood, wordPerTopicLikelihood);
			}
			
			if(Math.abs(wordLogLikelihood) < 1e-10){
				System.out.println("wordLoglikelihood\t"+wordLogLikelihood);
				wordLogLikelihood += 1e-10;
			}
			
			wordLogLikelihood = Math.log(wordLogLikelihood);

			docLogLikelihood += value * wordLogLikelihood;
		}

		return docLogLikelihood;
	}
	
	protected double getChildTheta(int k, _ChildDoc d){
		double childTheta = 0;
		for(int x=0; x<m_gamma.length; x++){
			childTheta += childTopicInDocProb(k, x, d)*childXInDocProb(x, d);
		}
		childTheta /= (d.getTotalDocLength()+Utils.sumOfArray(m_gamma));
		
		return childTheta;
	}
	
	protected double logLikelihoodByIntegrateTopics(_ChildDoc d) {
		double docLogLikelihood = 0.0;

		// prepare compute the normalizers
		_SparseFeature[] fv = d.getSparse();
		
		for (int i=0; i<fv.length; i++) {
			int wid = fv[i].getIndex();
			double value = fv[i].getValue();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = Math.log(childWordByTopicProb(k, wid))+Math.log(getChildTheta(k, d));
		
				if(wordLogLikelihood == 0)
					wordLogLikelihood = wordPerTopicLikelihood;
				else
					wordLogLikelihood = Utils.logSum(wordLogLikelihood, wordPerTopicLikelihood);
			}
			
			if(Math.abs(wordLogLikelihood) < 1e-10){
				System.out.println("wordLoglikelihood\t"+wordLogLikelihood);
				wordLogLikelihood += 1e-10;
			}
	
			docLogLikelihood += value * wordLogLikelihood;
		}
		
		return docLogLikelihood;
	}
	
	// p(w, z)=p(w|z)p(z) multinomial-dirichlet
	protected void logLikelihoodByIntegrateThetaPhi(int iter) {
		double logLikelihood = 0.0;
		double parentLogLikelihood = 0.0;
		double childLogLikelihood = 0.0;

		for (_Doc d : m_trainSet) {
			if (d instanceof _ParentDoc)
				parentLogLikelihood += parentLogLikelihoodByIntegrateThetaPhi((_ParentDoc) d);
			else if (d instanceof _ChildDoc)
				childLogLikelihood += childLogLikelihoodByIntegrateThetaPhi((_ChildDoc) d);
		}

		double term1 = 0.0;
		double term2 = 0.0;
		double term3 = 0.0;
		double term4 = 0.0;
		for (int k = 0; k < number_of_topics; k++) {
			for (int n = 0; n < vocabulary_size; n++) {
				term3 += Utils.lgamma(d_beta + word_topic_sstat[k][n]);
			}
			term4 -= Utils.lgamma(vocabulary_size * d_beta + m_sstat[k]);
		}

		term1 = number_of_topics * Utils.lgamma(vocabulary_size * d_beta);
		term2 = -number_of_topics * (vocabulary_size * Utils.lgamma(d_beta));

		parentLogLikelihood += term1 + term2 + term3 + term4;

		term1 = 0.0;
		term2 = 0.0;
		term3 = 0.0;
		term4 = 0.0;
		for (int k = 0; k < number_of_topics; k++) {
			for (int n = 0; n < vocabulary_size; n++) {
				term3 += Utils.lgamma(d_beta + word_topic_sstat[k][n]);
			}
			term4 -= Utils.lgamma(vocabulary_size * d_beta + m_sstat[k]);
		}

		term1 = number_of_topics * Utils.lgamma(vocabulary_size * d_beta);
		term2 = -number_of_topics * (vocabulary_size * Utils.lgamma(d_beta));

		childLogLikelihood += term1 + term2 + term3 + term4;

		System.out.format("iter %d, parent log likelihood %.3f\n", iter, parentLogLikelihood);
		System.out.format("iter %d, child log likelihood %.3f\n", iter, childLogLikelihood);
		logLikelihood = parentLogLikelihood + childLogLikelihood;

		System.out.format("iter %d, log likelihood %.3f\n", iter, logLikelihood);
	}

	// log space
	protected double parentLogLikelihoodByIntegrateThetaPhi(_ParentDoc pDoc) {
		double term1 = 0.0;
		double term2 = 0.0;

		for (int k = 0; k < number_of_topics; k++) {
			term2 += Utils.lgamma(pDoc.m_sstat[k] + d_alpha);
		}
		term2 -= Utils.lgamma((double) (number_of_topics * d_alpha + pDoc.getDocLength()));

		term1 = Utils.lgamma(number_of_topics * d_alpha) - number_of_topics * Utils.lgamma(d_alpha);

		return term1 + term2;
	}

	// sum_x p(z|x)p(x)
	protected double childLogLikelihoodByIntegrateThetaPhi(_ChildDoc cDoc) {
		double tempLogLikelihood = 0.0;
		double tempLogLikelihood1 = 0.0;
		double tempLogLikelihood2 = 0.0;
		double term11 = 0.0;
		double term12 = 0.0;
		double term13 = 0.0;
		double term14 = 0.0;
		double weight1 = 0.0;
		double weight2 = 0.0;

		double term21 = 0.0;

		for (int k = 0; k < number_of_topics; k++) {
			term12 -= Utils.lgamma(d_alpha + cDoc.m_parentDoc.m_sstat[k]);
			term13 += Utils.lgamma(d_alpha + cDoc.m_parentDoc.m_sstat[k] + cDoc.m_xTopicSstat[0][k]);

			term21 += Utils.lgamma(d_alpha + cDoc.m_xTopicSstat[1][k]);
		}
		term11 = Utils.lgamma(number_of_topics * d_alpha + cDoc.m_parentDoc.getTotalDocLength());
		term14 = -(Utils.lgamma(number_of_topics * d_alpha + cDoc.m_parentDoc.getTotalDocLength() + cDoc.m_xSstat[0]));

		tempLogLikelihood1 = term11 + term12 + term13 + term14;

		tempLogLikelihood2 = Utils.lgamma(number_of_topics * d_alpha) - number_of_topics * Utils.lgamma(d_alpha)
				+ term21 - Utils.lgamma(number_of_topics * d_alpha + cDoc.m_xSstat[1]);

		weight1 = Utils.lgamma(m_gamma[0] + m_gamma[1]) - Utils.lgamma(m_gamma[0]) - Utils.lgamma(m_gamma[1])
				+ Utils.lgamma(m_gamma[0] + cDoc.m_xSstat[0]) + Utils.lgamma(m_gamma[1])
				- Utils.lgamma(m_gamma[0] + m_gamma[1] + cDoc.m_xSstat[0]);

		weight2 = Utils.lgamma(m_gamma[0] + m_gamma[1]) - Utils.lgamma(m_gamma[0]) - Utils.lgamma(m_gamma[1])
				+ Utils.lgamma(m_gamma[0]) + Utils.lgamma(m_gamma[1] + cDoc.m_xSstat[1])
				- Utils.lgamma(m_gamma[0] + m_gamma[1] + cDoc.m_xSstat[1]);

		// tempLogLikelihood = tempLogLikelihood1 * cDoc.m_xProportion[0]
		// + tempLogLikelihood2 * cDoc.m_xProportion[1];

		tempLogLikelihood = tempLogLikelihood1 + weight1 + tempLogLikelihood2 + weight2;

		return tempLogLikelihood;
	}

}
