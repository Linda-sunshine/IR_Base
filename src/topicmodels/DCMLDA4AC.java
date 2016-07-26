package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.PrimitiveIterator.OfDouble;

import structures.MyPriorityQueue;
import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4DCM;
import structures._RankItem;
import structures._Stn;
import structures._Word;
import utils.Utils;

/**
 * 
 * per article and its comments sharing a topicWordDistribution
 * 
 * */

public class DCMLDA4AC extends LDA_Gibbs_Debug{
	protected double[] m_alpha;
	protected double[][] m_beta;
	
	protected double m_totalAlpha;
	protected double[] m_totalBeta;
	
	protected int m_newtonIter;
	protected double m_newtonConverge;
	
	public DCMLDA4AC(int number_of_iteration, double converge, double beta, _Corpus c, double lambda, int number_of_topics, double alpha, double  burnIn, int lag, double ksi, double tau, int newtonIter, double newtonConverge){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, ksi, tau);
	
		m_alpha = new double[number_of_topics];
		m_beta = new double[number_of_topics][vocabulary_size];
		
		m_totalAlpha = 0;
		m_totalBeta = new double[number_of_topics];
		
		m_newtonIter = newtonIter;
		m_newtonConverge = newtonConverge;
	}
	
	public String toString(){
		return String.format("DCMLDA4AC[k:%d, alphaA:%.2f, beta:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta);
	}
	
	public void EM(){
		System.out.format("Starting %s ... \n", toString());
		
		long startTime = System.currentTimeMillis();
		
		m_collectCorpusStats = true;
		initialize_probability(m_trainSet);
		
		String filePrefix = "./data/results/DCM_LDA";
		File weightFolder = new File(filePrefix+"");
		if(!weightFolder.exists()){
			weightFolder.mkdir();
		}
		
		double delta = 0, last = 0, current = 0;
		
		int i=0, displayCount = 0;
		do{
			long eStartTime = System.currentTimeMillis();
			for(int j=0; j<number_of_iteration; j++){
				init();
				for(_Doc d:m_trainSet){
					calculate_E_step(d);
				}
			}
			
			long eEndTime = System.currentTimeMillis();
			
			System.out.println("per iteration e step time\t"+(eEndTime-eStartTime));
			
			long mStartTime = System.currentTimeMillis();
			calculate_M_step(i, weightFolder);
			long mEndTime = System.currentTimeMillis();
			
			System.out.println("per iteration m step time\t"+(mEndTime-mStartTime));
			
			if (m_converge > 0
					|| (m_displayLap > 0 && i % m_displayLap == 0 && displayCount > 6)) {
				// required to display log-likelihood
				current = calculate_log_likelihood();
				// together with corpus-level log-likelihood

				if (i > 0)
					delta = (last - current) / last;
				else
					delta = 1.0;
				last = current;
			}

			if (m_displayLap > 0 && i % m_displayLap == 0) {
				if (m_converge > 0) {
					System.out.format(
							"Likelihood %.3f at step %s converge to %f...\n",
							current, i, delta);
					infoWriter.format(
							"Likelihood %.3f at step %s converge to %f...\n",
							current, i, delta);

				} else {
					System.out.print(".");
					if (displayCount > 6) {
						System.out.format("\t%d:%.3f\n", i, current);
						infoWriter.format("\t%d:%.3f\n", i, current);
					}
					displayCount++;
				}
			}

			if (m_converge > 0 && Math.abs(delta) < m_converge)
				break;// to speed-up, we don't need to compute likelihood in
						// many cases
			
		}while(++i<number_of_iteration);
		
		finalEst();
		
		long endTime = System.currentTimeMillis() - startTime;
		
		System.out.format("likelihood %.3f after step %s converge to %f after %d seconds ...\n", current, i, delta, endTime/1000);
	}
	
	protected void initialize_probability(Collection<_Doc>collection){
		for(_Doc d:collection){
			if(d instanceof _ParentDoc4DCM){
				_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
				pDoc.setTopics4Gibbs(number_of_topics, 0, vocabulary_size);
				
				for(_ChildDoc cDoc:pDoc.m_childDocs){
					cDoc.setTopics4Gibbs_LDA(number_of_topics, 0);
					
					for(_Word w:cDoc.getWords()){
						int wid = w.getIndex();
						int tid = w.getTopic();
						
						pDoc.m_wordTopic_stat[tid][wid] ++;
						pDoc.m_topic_stat[tid] ++;
					}
					computeMu4Doc(cDoc);
				}
			}
			
		}
		
		initialAlphaBeta();
		imposePrior();
	}
	
	protected void computeMu4Doc(_ChildDoc d){
		_ParentDoc tempParent = d.m_parentDoc;
		double mu = Utils.cosine(tempParent.getSparse(), d.getSparse());
		mu = 0.5;
		d.setMu(mu);
	}
	
	protected void computeTestMu4Doc(_ChildDoc d){
		_ParentDoc pDoc = d.m_parentDoc;
		
		double mu = Utils.cosine(d.getSparseVct4Infer(), pDoc.getSparseVct4Infer());
		mu = 0.05;
		d.setMu(mu);
	}
	
	public double calculate_E_step(_Doc d){
		d.permutation();
		
		if(d instanceof _ParentDoc)
			sampleInParentDoc((_ParentDoc)d);
		else if(d instanceof _ChildDoc)
			sampleInChildDoc((_ChildDoc)d);
		
		return 0;
	}
	
	protected void sampleInParentDoc(_Doc d){
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d;
		int wid, tid;
		double normalizedProb;
		
		for (_Word w : pDoc.getWords()) {
			tid = w.getTopic();
			wid = w.getIndex();
			
			pDoc.m_sstat[tid]--;
			pDoc.m_topic_stat[tid]--;
			pDoc.m_wordTopic_stat[tid][wid]--;
			
			normalizedProb = 0;
			for(tid=0; tid<number_of_topics; tid++){
				double pWordTopic = wordTopicProb(tid, wid, pDoc);
				double pTopicPDoc = topicInDocProb(tid, pDoc);
				
				m_topicProbCache[tid] = pWordTopic*pTopicPDoc;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<number_of_topics; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<=0)
					break;
			}
			
			if(tid==number_of_topics)
				tid --;
			
			w.setTopic(tid);
			pDoc.m_sstat[tid]++;
			pDoc.m_topic_stat[tid]++;
			pDoc.m_wordTopic_stat[tid][wid]++;
		}
		
	}
	
	protected void sampleInChildDoc(_ChildDoc d){
		int wid, tid;
		double normalizedProb;
		
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d.m_parentDoc;

		for(_Word w:d.getWords()){
			tid = w.getTopic();
			wid = w.getIndex();
			
			pDoc.m_wordTopic_stat[tid][wid]--;
			pDoc.m_topic_stat[tid] --;
			d.m_sstat[tid] --;

			normalizedProb = 0;
			for (tid = 0; tid < number_of_topics; tid++) {
				double pWordTopic = wordTopicProb(tid, wid, pDoc);
				double pTopic = topicInDocProb(tid, d);
				
				m_topicProbCache[tid] = pWordTopic * pTopic;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for (tid = 0; tid < m_topicProbCache.length; tid++) {
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<=0)
					break;
			}
			
			if(tid==m_topicProbCache.length)
				tid--;
			
			w.setTopic(tid);
			d.m_sstat[tid]++;
			pDoc.m_topic_stat[tid]++;
			pDoc.m_wordTopic_stat[tid][wid]++;
		}
	}
	
	protected double topicInDocProb(int tid, _Doc d){
		double term1 = d.m_sstat[tid];
		
		return (d.m_sstat[tid]+m_alpha[tid])/(d.getDocInferLength()+m_totalAlpha-1);
	}
	
	protected double wordTopicProb(int tid, int wid, _ParentDoc4DCM d){
		double term1 = d.m_wordTopic_stat[tid][wid];
		
		return (term1+m_beta[tid][wid])/(d.m_sstat[tid]+m_totalBeta[tid]);
	}
	
	public void calculate_M_step(int iter, File weightFolder) {

		for (_Doc d : m_trainSet){
			if(d instanceof _ParentDoc4DCM)
				collectParentStats((_ParentDoc4DCM)d);
			else
				collectChildStats((_ChildDoc)d);
		}
			
		File weightIterFolder = new File(weightFolder, "_" + iter);
		if (!weightIterFolder.exists()) {
			weightIterFolder.mkdir();
		}

		updateParameter(iter, weightIterFolder);

	}
	
	protected void updateParameter(int iter, File weightIterFolder) {
		initialAlphaBeta();
		updateAlpha();

		for (int k = 0; k < number_of_topics; k++)
			updateBeta(k);

		for (int k = 0; k < number_of_topics; k++)
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);

		String fileName = iter + ".txt";
		saveParameter2File(weightIterFolder, fileName);

	}
	
	protected void collectParentStats(_ParentDoc4DCM d) {

		for (int k = 0; k < this.number_of_topics; k++) {
			d.m_topics[k] += d.m_sstat[k] + m_alpha[k];

			for (int v = 0; v < vocabulary_size; v++){
				d.m_wordTopic_prob[k][v] += d.m_wordTopic_stat[k][v] + m_beta[k][v];
			}
		}

	}
	
	protected void collectChildStats(_ChildDoc d){
		for(int k=0; k<this.number_of_topics; k++){
			d.m_topics[k] += d.m_sstat[k]+m_alpha[k];
		}
	}

	protected void initialAlphaBeta() {

		Arrays.fill(m_sstat, 0);
		for (int k = 0; k < number_of_topics; k++) {
			Arrays.fill(topic_term_probabilty[k], 0);
			Arrays.fill(word_topic_sstat[k], 0);
		}

		for (_Doc d : m_trainSet) {
			if(d instanceof _ParentDoc4DCM){
				_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
				for (int k = 0; k < number_of_topics; k++) {
					double tempProb = pDoc.m_sstat[k] / pDoc.getTotalDocLength();
					m_sstat[k] += tempProb;
					if (pDoc.m_sstat[k] == 0)
						continue;
					for (int v = 0; v < vocabulary_size; v++) {
						tempProb = pDoc.m_wordTopic_prob[k][v]
								/pDoc.m_sstat[k];
	
						topic_term_probabilty[k][v] += tempProb;
					}
				}
				
				for(_ChildDoc cDoc:pDoc.m_childDocs){
					for(int k=0; k<number_of_topics; k++){
						double tempProb = cDoc.m_sstat[k]/cDoc.getTotalDocLength();
						m_sstat[k] += tempProb;
						
					}
				}
			}
		}

		int trainSetSize = m_trainSet.size();
		for (int k = 0; k < number_of_topics; k++) {
			m_sstat[k] /= trainSetSize;
			for (int v = 0; v < vocabulary_size; v++) {
				topic_term_probabilty[k][v] /= trainSetSize;
			}
		}
	
		 for (int k = 0; k < number_of_topics; k++){
			m_alpha[k] = m_sstat[k];
			for (int v = 0; v < vocabulary_size; v++) {
				m_beta[k][v] = topic_term_probabilty[k][v] + d_beta;
			}
		}
		
		m_totalAlpha = Utils.sumOfArray(m_alpha);
		for (int k = 0; k < number_of_topics; k++) {
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);
		}

	}

	protected void updateAlpha() {
		double diff = 0;
		double smallAlpha = 0.1;
		int iteration = 0;
		do {

			diff = 0;

			double totalAlphaDenominator = 0;
			m_totalAlpha = Utils.sumOfArray(m_alpha);
			double digAlpha = Utils.digamma(m_totalAlpha);

			double deltaAlpha = 0;

			for (_Doc d : m_trainSet) {
				totalAlphaDenominator += Utils.digamma(d.getTotalDocLength()
						+ m_totalAlpha)
						- digAlpha;
			}
			
			for (int k = 0; k < number_of_topics; k++) {
				double totalAlphaNumerator = 0;
				for (_Doc d : m_trainSet) {
					totalAlphaNumerator += Utils.digamma(m_alpha[k]
							+ d.m_sstat[k])
							- Utils.digamma(m_alpha[k]);
				}

				deltaAlpha = totalAlphaNumerator * 1.0
						/ totalAlphaDenominator;


				double newAlpha = m_alpha[k] * deltaAlpha;
				double t_diff = Math.abs(m_alpha[k] - newAlpha);
				if (t_diff > diff)
					diff = t_diff;

				m_alpha[k] = newAlpha;
			}
			
			iteration++;
			System.out.println("alpha iteration\t" + iteration);
	
			if(iteration > m_newtonIter)
				break;
			
		}while(diff>m_newtonConverge);

		System.out.println("iteration\t" + iteration);
		m_totalAlpha = 0;
		for (int k = 0; k < number_of_topics; k++) {
			m_totalAlpha += m_alpha[k];
		}
	}

	protected void updateBeta(int tid) {

		double diff = 0;
		int iteration = 0;
		double smoothingBeta = 0.1;

		do {
			diff = 0;
			double deltaBeta = 0;
			double wordNum4Tid = 0;
			double[] wordNum4Tid4V = new double[vocabulary_size];
			double totalBetaDenominator = 0;
			double[] totalBetaNumerator = new double[vocabulary_size];
			Arrays.fill(totalBetaNumerator, 0);
			Arrays.fill(wordNum4Tid4V, 0);
			m_totalBeta[tid] = Utils.sumOfArray(m_beta[tid]);
			double digBeta4Tid = Utils.digamma(m_totalBeta[tid]);

			for (_Doc d : m_trainSet) {
				if(d instanceof _ParentDoc4DCM){
					_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
					totalBetaDenominator += Utils.digamma(m_totalBeta[tid]
							+ pDoc.m_topic_stat[tid])
							- digBeta4Tid;
					for (int v = 0; v < vocabulary_size; v++) {
						wordNum4Tid += pDoc.m_wordTopic_stat[tid][v];
						wordNum4Tid4V[v] += pDoc.m_wordTopic_stat[tid][v];
						totalBetaNumerator[v] += Utils.digamma(m_beta[tid][v]
								+ pDoc.m_wordTopic_stat[tid][v]);
						totalBetaNumerator[v] -= Utils.digamma(m_beta[tid][v]);
					}
				}
			}

			for (int v = 0; v < vocabulary_size; v++) {
				if (wordNum4Tid == 0)
					break;
				if (wordNum4Tid4V[v] == 0) {
					deltaBeta = 0;

				} else {
					deltaBeta = totalBetaNumerator[v] / totalBetaDenominator;

				}
				
				double newBeta = m_beta[tid][v] * deltaBeta + d_beta;

				double t_diff = Math.abs(m_beta[tid][v] - newBeta);
				if (t_diff > diff)
					diff = t_diff;

				m_beta[tid][v] = newBeta;

			}

			iteration++;

			System.out.println("beta iteration\t" + iteration);
		} while (diff > m_newtonConverge);

		System.out.println("iteration\t" + iteration);

	}

	protected void saveParameter2File(File fileFolder, String fileName) {
		try {
			File paramFile = new File(fileFolder, fileName);

			PrintWriter pw = new PrintWriter(paramFile);
			pw.println("alpha");
			for (int k = 0; k < number_of_topics; k++) {
				pw.print(m_alpha[k] + "\t");
			}

			pw.println("beta");
			for (int k = 0; k < number_of_topics; k++) {
				pw.print("topic" + k + "\t");
				for (int v = 0; v < vocabulary_size; v++) {
					pw.print(m_beta[k][v] + "\t");
				}
				pw.println();
			}

			pw.flush();
			pw.close();
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}
	
	protected double calculate_log_likelihood() {
		double logLikelihood = 0.0;
		for (_Doc d : m_trainSet) {
			if(d instanceof _ParentDoc4DCM)
				logLikelihood += calculate_log_likelihood((_ParentDoc4DCM)d);
		}
		return logLikelihood;
	}

	protected double calculate_log_likelihood(_ParentDoc4DCM d) {
	
		double docLogLikelihood = 0.0;

		double parentDocLength = d.getTotalDocLength();
		
		for (int k = 0; k < number_of_topics; k++) {
			double term = Utils.lgamma(d.m_sstat[k] + m_alpha[k]);
			docLogLikelihood += term;

			term = Utils.lgamma(m_alpha[k]);
			docLogLikelihood -= term;

		}

		docLogLikelihood += Utils.lgamma(m_totalAlpha);
		docLogLikelihood -= Utils.lgamma(parentDocLength + m_totalAlpha);

		for (int k = 0; k < number_of_topics; k++) {
			for (int v = 0; v < vocabulary_size; v++) {
				double term = Utils.lgamma(d.m_wordTopic_stat[k][v]
						+ m_beta[k][v]);
				docLogLikelihood += term;

				term = Utils.lgamma(m_beta[k][v]);
				docLogLikelihood -= term;

			}
			docLogLikelihood += Utils.lgamma(m_totalBeta[k]);
			docLogLikelihood -= Utils.lgamma(d.m_topic_stat[k] + m_totalBeta[k]);
		}

		for(_ChildDoc cDoc:d.m_childDocs){
			int cDocLength = cDoc.getTotalDocLength();
			for(int k=0; k<number_of_topics; k++){
				double term = Utils.lgamma(cDoc.m_sstat[k]+m_alpha[k]);
				docLogLikelihood += term;
				
				term = Utils.lgamma(m_alpha[k]);
				docLogLikelihood -= term;
			}
			
			docLogLikelihood += Utils.lgamma(m_totalAlpha);
			docLogLikelihood -= Utils.lgamma(cDocLength+m_totalAlpha);
			
		}
		
		return docLogLikelihood;
	}

	protected void estThetaInDoc(_Doc d) {
		if(d instanceof _ParentDoc4DCM){
			_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d;
			for (int i = 0; i < number_of_topics; i++)
				Utils.L1Normalization(pDoc.m_wordTopic_prob[i]);
		}
		Utils.L1Normalization(d.m_topics);
	}

	public void printTopWords(int k, String betaFile) {
		double logLikelihood = calculate_log_likelihood();
		System.out.format("final log likelihood %.3f\t", logLikelihood);
		
		String filePrefix = betaFile.replace("topWords.txt", "");
		debugOutput(filePrefix);

		Arrays.fill(m_sstat, 0);

		System.out.println("print top words");
		printTopWords_liteVersion(k, betaFile);
	}
	
	public void debugOutput(String filePrefix) {
		
		int topK = 10;
		File parentTopicFolder = new File(filePrefix + "parentTopicAssignment");
		File childTopicFolder = new File(filePrefix + "childTopicAssignment");
		
		if(!parentTopicFolder.exists()){
			System.out.println("creating directory\t"+parentTopicFolder);
			parentTopicFolder.mkdir();
		}
		
		if(!childTopicFolder.exists()){
			System.out.println("creating directory\t"+childTopicFolder);
			childTopicFolder.mkdir();
		}
	
		File parentWordTopicDistributionFolder = new File(filePrefix
				+ "wordTopicDistribution");
		if (!parentWordTopicDistributionFolder.exists()) {
			System.out.println("creating word topic distribution folder\t"
					+ parentWordTopicDistributionFolder);
			parentWordTopicDistributionFolder.mkdir();
		}

		for (_Doc d : m_trainSet) {
			if(d instanceof _ParentDoc4DCM){
				printParentTopicAssignment(d, parentTopicFolder);
				printWordTopicDistribution(d, parentWordTopicDistributionFolder, topK);
			}else{
				printChildTopicAssignment(d, childTopicFolder);
			}
		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";

		printParameter(parentParameterFile, childParameterFile, m_trainSet);


	}

	public void printTopWords_liteVersion(int k, String topWordPath) {
		System.out.println("TopWord FilePath:" + topWordPath);

		Arrays.fill(m_sstat, 0);
		for (_Doc d : m_trainSet) {
			for (int i = 0; i < number_of_topics; i++)
				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
						: d.m_topics[i];
		}
		Utils.L1Normalization(m_sstat);

		try {
			PrintWriter topWordWriter = new PrintWriter(new File(topWordPath));

			for (int i = 0; i < m_beta.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j),
							m_beta[i][j]));

				topWordWriter.format("Topic %d(%.5f):\t", i, m_sstat[i]);
				for (_RankItem it : fVector)
					topWordWriter.format("%s(%.5f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
				topWordWriter.write("\n");
			}
			topWordWriter.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}

	protected void printParentTopicAssignment(_Doc d, File topicFolder) {
		String topicAssignmentFile = d.getName() + ".txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topicFolder,
					topicAssignmentFile));
			
			for (_Word w : d.getWords()) {
				int index = w.getIndex();
				int topic = w.getTopic();

				String featureName = m_corpus.getFeature(index);
				pw.print(featureName + ":" + topic + "\t");
			}
			
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	protected void printParameter(String parentParameterFile,
			String childParameterFile, ArrayList<_Doc> docList) {
		System.out.println("printing parameter");
		
		try{
			System.out.println(parentParameterFile);
			System.out.println(childParameterFile);
			
			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(childParameterFile));
			
			for(_Doc d:docList){
				parentParaOut.print(d.getName()+"\t");
				parentParaOut.print("topicProportion\t");
				for(int k=0; k<number_of_topics; k++){
					parentParaOut.print(d.m_topics[k]+"\t");
				}
				
				parentParaOut.println();
			}
			
			parentParaOut.flush();
			parentParaOut.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	protected void printWordTopicDistribution(_Doc d,
			File wordTopicDistributionFolder, int k) {
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
		
		String wordTopicDistributionFile = pDoc.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(
					wordTopicDistributionFolder, wordTopicDistributionFile));

			for (int i = 0; i < number_of_topics; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int v = 0; v < vocabulary_size; v++) {
					String featureName = m_corpus.getFeature(v);
					double wordProb = pDoc.m_wordTopic_prob[i][v];
					_RankItem ri = new _RankItem(featureName, wordProb);
					fVector.add(ri);
				}

				pw.format("Topic %d(%.5f):\t", i, d.m_topics[i]);
				for (_RankItem it : fVector)
					pw.format("%s(%.5f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
				pw.write("\n");
			}

			pw.flush();
			pw.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public void initTestDoc(ArrayList<_Doc> sampleTestSet, _Doc d) {
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
		
		for(_Stn stnObj:pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}
		
		int testLength = 0;
		pDoc.setTopics4GibbsTest(number_of_topics, 0, testLength, vocabulary_size);
		
		sampleTestSet.add(pDoc);
		
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			testLength = (int)(m_testWord4PerplexityProportion*cDoc.getTotalDocLength());
			cDoc.setTopics4GibbsTest(number_of_topics, d_alpha, testLength);
			
			for (_Word w : d.getWords()) {
				int wid = w.getIndex();
				int tid = w.getTopic();
				pDoc.m_wordTopic_stat[tid][wid]++;
				pDoc.m_topic_stat[tid] ++;
			}
			sampleTestSet.add(cDoc);
			cDoc.createSparseVct4Infer();
			computeTestMu4Doc(cDoc);
		}
	}

	protected double calculate_test_log_likelihood(_ParentDoc4DCM d) {
		double likelihood = 0;
		
		for (_Word w : d.getWords()) {
			int wid = w.getIndex();
			double wordLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				wordLikelihood += d.m_topics[k]
						* d.m_wordTopic_prob[k][wid];
			}
			likelihood += Math.log(wordLikelihood);
		}
		
		return likelihood;
	}
	
	protected double calculate_test_log_likelihood(_ChildDoc d) {
		double likelihood = 0;
		
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d.m_parentDoc;
		
		for (_Word w : d.getWords()) {
			int wid = w.getIndex();
			double wordLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				wordLikelihood += d.m_topics[k]
						* pDoc.m_wordTopic_prob[k][wid];
			}
			likelihood += Math.log(wordLikelihood);
		}
		
		return likelihood;
	}

}
