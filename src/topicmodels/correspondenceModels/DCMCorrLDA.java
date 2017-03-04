package topicmodels.correspondenceModels;

/*******
comments of an article has its own topic proportion,
**********/


import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.netlib.util.doubleW;

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

public class DCMCorrLDA extends DCMLDA4AC {
	
	protected double[] m_alpha_c;
	protected double m_totalAlpha_c;
	protected double[] m_alphaAuxilary;
	protected double d_alpha_c;
	
	public DCMCorrLDA(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha_a,
			double alpha_c, double burnIn, double ksi, double tau, int lag,
			int newtonIter, double newtonConverge) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha_a, burnIn, lag, ksi, tau, newtonIter, newtonConverge);
		
		d_alpha_c = alpha_c;

	}
	
	public String toString(){
		return String
				.format("DCMCorrLDA[k:%d, alphaA:%.2f, beta:%.2f, trainProportion:%.2f, Gibbs Sampling]",
				number_of_topics, d_alpha, d_beta,
				1 - m_testWord4PerplexityProportion);
	}
	
	public void LoadPrior(String fileName, double eta) {
		if (fileName == null || fileName.isEmpty()) {
			return;
		}
		
		try{

			if (word_topic_prior == null) {
				word_topic_prior = new double[number_of_topics][vocabulary_size];
			}
				 
			for (int k = 0; k < number_of_topics; k++)
				Arrays.fill(word_topic_prior[k], 0);

			String tmpTxt;
			String[] lineContainer;
			String[] featureContainer;
			int tid = 0;
			
			HashMap<String, Integer> featureNameIndex = new HashMap<String, Integer>();
			for(int i=0; i<m_corpus.getFeatureSize(); i++){
				featureNameIndex.put(m_corpus.getFeature(i), featureNameIndex.size());
			}
			
			BufferedReader br = new BufferedReader(new InputStreamReader(
					new FileInputStream(fileName), "UTF-8"));

			while ((tmpTxt = br.readLine()) != null) {
				tmpTxt = tmpTxt.trim();
				if (tmpTxt.isEmpty())
					continue;

				lineContainer = tmpTxt.split("\t");
				
				tid = Integer.parseInt(lineContainer[0]);
				for (int i = 1; i < lineContainer.length; i++) {
					featureContainer = lineContainer[i].split(":");
					
					String featureName = featureContainer[0];
					double featureProb = Double.parseDouble(featureContainer[1]);
					
					int featureIndex = featureNameIndex.get(featureName);
					
					word_topic_prior[tid][featureIndex] = featureProb;
				}
			}

			System.out.println("prior is added");

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	protected void imposePrior() {
		if (word_topic_prior != null) {
			Arrays.fill(m_totalBeta, 0);
			for (int k = 0; k < number_of_topics; k++) {
				for (int v = 0; v < vocabulary_size; v++) {
					m_beta[k][v] = word_topic_prior[k][v];
					m_totalBeta[k] += m_beta[k][v];
				}
			}
		}
	}
	
	protected void initialize_probability(Collection<_Doc>collection){
		m_alpha_c = new double[number_of_topics];
		m_alphaAuxilary = new double[number_of_topics];

		m_alpha = new double[number_of_topics];
		m_beta = new double[number_of_topics][vocabulary_size];

		m_totalAlpha = 0;
		m_totalAlpha_c = 0;
		m_totalBeta = new double[number_of_topics];

		m_topic_word_prob = new double[number_of_topics][vocabulary_size];
		for (int k = 0; k < number_of_topics; k++)
			Arrays.fill(word_topic_sstat[k], 0);

		for(_Doc d:collection){
			if(d instanceof _ParentDoc4DCM){
				_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
				pDoc.setTopics4Gibbs(number_of_topics, 0, vocabulary_size);

				for (_Stn stnObj : d.getSentences()) {
					stnObj.setTopicsVct(number_of_topics);
				}

				for(_ChildDoc cDoc: pDoc.m_childDocs){
					cDoc.setTopics4Gibbs_LDA(number_of_topics, 0);
					for(_Word w:cDoc.getWords()){
						int wid = w.getIndex();
						int tid = w.getTopic();

						pDoc.m_wordTopic_stat[tid][wid] ++;
						pDoc.m_topic_stat[tid]++;
					}
					computeMu4Doc(cDoc);
				}
			}

		}

		initialAlphaBeta();
		// imposePrior();

	}

	/**
	 *the alpha, beta, alpha_c parameters does not make sense to the initialization.
	 * */
	protected void initialAlphaBeta(){
		Arrays.fill(m_alpha, 1.0/number_of_topics);
		Arrays.fill(m_alpha_c, 1.0/number_of_topics);

		for(int k=0; k<number_of_topics; k++){
			Arrays.fill(m_beta[k], 1.0/vocabulary_size);
			Arrays.fill(topic_term_probabilty[k], 1.0/vocabulary_size);
			Arrays.fill(word_topic_sstat[k], 0);
		}

		m_totalAlpha = Utils.sumOfArray(m_alpha);
		for (int k = 0; k < number_of_topics; k++) {
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);
		}
	}

	protected void computeMu4Doc(_ChildDoc d){
		_ParentDoc tempParent = d.m_parentDoc;
		double mu = Utils.cosine(tempParent.getSparse(), d.getSparse());

		mu = 0.05;// 0.5, 0.05, 0.005, 0.00001
		d.setMu(mu);
	}
	
	protected void computeTestMu4Doc(_ChildDoc d){
		_ParentDoc pDoc = d.m_parentDoc;
		
		double mu = Utils.cosine(d.getSparseVct4Infer(), pDoc.getSparseVct4Infer());
		mu = 0.05;// 0.5, 0.05, 0.005, 0.00001
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
			
			if(m_collectCorpusStats)
				word_topic_sstat[tid][wid]--;
			
			normalizedProb = 0;
			for(tid=0; tid<number_of_topics; tid++){
				double pWordTopic = parentWordByTopicProb(tid, wid, pDoc);
				double pTopicPDoc = parentTopicInDocProb(tid, pDoc);
				double pTopicCDoc = parentChildInfluenceProb(tid, pDoc);
				
				m_topicProbCache[tid] = pWordTopic*pTopicPDoc*pTopicCDoc;
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
			
			if(m_collectCorpusStats)
				word_topic_sstat[tid][wid]++;
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
			if (m_collectCorpusStats)
				word_topic_sstat[tid][wid]--;

			normalizedProb = 0;
			for (tid = 0; tid < number_of_topics; tid++) {
				double pWordTopic = childWordByTopicProb(tid, wid, pDoc);
				double pTopic = childTopicInDocProb(tid, d, pDoc);
				
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

			if (m_collectCorpusStats)
				word_topic_sstat[tid][wid]++;
		}
	}

	protected double parentWordByTopicProb(int tid, int wid, _ParentDoc4DCM d) {
		double prob = 0;
		prob = (d.m_wordTopic_stat[tid][wid] + m_beta[tid][wid])
				/ (d.m_topic_stat[tid] + m_totalBeta[tid]);
		
		return prob;
	}
	
	protected double parentTopicInDocProb(int tid, _ParentDoc4DCM d) {
		double prob = 0;
		
		prob = (d.m_sstat[tid] + m_alpha[tid])
				/ (d.getDocInferLength() + m_totalAlpha);

		return prob;
	}
	
	protected double parentChildInfluenceProb(int tid, _ParentDoc4DCM d) {
		double term = 1.0;

		if (tid == 0)
			return term;
		
		for (_ChildDoc cDoc : d.m_childDocs) {
			double muDp = cDoc.getMu() / d.getDocInferLength();
			term *= gammaFuncRatio((int) cDoc.m_sstat[tid], muDp, m_alpha_c[tid]
					+ d.m_sstat[tid] * muDp)
					/ gammaFuncRatio((int) cDoc.m_sstat[0], muDp, m_alpha_c[0]
							+ d.m_sstat[0] * muDp);
		}

		return term;

	}

	protected double gammaFuncRatio(int nc, double muDp, double alphaMuDp) {
		if (nc == 0)
			return 1.0;

		double result = 1.0;
		for (int n = 1; n <= nc; n++) {
			result *= 1 + muDp / (alphaMuDp + n - 1);
		}

		return result;
	}
	
	protected double childWordByTopicProb(int tid, int wid, _ParentDoc4DCM d) {
		double prob = 0;
		prob = (d.m_wordTopic_stat[tid][wid] + m_beta[tid][wid])
				/ (d.m_topic_stat[tid] + m_totalBeta[tid]);
		return prob;
	}
	
	protected double childTopicInDocProb(int tid, _ChildDoc d, _ParentDoc4DCM pDoc) {
		double prob = 0;
		double childTopicSum = Utils.sumOfArray(d.m_sstat);
		double parentTopicSum = Utils.sumOfArray(pDoc.m_sstat);
			
		double muDp = d.getMu() / parentTopicSum;
		prob = (m_alpha_c[tid] + muDp * pDoc.m_sstat[tid] + d.m_sstat[tid])
				/ (m_totalAlpha_c + muDp * parentTopicSum + childTopicSum);
		
		return prob;
	}
	
	protected void updateAlpha(){
		double diff = 0;
		int iteration = 0;
		
		do{
			diff = 0;
			
			double[] wordNum4Tid = new double[number_of_topics];
			double totalAlphaDenominator = 0;
			m_totalAlpha = Utils.sumOfArray(m_alpha);
			double digAlpha = Utils.digamma(m_totalAlpha);
			Arrays.fill(wordNum4Tid, 0);
			double deltaAlpha = 0;
			
			for(_Doc d:m_trainSet){
				if(d instanceof _ParentDoc){
					totalAlphaDenominator += Utils.digamma(d.getTotalDocLength()+m_totalAlpha) - digAlpha;
				}
			}
			
			for(int k=0; k<number_of_topics; k++){
				double totalAlphaNumerator = 0;
				
				wordNum4Tid[k] = 0;
				for(_Doc d:m_trainSet){
					if(d instanceof _ParentDoc){
						totalAlphaNumerator += Utils.digamma(m_alpha[k]+d.m_sstat[k])-Utils.digamma(m_alpha[k]);
						wordNum4Tid[k] += d.m_sstat[k];
					}
				}
				
				if(wordNum4Tid[k]==0){
					deltaAlpha = 0;
				}else{
					deltaAlpha = totalAlphaNumerator*1.0/totalAlphaDenominator;
				}
				
				double newAlpha = m_alpha[k]*deltaAlpha+d_alpha;
				double t_diff = Math.abs(m_alpha[k]-newAlpha);
				
				if(t_diff>diff)
					diff = t_diff;
				
				m_alpha[k] = newAlpha;
			}
			
			iteration ++;
			// System.out.println("alpha  iteration\t" + iteration);
			
			if(iteration>m_newtonIter)
				break;
			
		}while(diff>m_newtonConverge);
		
//		System.out.println("iteration\t" + iteration);
		m_totalAlpha = 0;
		for(int k=0; k<number_of_topics; k++){
			m_totalAlpha += m_alpha[k];
			System.out.println("alpha\t"+m_alpha[k]);
		}
	}
	
	protected void updateAlphaC(){
		double diff = 0;
		int iteration = 0;
		
		do{
			diff = 0;
			double totalAlphaDenominator = 0;
			double[] wordNum4Tid = new double[number_of_topics];
			double[] totalAlphaNumerator = new double[number_of_topics];
			Arrays.fill(totalAlphaNumerator, 0);
			m_totalAlpha_c = Utils.sumOfArray(m_alpha_c);
			Arrays.fill(wordNum4Tid, 0);
						
			double deltaAlpha = 0;
			for(_Doc d:m_trainSet){
				if(d instanceof _ParentDoc){
					_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
					
					double pDocLen = pDoc.getTotalDocLength();
					for(_ChildDoc cDoc:pDoc.m_childDocs){
						double muDp = cDoc.getMu()/pDocLen;
						double t_totalAlpha_c = m_totalAlpha_c+cDoc.getMu();
						double digAlpha = Utils.digamma(t_totalAlpha_c);
						totalAlphaDenominator += Utils.digamma(cDoc.getTotalDocLength()+t_totalAlpha_c)-digAlpha;
						
						for(int k=0; k<number_of_topics; k++){
							wordNum4Tid[k] += cDoc.m_sstat[k];
							totalAlphaNumerator[k] += Utils.digamma(m_alpha_c[k]+muDp*pDoc.m_sstat[k]+cDoc.m_sstat[k])-Utils.digamma(m_alpha_c[k]+muDp*pDoc.m_sstat[k]);
						}
					}
				}
			}
			
			for(int k=0; k<number_of_topics; k++){
				if(wordNum4Tid[k]==0){
					deltaAlpha = 0;
				}else{
					deltaAlpha = totalAlphaNumerator[k]*1.0/totalAlphaDenominator;
				}
				
				double newAlpha = m_alpha_c[k]*deltaAlpha+d_alpha_c;
				double t_diff = Math.abs(m_alpha_c[k]-newAlpha);
				if(t_diff>diff)
					diff = t_diff;
				
				m_alpha_c[k] = newAlpha;
			}
			
			iteration ++;
			// System.out.println("alpha iteration\t" + iteration);
			
			if(iteration > m_newtonIter)
				break;
			
		}while(diff>m_newtonConverge);
		
		// System.out.println("iteration\t" + iteration);
		m_totalAlpha_c = 0;
		for (int k = 0; k < number_of_topics; k++) {
			m_totalAlpha_c += m_alpha_c[k];
		}
	}
	
	protected void updateBeta(int tid){
		double diff = 0;
		
		int iteration = 0;
		do{
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
			
			for(_Doc d:m_trainSet){
				if(d instanceof _ParentDoc){
					_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
					totalBetaDenominator += Utils.digamma(m_totalBeta[tid]+pDoc.m_topic_stat[tid])-digBeta4Tid;
					for(int v=0; v<vocabulary_size; v++){
						wordNum4Tid += pDoc.m_wordTopic_stat[tid][v];
						wordNum4Tid4V[v] += pDoc.m_wordTopic_stat[tid][v];
						
						totalBetaNumerator[v] += Utils.digamma(m_beta[tid][v]+pDoc.m_wordTopic_stat[tid][v]);
						totalBetaNumerator[v] -= Utils.digamma(m_beta[tid][v]);
					}
				}
				
			}
			
			for(int v=0; v<vocabulary_size; v++){
				if(wordNum4Tid == 0)
					break;
				if(wordNum4Tid4V[v] == 0){
					deltaBeta = 0;
				}else{
					deltaBeta = totalBetaNumerator[v]/totalBetaDenominator;
				}
					
				double newBeta = m_beta[tid][v]*deltaBeta+d_beta;
				double t_diff = Math.abs(m_beta[tid][v]-newBeta);
				if(t_diff>diff)
					diff = t_diff;
				
				m_beta[tid][v] = newBeta;
			}
			
			iteration ++;
			if(iteration > m_newtonIter)
				break;
			
			// System.out.println("beta iteration\t"+iteration);
		}while(diff > m_newtonConverge);
		
//		System.out.println("beta iteration\t" + iteration);
	}
	
	protected double calculate_log_likelihood(){
		double logLikelihood = 0.0;
		for(_Doc d:m_trainSet){
			if(d instanceof _ParentDoc)
				logLikelihood += calculate_log_likelihood((_ParentDoc4DCM)d);
		}
		
		return logLikelihood;
	}

	public void updateParameter(int iter, File weightFolder){
		File weightIterFolder = new File(weightFolder, "_" + iter);
		if (!weightIterFolder.exists()) {
			weightIterFolder.mkdir();
		}
		
		initialAlphaBeta();
		updateAlpha();
		updateAlphaC();

		for (int k = 0; k < number_of_topics; k++)
			updateBeta(k);

		for (int k = 0; k < number_of_topics; k++)
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);

		String fileName = iter + ".txt";
//		saveParameter2File(weightIterFolder, fileName);
		
	}

	protected void collectStats(_Doc d){
		if(d instanceof _ParentDoc4DCM){
			_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
			for(int k=0; k<number_of_topics; k++){
				pDoc.m_topics[k] += pDoc.m_sstat[k]+m_alpha[k];
				for(int v=0; v<vocabulary_size; v++){
					pDoc.m_wordTopic_prob[k][v] += pDoc.m_wordTopic_stat[k][v]+m_beta[k][v];
				}
			}
		}else if(d instanceof _ChildDoc){
			_ChildDoc cDoc = (_ChildDoc)d;
			_ParentDoc pDoc = cDoc.m_parentDoc;
			double topicSum = Utils.sumOfArray(pDoc.m_sstat);
			double muDp = cDoc.getMu()/topicSum;
			for(int k=0; k<number_of_topics; k++){
				cDoc.m_topics[k] += cDoc.m_sstat[k]+m_alpha_c[k];
			}
		}
	}
	
	protected void saveParameter2File(File fileFolder, String fileName){
		try{
			File paramFile = new File(fileFolder, fileName);
			
			PrintWriter pw = new PrintWriter(paramFile);
			pw.println("alpha");
			
			for(int k=0; k<number_of_topics; k++){
				pw.print(m_alpha[k]+"\t");
				System.out.print(m_alpha[k]+"\t");
			}
			pw.println();
			pw.println("alpha c");
			for(int k=0; k<number_of_topics; k++){
				pw.print(m_alpha_c[k]+"\t");
				System.out.print(m_alpha_c[k]+"\t");

			}
			pw.println();
			pw.println("beta");
			for (int k = 0; k < number_of_topics; k++) {
				pw.print("topic" + k + "\t");
				System.out.print("topic" + k + "\t");
				for (int v = 0; v < vocabulary_size; v++) {
					pw.print(m_beta[k][v] + "\t");
					System.out.print(m_beta[k][v] + "\t");

				}
				pw.println();
			}
			pw.flush();
			pw.close();
		}catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}
	
	protected double calculate_log_likelihood(_ParentDoc4DCM d){
		double docLogLikelihood = 0;
		int docID = d.getID();
		
		double parentDocLength = d.getTotalDocLength();
		
		for(int k=0; k<number_of_topics; k++){
			double term = Utils.lgamma(d.m_sstat[k]+m_alpha[k]);
			docLogLikelihood += term;
			
			term = Utils.lgamma(m_alpha[k]);
			docLogLikelihood -= term;
		}
		
		docLogLikelihood += Utils.lgamma(m_totalAlpha);
		docLogLikelihood -= Utils.lgamma(parentDocLength+m_totalAlpha);
		
		for(int k=0; k<number_of_topics; k++){
			for(int v=0; v<vocabulary_size; v++){
				double term = Utils.lgamma(d.m_wordTopic_stat[k][v]+m_beta[k][v]);
				docLogLikelihood += term;
				
				term = Utils.lgamma(m_beta[k][v]);
				docLogLikelihood -= term;
			}
			
			docLogLikelihood += Utils.lgamma(m_totalBeta[k]);
			docLogLikelihood -= Utils.lgamma(d.m_topic_stat[k]+m_totalBeta[k]);
		}
		
		for(_ChildDoc cDoc:d.m_childDocs){
			double muDp = cDoc.getMu()/parentDocLength;
			docLogLikelihood += Utils.lgamma(m_totalAlpha_c+cDoc.getMu());
			docLogLikelihood -= Utils.lgamma(m_totalAlpha_c+cDoc.getMu()+cDoc.getTotalDocLength());
			double term = 0;
			for(int k=0; k<number_of_topics; k++){
				double term1 = 0;
				term1 += Utils.lgamma(m_alpha_c[k]+muDp*d.m_sstat[k]+cDoc.m_sstat[k]);
				term1 -= Utils.lgamma(m_alpha_c[k]+muDp*d.m_sstat[k]);	
//				if(term1>0)
//					System.out.println("+\t"+(m_alpha_c[k]+muDp*d.m_sstat[k]+cDoc.m_sstat[k])+"-\t"+(m_alpha_c[k]+muDp*d.m_sstat[k]));
				term += term1;
			}
			docLogLikelihood += term;
		}
		
		return docLogLikelihood;
		
	}
	
	protected void estThetaInDoc(_Doc d) {

		if (d instanceof _ParentDoc4DCM) {
			for (int i = 0; i < number_of_topics; i++)
				Utils.L1Normalization(((_ParentDoc4DCM) d).m_wordTopic_prob[i]);
		}
		Utils.L1Normalization(d.m_topics);
		
	}
	
	protected double calConditionalPerplexity(ArrayList<_Doc> sampleTestSet) {
		double logLikelihood = 0;
		
		for (_Doc d : sampleTestSet) {
			estThetaInDoc(d);
		}

		for (_Doc d : sampleTestSet) {
			if (d instanceof _ChildDoc) {
				logLikelihood += cal_logLikelihood_partial4Child((_ChildDoc) d);
			}
		}

		return logLikelihood;
	}

	protected double calPerplexity(ArrayList<_Doc> sampleTestSet) {
		double logLikelihood = 0;

		for (_Doc d : sampleTestSet) {
			estThetaInDoc(d);
		}
		
		for (_Doc d : sampleTestSet) {
			if(d instanceof _ParentDoc)
				logLikelihood += cal_log_likelihood4Parent((_ParentDoc) d);
			else
				logLikelihood += cal_log_likelihood4Child((_ChildDoc) d);
		}

		return logLikelihood;
	}
	
	protected double cal_log_likelihood4Parent(_ParentDoc d) {
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d;
		double parentLikelihood = 0;
		for (_Word w : pDoc.getWords()) {
			int wid = w.getIndex();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = pDoc.m_topics[k]
						* pDoc.m_wordTopic_prob[k][wid];
				wordLogLikelihood += wordPerTopicLikelihood;
			}

			parentLikelihood += Math.log(wordLogLikelihood);
		}

		return parentLikelihood;
		
	}
	
	protected double cal_log_likelihood4Child(_ChildDoc d) {
		double childLikelihood = 0;

		_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d.m_parentDoc;

		for (_Word w : d.getWords()) {
			int wid = w.getIndex();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = d.m_topics[k]
						* pDoc.m_wordTopic_prob[k][wid];
				wordLogLikelihood += wordPerTopicLikelihood;
			}

			childLikelihood += Math.log(wordLogLikelihood);
		}

		return childLikelihood;

	}
	
	protected double calPerplexity4Child(ArrayList<_Doc> sampleTestSet) {
		double logLikelihood = 0;
		for(_Doc d:sampleTestSet){
			estThetaInDoc(d);
			
		}
		
		for(_Doc d:sampleTestSet){
			if(d instanceof _ChildDoc)
				logLikelihood += cal_logLikelihood_4Child(d);
		}
		return logLikelihood;
	}

	protected double cal_logLikelihood_4Child(_Doc d){
		double docLogLikelihood = 0.0;
		_ChildDoc cDoc = (_ChildDoc)d;
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM)cDoc.m_parentDoc;
		for (_Word w : d.getTestWords()) {
			int wid = w.getIndex();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = pDoc.m_topics[k]
						* pDoc.m_wordTopic_prob[k][wid];
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			docLogLikelihood += Math.log(wordLogLikelihood);
		}

		return docLogLikelihood;
	}

	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d){
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
		for(_Stn stnObj:pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}
		
		int testLength = 0;
//		testLength = (int) (m_testWord4PerplexityProportion * pDoc.getTotalDocLength());
		pDoc.setTopics4GibbsTest(number_of_topics, 0, testLength, vocabulary_size);
		pDoc.createSparseVct4Infer();

		sampleTestSet.add(pDoc);
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			testLength = (int)(m_testWord4PerplexityProportion*cDoc.getTotalDocLength());
//			testLength = cDoc.getTotalDocLength();
			testLength = 0;
			cDoc.setTopics4GibbsTest(number_of_topics, 0, testLength);
			
			for(_Word w:cDoc.getWords()){
				int wid = w.getIndex();
				int tid = w.getTopic();
				
				pDoc.m_wordTopic_stat[tid][wid] ++;
				pDoc.m_topic_stat[tid] ++;
			}
			
			sampleTestSet.add(cDoc);
			cDoc.createSparseVct4Infer();
			//cDoc computeMu
			computeTestMu4Doc(cDoc);
		}
		
	}

	protected double cal_logLikelihood_partial4Parent(_Doc d) {
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
		double docLogLikelihood = 0.0;

		for (_Word w : pDoc.getTestWords()) {
			int wid = w.getIndex();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = pDoc.m_topics[k]
						*pDoc.m_wordTopic_prob[k][wid];
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			docLogLikelihood += Math.log(wordLogLikelihood);
		}

		return docLogLikelihood;
	}

	protected double cal_logLikelihood_partial4Child(_Doc d) {

		double childLikelihood = 0;

		_ChildDoc cDoc = (_ChildDoc)d;
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM)cDoc.m_parentDoc;
		
		for(_Word w:cDoc.getTestWords()){
			int wid= w.getIndex();
			
			double wordLogLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = cDoc.m_topics[k]*pDoc.m_wordTopic_prob[k][wid];
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			
			childLikelihood += Math.log(wordLogLikelihood);
			// System.out.println("word log likelihood\t" + childLikelihood);
		}
		
		return childLikelihood;
	}
	
	protected double calculate_log_likelihood4ParentPerplexity(_Doc d){
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d;
		double docLogLikelihood = 0.0;

		for (_Word w : pDoc.getWords()) {
			int wid = w.getIndex();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = pDoc.m_topics[k]
						* pDoc.m_wordTopic_prob[k][wid];
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			docLogLikelihood += Math.log(wordLogLikelihood);
		}

		return docLogLikelihood;
	} 
	
	protected double calculate_log_likelihood4ChildPerplexity(_Doc d){
		_ChildDoc cDoc = (_ChildDoc)d;
		double docLogLikelihood = 0.0;

		_ParentDoc4DCM pDoc = (_ParentDoc4DCM)cDoc.m_parentDoc;

		for (_Word w : cDoc.getWords()) {
			int wid = w.getIndex();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = cDoc.m_topics[k]
						* pDoc.m_wordTopic_prob[k][wid];
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			docLogLikelihood += Math.log(wordLogLikelihood);
		}

		return docLogLikelihood;
	}
	
	public void printTopWords(int k, String betaFile) {
		try {
			PrintWriter topWordWriter = new PrintWriter(new File(betaFile));

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

	protected double cal_logLikelihood_Perplexity4Parent(_Doc d){
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM) d;
		double docLogLikelihood = 0.0;

		for (_Word w : pDoc.getWords()) {
			int wid = w.getIndex();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = pDoc.m_topics[k]
						*pDoc.m_wordTopic_prob[k][wid];
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			docLogLikelihood += Math.log(wordLogLikelihood);
		}

		return docLogLikelihood;
	}

	protected double cal_logLikelihood_Perplexity4Child(_Doc d){
		_ChildDoc cDoc = (_ChildDoc)d;
		double docLogLikelihood = 0.0;
		_ParentDoc4DCM pDoc = (_ParentDoc4DCM)cDoc.m_parentDoc;

		for (_Word w : cDoc.getWords()) {
			int wid = w.getIndex();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = cDoc.m_topics[k]
						* pDoc.m_wordTopic_prob[k][wid];
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			docLogLikelihood += Math.log(wordLogLikelihood);
		}

		return docLogLikelihood;
	}
}

