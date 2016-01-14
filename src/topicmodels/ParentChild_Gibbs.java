package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import structures.MyPriorityQueue;
import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._RankItem;
import structures._SparseFeature;
import structures._Stn;
import utils.Utils;

public class ParentChild_Gibbs extends LDA_Gibbs {
	double[] m_gamma, m_topicProbCache;
	double[][] m_xTopicProbCache;
	double m_mu;
	
	public ParentChild_Gibbs(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double mu) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag);

		m_mu = mu;
		m_gamma = new double[gamma.length];
		System.arraycopy(gamma, 0, m_gamma, 0, gamma.length);
		m_topicProbCache = new double[number_of_topics];
		m_xTopicProbCache = new double[gamma.length][number_of_topics];
	}
	
	@Override
	public String toString(){
		return String.format("Parent Child topic model [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta, m_gamma[1], m_gamma[2]);
	}
	
	//will be called before entering EM iterations
	@Override
	protected void initialize_probability(Collection<_Doc> collection){
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], 0);
		Arrays.fill(m_sstat, 0);
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc){
				for(_Stn stnObj: d.getSentences()){
					stnObj.setTopicsVct(number_of_topics);
				}
			}
			if(d instanceof _ChildDoc)
				((_ChildDoc) d).createXSpace(number_of_topics, m_gamma.length);
			
			d.setTopics4Gibbs(number_of_topics, 0);
			for (int i = 0; i < d.m_words.length; i++) {
				word_topic_sstat[d.m_topicAssignment[i]][d.m_words[i]]++;
				m_sstat[d.m_topicAssignment[i]]++;
			}			
		}
		
		imposePrior();
	}
	
	@Override
	public double calculate_E_step(_Doc d){
		d.permutation();
		
		if(d instanceof _ParentDoc)
			sampleInParentDoc((_ParentDoc)d);
		else if(d instanceof _ChildDoc)
			sampleInChildDoc((_ChildDoc)d);
		
		return 1;
	}
	
	void sampleInParentDoc(_ParentDoc d){
		int wid, tid;
		double normalizedProb;		
		
		for(int i=0; i<d.m_words.length; i++){
			wid = d.m_words[i];
			tid = d.m_topicAssignment[i];
			
			d.m_sstat[tid] --;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}

			normalizedProb = 0;
			for(tid=0; tid<number_of_topics; tid++){
				double term1 = parentWordByTopicProb(tid, wid);
				double term2 = topicInParentDocProb(tid, d);
				double term3 = parentChildInfluenceProb(tid, d);
					
				m_topicProbCache[tid] = term1*term2*term3;
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
			
			d.m_topicAssignment[i] = tid;
			d.m_sstat[tid] ++;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] ++;
				m_sstat[tid] ++;
			}
		}
	}

	//probability of word given topic p(w|z, phi^p, beta)
	protected double parentWordByTopicProb(int tid, int wid){
		return (d_beta+word_topic_sstat[tid][wid])/(d_beta*vocabulary_size+m_sstat[tid]);
	}

	//probability of topic given doc p(z|d, alpha)
	protected double topicInParentDocProb(int tid, _ParentDoc d){
		return d_alpha+d.m_sstat[tid];
	}

	protected double parentChildInfluenceProb(int tid, _ParentDoc d){
		double term = 0;
		double docLength = d.getTotalDocLength()/m_mu;
		for (_ChildDoc cDoc : d.m_childDocs) {
			double term11 = (Utils.lgamma(d_alpha + (d.m_sstat[0]) / (docLength) + cDoc.m_xTopicSstat[0][0]));
			double term12 = (Utils.lgamma(d_alpha + (d.m_sstat[0] + 1) / (docLength)+ cDoc.m_xTopicSstat[0][0]));

			double term21 = (Utils.lgamma(d_alpha + (d.m_sstat[tid] + 1) / (docLength) + cDoc.m_xTopicSstat[0][tid]));
			double term22 = (Utils.lgamma(d_alpha + (d.m_sstat[tid]) / (docLength) + cDoc.m_xTopicSstat[0][tid]));

			double term31 = (Utils.lgamma(d_alpha + (d.m_sstat[0] + 1) / (docLength)));
			double term32 = (Utils.lgamma(d_alpha + (d.m_sstat[0]) / (docLength)));

			double term41 = (Utils.lgamma(d_alpha + (d.m_sstat[tid]) / (docLength)));
			double term42 = (Utils.lgamma(d_alpha + (d.m_sstat[tid] + 1) / (docLength)));

			term += term11 - term12 + term21 - term22 + term31 - term32 + term41 - term42;

			if (invalidValue(term11) || invalidValue(term12)
							|| invalidValue(term21) || invalidValue(term22)
							|| invalidValue(term31) || invalidValue(term32)
							|| invalidValue(term41) || invalidValue(term42)
							|| invalidValue(term)) {
				System.out.println("invalid term");
			}
		} 

		return Math.exp(term);
	}

	public boolean invalidValue(double term) {
		if (Math.abs(term) > Double.MAX_VALUE - 1) {
			System.out.println(term);
			return true;
		} else {
			return false;
		}
	}

	public void sampleInChildDoc(_ChildDoc d){
		int wid, tid, xid;		
		double normalizedProb;
		
		for(int i=0; i<d.m_words.length; i++){			
			wid = d.m_words[i];
			tid = d.m_topicAssignment[i];
			xid = d.m_xIndicator[i];
			
			d.m_xTopicSstat[xid][tid] --;
			d.m_xSstat[xid] --;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid]--;
				m_sstat[tid]--;
			}
			
			normalizedProb = 0;
			for(tid=0; tid<number_of_topics; tid++){
				double term1 = childWordByTopicProb(tid, wid);
				
				//p(z=tid,x=1) from specific
				double term2 = childTopicInDocProb(tid, 1, d);
				double term3 = childXInDocProb(1, d);
				
				//p(z=tid,x=0) from background
				double term4 = childTopicInDocProb(tid, 0, d);
				double term5 = childXInDocProb(0, d);
				
				m_xTopicProbCache[1][tid] = term1*term2*term3;
				normalizedProb += m_xTopicProbCache[1][tid];
				
				m_xTopicProbCache[0][tid] = term1*term4*term5;
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

			if (xid == 2)
				xid--;
			
			if (tid == number_of_topics)
				tid--;
			
			d.m_topicAssignment[i] = tid;
			d.m_xIndicator[i] = xid;

			d.m_xTopicSstat[xid][tid] ++;
			d.m_xSstat[xid] ++;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid]++;
				m_sstat[tid]++;
			}			
		}
	}

	//probability of word given topic p(w|z, phi^c, beta)
	public double childWordByTopicProb(int tid, int wid){
		return (d_beta + word_topic_sstat[tid][wid])/(d_beta*vocabulary_size + m_sstat[tid]);
	}

	//probability of topic in given child doc p(z^c|d, alpha, z^p)
	public double childTopicInDocProb(int tid, int xid, _ChildDoc d){
		double docLength = d.m_parentDoc.getTotalDocLength();

		if(xid == 1){
			return (d_alpha + d.m_xTopicSstat[1][tid])
					/(number_of_topics*d_alpha + d.m_xSstat[1]);
		} else if(xid == 0){
			return (d_alpha + m_mu*d.m_parentDoc.m_sstat[tid]/docLength + d.m_xTopicSstat[0][tid])
					/(number_of_topics*d_alpha + m_mu + d.m_xSstat[0]);
		} else
			return -1;//this branch is impossible
	}

	public double childXInDocProb(int xid, _ChildDoc d){
		return m_gamma[xid]+ d.m_xSstat[xid];
	}	

	public void calculate_M_step(int iter){
//		if (iter % m_lag == 0) 
//			calLogLikelihood2(iter);

		if(iter>m_burnIn && iter%m_lag==0){
			for(int i=0; i<this.number_of_topics; i++){
				for(int v=0; v<this.vocabulary_size; v++){
					topic_term_probabilty[i][v] += word_topic_sstat[i][v];
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
	
	protected void collectParentStats(_ParentDoc d) {
		for (int k = 0; k < this.number_of_topics; k++) {
			d.m_topics[k] += (d.m_sstat[k] + d_alpha);
		}
		collectStnStats(d);
	}
	
	protected void collectChildStats(_ChildDoc d) {
		for (int j = 0; j < m_gamma.length; j++)
			d.m_xProportion[j] += d.m_xSstat[j] + m_gamma[j];

		double parentDocLength = d.m_parentDoc.getTotalDocLength();
		// used to output the topK words and parameters
		for (int k = 0; k < this.number_of_topics; k++) {
			d.m_xTopics[1][k] += (d.m_xTopicSstat[1][k] + d_alpha);
			d.m_xTopics[0][k] += d.m_xTopics[0][k] += (d.m_xTopicSstat[0][k] + d_alpha + m_mu
					* d.m_parentDoc.m_sstat[k] / parentDocLength);
			d.m_topics[k] += d.m_xTopics[1][k] + d.m_xTopics[0][k];
		}
	}
	
	public void collectStnStats(_Doc d) {
		_SparseFeature[] fv = d.getSparse();
		double[][] phi = new double[fv.length][number_of_topics];
		HashMap<Integer, Integer> indexMap = new HashMap<Integer, Integer>();

		// //computeWordTopicProportionInDoc
		////compute phi
		for (int i = 0; i < fv.length; i++) {
			int index = fv[i].getIndex();
			indexMap.put(index, i);
		}

		for (int n = 0; n < d.m_words.length; n++) {
			int index = d.m_words[n];
			int topic = d.m_topicAssignment[n];
			phi[indexMap.get(index)][topic]++;
		}

		for (int i = 0; i < fv.length; i++) {
			Utils.L1Normalization(phi[i]);
		}

		for (_Stn stnObject:d.getSentences()) {
			// initial topic proportions (m_topics) of sentences
			_SparseFeature[] sv = stnObject.getFv();
			
			//m_stnLength: the length of sentence
			//m_words: the index in CV of each word in the sentence 
			for (int j = 0; j < sv.length; j++) {
				int index = sv[j].getIndex();
				double value = sv[j].getValue();
				for (int k = 0; k < number_of_topics; k++) {
					stnObject.m_topics[k] += value*phi[indexMap.get(index)][k];
				}
			}
		}

	}

	
	protected void finalEst() {
		normalizedTopicTermProb();

		for (_Doc d : m_trainSet)
			estThetaInDoc(d);
		discoverSpecificComments();
	}

	protected void normalizedTopicTermProb(){
		for (int i = 0; i < this.number_of_topics; i++) {
			Utils.L1Normalization(topic_term_probabilty[i]);
		}

	}
	
	protected void estThetaInDoc(_Doc d) {
		if (d instanceof _ParentDoc){
			Utils.L1Normalization(d.m_topics);

		// estimate topic proportion of sentences in parent documents
			estStnThetaInParentDoc((_ParentDoc) d);
		} else if (d instanceof _ChildDoc) {
			Utils.L1Normalization(((_ChildDoc) d).m_xProportion);
			Utils.L1Normalization(d.m_topics);
			for(int x=0; x<m_gamma.length; x++){
				Utils.L1Normalization(((_ChildDoc) d).m_xTopics[x]);
			}
		}

	}
	
	public void estStnThetaInParentDoc(_Doc d){
		for(_Stn stnObj: d.getSentences()){
			Utils.L1Normalization(stnObj.m_topics);
		}
	}
	
	
	public void discoverSpecificComments() {
		System.out.println("topic similarity");
		String fileName = "topicSimilarity.txt";

		try {
			PrintWriter pw = new PrintWriter(new File(fileName));

			for (_Doc doc : m_trainSet) {
				if (doc instanceof _ParentDoc) {
					pw.print(doc.getName() + "\t");
					double stnTopicSimilarity = 0.0;
					double docTopicSimilarity = 0.0;
					for (_ChildDoc cDoc : ((_ParentDoc) doc).m_childDocs) {
						pw.print(cDoc.getName() + ":");

						docTopicSimilarity = computeSimilarity(
								((_ParentDoc) doc).m_topics, cDoc.m_topics);
						pw.print(docTopicSimilarity);
						for (_Stn stnObj:doc.getSentences()) {
							double[] stnTopics = stnObj.m_topics;
							stnTopicSimilarity = computeSimilarity(stnTopics,
									cDoc.m_topics);
//							pw.println("add");
//							System.out.println("index"+stnObj.getIndex());
							pw.print(":"+(stnObj.getIndex()+1) + ":"+stnTopicSimilarity);
						}
						pw.print("\t");
					}
					pw.println();
				} else {
					continue;
				}
			}
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public double computeSimilarity(double[] topic1, double[] topic2) {
		return Utils.cosine(topic1, topic2);
	}
	
	public void printTopWords(int k, String betaFile) {
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

		for (_Doc d : m_trainSet) {
		if (d instanceof _ParentDoc) {
				printParentTopicAssignment((_ParentDoc) d, parentTopicFolder);
			} else if (d instanceof _ChildDoc) {
				printChildTopicAssignment((_ChildDoc) d, childTopicFolder);
			}

		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";
		printParameter(parentParameterFile, childParameterFile);

	}

	public void printParentTopicAssignment(_ParentDoc d, File parentFolder) {
	//	System.out.println("printing topic assignment parent documents");
		
		String topicAssignmentFile = d.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(parentFolder,
					topicAssignmentFile));
			
			for(int n=0; n<d.m_words.length; n++){
				int index = d.m_words[n];
				int topic = d.m_topicAssignment[n];
				String featureName = m_corpus.getFeature(index);
				pw.print(featureName + ":" + topic + "\t");
			}
			
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public void printChildTopicAssignment(_ChildDoc d, File childFolder) {
	//	System.out.println("printing topic assignment child documents");
		
		String topicAssignmentfile = d.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(childFolder,
					topicAssignmentfile));

			for (int n = 0; n < d.m_words.length; n++) {
				int index = d.m_words[n];
				int topic = d.m_topicAssignment[n];
				String featureName = m_corpus.getFeature(index);
					
				pw.print(featureName + ":" + topic + "\t");
			}
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
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
			for(_Doc d: m_trainSet){
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
//			e.printStackTrace();
//			System.err.print("para File Not Found");
		}

	}

	//p(w, z)=p(w|z)p(z) multinomial-dirichlet
	public void calLogLikelihood(int iter) {
		double logLikelihood = 0.0;
		double parentLogLikelihood = 0.0;
		double childLogLikelihood = 0.0;

		for (_Doc d : m_trainSet) {
			if (d instanceof _ParentDoc) {
				collectParentStats((_ParentDoc) d);
				parentLogLikelihood += calParentLogLikelihood((_ParentDoc) d);
			} else if (d instanceof _ChildDoc) {
				collectChildStats((_ChildDoc) d);
				childLogLikelihood += calChildLogLikelihood((_ChildDoc) d);
			}
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

		System.out.format("iter %d, parent log likelihood %.3f\n", iter,
				parentLogLikelihood);
		infoWriter.format("iter %d, parent log likelihood %.3f\n", iter,
				parentLogLikelihood);
		System.out.format("iter %d, child log likelihood %.3f\n", iter,
				childLogLikelihood);
		infoWriter.format("iter %d, child log likelihood %.3f\n", iter,
				childLogLikelihood);
		logLikelihood = parentLogLikelihood + childLogLikelihood;

		System.out
				.format("iter %d, log likelihood %.3f\n", iter, logLikelihood);
		infoWriter
				.format("iter %d, log likelihood %.3f\n", iter, logLikelihood);
	}
	
	// log space
	public double calParentLogLikelihood(_ParentDoc pDoc) {
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
	public double calChildLogLikelihood(_ChildDoc cDoc) {
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
			term13 += Utils.lgamma(d_alpha + cDoc.m_parentDoc.m_sstat[k]
					+ cDoc.m_xTopicSstat[0][k]);
			
			term21 += Utils.lgamma(d_alpha + cDoc.m_xTopicSstat[1][k]);
		}
		term11 = Utils.lgamma(number_of_topics * d_alpha
				+ cDoc.m_parentDoc.getTotalDocLength());
		term14 = -(Utils.lgamma(number_of_topics * d_alpha
				+ cDoc.m_parentDoc.getTotalDocLength() + cDoc.m_xSstat[0]));

		tempLogLikelihood1 = term11 + term12 + term13 + term14;

		tempLogLikelihood2 = Utils.lgamma(number_of_topics * d_alpha)
				- number_of_topics * Utils.lgamma(d_alpha) + term21
				- Utils.lgamma(number_of_topics * d_alpha + cDoc.m_xSstat[1]);

		weight1 = Utils.lgamma(m_gamma[0] + m_gamma[1])
				- Utils.lgamma(m_gamma[0]) - Utils.lgamma(m_gamma[1])
				+ Utils.lgamma(m_gamma[0] + cDoc.m_xSstat[0])
				+ Utils.lgamma(m_gamma[1])
				- Utils.lgamma(m_gamma[0] + m_gamma[1] + cDoc.m_xSstat[0]);

		weight2 = Utils.lgamma(m_gamma[0] + m_gamma[1])
				- Utils.lgamma(m_gamma[0]) - Utils.lgamma(m_gamma[1])
				+ Utils.lgamma(m_gamma[0])
				+ Utils.lgamma(m_gamma[1] + cDoc.m_xSstat[1])
				- Utils.lgamma(m_gamma[0] + m_gamma[1] + cDoc.m_xSstat[1]);


		// tempLogLikelihood = tempLogLikelihood1 * cDoc.m_xProportion[0]
		// + tempLogLikelihood2 * cDoc.m_xProportion[1];
		
		tempLogLikelihood = tempLogLikelihood1 + weight1 + tempLogLikelihood2
				+ weight2;

		return tempLogLikelihood;
	}

	
	//p(w)=\sum_z p(w|z)p(z|d)
	

}
