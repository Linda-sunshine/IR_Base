package topicmodels;

import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collection;

import structures.MyPriorityQueue;
import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._RankItem;
import utils.Utils;

public class ParentChild_Gibbs extends LDA_Gibbs{
	public double[][] m_parentWordTopicSstat;
	public double[][] m_childWordTopicSstat;
	
	public double[] m_parentSstat;
	public double[] m_childSstat;
	
	public double[][] m_parentTopicTermProb;
	public double[][] m_childTopicTermProb;
	
	public double[] m_gamma;
	protected int m_indicatorNum;
	
	public ParentChild_Gibbs(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, int indicatorNum) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag);
		
		m_indicatorNum = indicatorNum;
		m_gamma = new double[m_indicatorNum];
		for(int i=0; i<m_indicatorNum; i++){
			m_gamma[i] = gamma[i];
		}
		//converge = 0
		
		// TODO Auto-generated constructor stub
	}
	
	@Override
	protected void createSpace(){
		super.createSpace();
		
		//sufficient statistics the number of each word assigned to a topic in parent documents
		m_parentWordTopicSstat = new double[number_of_topics][vocabulary_size];
		//sufficient statistics the number of each word assigned to a topic in child documents
		m_childWordTopicSstat = new double[number_of_topics][vocabulary_size];
		
		// store the number of words of each topic in parent document
		m_parentSstat = new double[number_of_topics];
		//store the number of words in each topic
		m_childSstat = new double[number_of_topics];
		
		//the probability of a word assigned to a topic  in 
		m_parentTopicTermProb = new double[number_of_topics][vocabulary_size];
		m_childTopicTermProb = new double[number_of_topics][vocabulary_size];
	}
	
	@Override
	public String toString(){
		return String.format("Parent Child topic model [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta, m_gamma[1], m_gamma[2]);
	}
	
//	@Override
	protected void initialize_probability(Collection<_Doc> collection){
//		for(int i=0; i<number_of_topics; i++)
//			Arrays.fill(word_topic_sstat[i], d_beta);
//		Arrays.fill(m_sstat, d_beta*vocabulary_size);
		
		for(int i=0; i<number_of_topics; i++){
			Arrays.fill(m_parentWordTopicSstat[i], 0);
			Arrays.fill(m_childWordTopicSstat[i], 0);
		}
		Arrays.fill(m_parentSstat, 0);
		Arrays.fill(m_childSstat, 0);
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc){
				((_ParentDoc) d).setTopics4Gibbs(number_of_topics);
				for (int i = 0; i < d.m_words.length; i++) {
					m_parentWordTopicSstat[d.m_topicAssignment[i]][d.m_words[i]]++;
					m_parentSstat[d.m_topicAssignment[i]]++;
				}
			}else{
				if(d instanceof _ChildDoc){
					((_ChildDoc) d).setTopics4Gibbs(number_of_topics, d_alpha, m_indicatorNum, m_gamma);
					for(int i=0; i<d.m_words.length; i++){
						m_childWordTopicSstat[d.m_topicAssignment[i]][d.m_words[i]] ++;
						m_childSstat[d.m_topicAssignment[i]]++;
								
					}
				}
			}
			

		}
		
		imposePrior();
	}
	
	public double calculate_E_step(_Doc d){
		d.permutation();
		
		sampleTopic(d);
		
		return 1;
	}
	
	public void sampleTopic(_Doc d){
		
		if(d instanceof _ParentDoc){
			sampleParentDocTopic((_ParentDoc)d);

		}else{
			if(d instanceof _ChildDoc){
				sampleChildDocTopic((_ChildDoc)d);
			}
		}
		
		//return samplingTopics;
	}
	
	public void sampleParentDocTopic(_ParentDoc d){
		int samplingTopic = 0;
		int wid, tid;
		double[] topicProb = new double[number_of_topics];
		double prob;
		double normalizedProb;
		
		for(int i=0; i<d.m_words.length; i++){
			normalizedProb = 0;
			prob = 0;
			wid = d.m_words[i];
			tid = d.m_topicAssignment[i];
			
			d.m_sstat[tid] --;
			if(m_collectCorpusStats){
				m_parentWordTopicSstat[tid][wid] --;
				m_parentSstat[tid] --;
			}
			
			for(tid=0; tid<number_of_topics; tid++){
				topicProb[tid] = 0;
				double term1 = (m_parentWordTopicSstat[tid][wid]+d_beta)/(m_parentSstat[tid]+vocabulary_size*d_beta);
				double term2 = (d.m_sstat[tid]+d_alpha);
				double term3 = 1;

				// System.out.println("parent" + d.getName());

				for(_ChildDoc cDoc: d.m_childDocs){
					// System.out.println("child" + cDoc.getName());

					term3 *= (d.m_sstat[tid]+cDoc.m_xTopicSstat[0][tid]+d_alpha)/(d.m_sstat[tid]+d_alpha);
				}
					
				topicProb[tid] = term1*term2*term3;
				normalizedProb += topicProb[tid];
//				if(tid>0){
//					topicProb[tid] += topicProb[tid-1];
//				}
			}
			
			prob = normalizedProb*m_rand.nextDouble();
			
			for(tid=0; tid<number_of_topics; tid++){
				prob -= topicProb[tid];
				if(prob<=0){
					break;
				}
			}
			if(tid==number_of_topics)
				tid --;
			samplingTopic = tid;
			
			d.m_topicAssignment[i] = samplingTopic;
			d.m_sstat[samplingTopic] ++;
			if(m_collectCorpusStats){
				m_parentWordTopicSstat[samplingTopic][wid] ++;
				m_parentSstat[samplingTopic] ++;
			}
		}
	}

	public void sampleChildDocTopic(_ChildDoc d){
		int wid, tid, xid;
		
		double[][] xTopicProb = new double[2][number_of_topics];
		double prob;
		double normalizedProb = 0;
		
		for(int i=0; i<d.m_words.length; i++){
			int samplingX = 0;
			int samplingTopic = 0;
			prob = 0;
			normalizedProb = 0;
			
			wid = d.m_words[i];
			tid = d.m_topicAssignment[i];
			xid = d.m_xIndicator[i];
			
			d.m_xTopicSstat[xid][tid] --;
			d.m_xSstat[xid] --;
			if(m_collectCorpusStats){
				m_childWordTopicSstat[tid][wid] --;
				m_childSstat[tid] --;
			}
			
			//p(z=tid,x=1) from specific
			for(tid=0; tid<number_of_topics; tid++){
				double term1 = (m_childWordTopicSstat[tid][wid]+d_beta)/(m_childSstat[tid]+d_beta*vocabulary_size);
				double term2 = (d.m_xTopicSstat[1][tid]+d_alpha)/(number_of_topics*d_alpha+d.m_xSstat[1]);
//				double term3 = (m_gamma[1]+d.m_xSstat[0])/(m_gamma[1]+m_gamma[2]+d.m_xSstat[0]+d.m_xSstat[1]);
				double term3 = (m_gamma[1]+d.m_xSstat[1]);
				xTopicProb[1][tid] = term1*term2*term3;
				normalizedProb += xTopicProb[1][tid];
			}
			
			if (d.m_parentDoc == null) {
				System.out.println("null parent in child doc" + d.getName());
			}
			
			//p(z=tid x=0) from background
			for(tid=0; tid<number_of_topics; tid++){
				double term1 = (m_childWordTopicSstat[tid][wid]+d_beta)/(m_childSstat[tid]+d_beta*vocabulary_size);
				double term2 = (d_alpha+d.m_parentDoc.m_sstat[tid]+d.m_xTopicSstat[0][tid])/(number_of_topics*d_alpha+d.m_parentDoc.getTotalDocLength()+d.m_xSstat[0]);
				double term3 = (m_gamma[0]+d.m_xSstat[0]);
				xTopicProb[0][tid] = term1*term2*term3;
				normalizedProb += xTopicProb[0][tid];
			}
			
			boolean finishLoop = false;
			prob = normalizedProb*m_rand.nextDouble();
			for(xid=0; xid<m_indicatorNum; xid++){
				for(tid=0; tid<number_of_topics; tid++){
					prob -= xTopicProb[xid][tid];
					if(prob<=0){
						finishLoop = true;
						break;
					}
				}
				if (finishLoop) {
					break;
				}
			}
			

			if((xid==2)&&(tid==number_of_topics)){
				xid--;
				tid--;
			}
			
			samplingX = xid;
			samplingTopic = tid;
			
			d.m_topicAssignment[i] = samplingTopic;
			d.m_xIndicator[i] = samplingX;

			d.m_xTopicSstat[samplingX][samplingTopic] ++;
			d.m_xSstat[samplingX] ++;
			if(m_collectCorpusStats){
				m_childWordTopicSstat[samplingTopic][wid] ++;
				m_childSstat[samplingTopic] ++;
			}
			
		}
	}

	public void calculate_M_step(int iter){
		if(iter>m_burnIn && iter%m_lag==0){
			for(int i=0; i<this.number_of_topics; i++){
				for(int v=0; v<this.vocabulary_size; v++){
					m_parentTopicTermProb[i][v] += (m_parentWordTopicSstat[i][v]+d_beta);
					m_childTopicTermProb[i][v] += (m_childWordTopicSstat[i][v]+d_beta);
				}
			}
			
			for(_Doc d:m_trainSet){
				if(d instanceof _ParentDoc)
					collectParentStats((_ParentDoc)d);
				else{
					if(d instanceof _ChildDoc){
						collectChildStats((_ChildDoc)d);
					}
				}
					
			}
		}
	}
	
	protected void collectParentStats(_ParentDoc d){
		for(int k=0; k<this.number_of_topics; k++){
			d.m_topics[k] += (d.m_sstat[k]+d_alpha);
		}
	}
	
	protected void collectChildStats(_ChildDoc d){
		for(int k=0; k<this.number_of_topics; k++){
			for(int j=0; j<m_indicatorNum; j++){
				d.m_topics[k] += (d.m_xTopicSstat[j][k]);
			}
		}
	}
	
	protected void finalEst() {
		for (int i = 0; i < this.number_of_topics; i++) {
			Utils.L1Normalization(m_parentTopicTermProb[i]);
			Utils.L1Normalization(m_childTopicTermProb[i]);
		}
		
		for (_Doc d : m_trainSet) {
			estThetaInDoc(d);
		}
	}
	
	protected void estThetaIndoc(_Doc d) {
		Utils.L1Normalization(d.m_topics);
	}
	
	public void printTopWords(int k, String betaFile) {
		Arrays.fill(m_parentSstat, 0);
		Arrays.fill(m_childSstat, 0);

		System.out.println("print top words");
		for (_Doc d : m_trainSet) {
			if (d instanceof _ParentDoc) {
				for (int i = 0; i < number_of_topics; i++)
					m_parentSstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
							: d.m_topics[i];
			} else if (d instanceof _ChildDoc) {
				for (int i = 0; i < number_of_topics; i++)
					m_childSstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
							: d.m_topics[i];
			}

		}

		Utils.L1Normalization(m_parentSstat);
		Utils.L1Normalization(m_childSstat);
		
		String parentBetaFile = betaFile.replace(".txt", "parent.txt");
		String childBetaFile = betaFile.replace(".txt", "child.txt");
		
		printParentTopWords(k, parentBetaFile);
		printChildTopWords(k, childBetaFile);

	}

	public void printParentTopWords(int k, String parentBetaFile) {

		try {
			System.out.println("parent beta file");
			PrintWriter parentBeta = new PrintWriter(new File(parentBetaFile));
			for (int i = 0; i < m_parentTopicTermProb.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j),
							m_parentTopicTermProb[i][j]));

				parentBeta.format("Topic %d(%.3f):\t", i, m_parentSstat[i]);
				for (_RankItem it : fVector) {
					parentBeta.format("%s(%.3f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
					System.out.format("%s(%.3f)\t", it.m_name,
						m_logSpace ? Math.exp(it.m_value) : it.m_value);
				}
				parentBeta.println();
				System.out.println();
			}
	
			parentBeta.flush();
			parentBeta.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}
	
	public void printChildTopWords(int k, String childBetaFile) {
		try {
			System.out.println("child beta file");
			PrintWriter childBeta = new PrintWriter(new File(childBetaFile));

			for (int i = 0; i < m_childTopicTermProb.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j),
							m_childTopicTermProb[i][j]));

				childBeta.format("Topic %d(%.3f):\t", i, m_childSstat[i]);
				System.out.format("Topic %d(%.3f):\t", i, m_childSstat[i]);
				for (_RankItem it : fVector) {
					childBeta.format("%s(%.3f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
					System.out.format("%s(%.3f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
				}
				childBeta.println();
				System.out.println();
			}

			childBeta.flush();
			childBeta.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}

}
