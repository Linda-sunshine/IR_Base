package topicmodels;

import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collection;

import structures.MyPriorityQueue;
import structures._ChildDoc;
import structures._ChildDoc4BaseWithPhi;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._RankItem;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class ACCTM_CZGlobalPhi extends ACCTM_CZ{
	protected double[][] m_childWordTopicProb;
	protected double[][] m_childWordTopicSstat;
	protected double[] m_childSstat;
	protected double m_childBeta;
	protected int m_childNumofTopic;
	
	public ACCTM_CZGlobalPhi(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double ksi, double tau){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma, ksi, tau);
	}
	
	protected void initialize_probability(Collection<_Doc> collection){
		createSpace();
		
		m_childNumofTopic = 1;
		m_childWordTopicProb = new double[m_childNumofTopic][vocabulary_size];
		m_childWordTopicSstat = new double[m_childNumofTopic][vocabulary_size];
		
		m_childSstat = new double[m_childNumofTopic];
		
		m_childBeta = d_beta*0.01;
		
		for(int i=0; i<number_of_topics; i++){
			Arrays.fill(word_topic_sstat[i], d_beta);
		}
		
		for(int i=0; i<m_childNumofTopic; i++)
			Arrays.fill(m_childWordTopicSstat[i], m_childBeta);

		Arrays.fill(m_sstat, d_beta*vocabulary_size);
		Arrays.fill(m_childSstat, m_childBeta*vocabulary_size);
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc){
				d.setTopics4Gibbs(number_of_topics, 0);
				for(_Stn stnObj: d.getSentences())
					stnObj.setTopicsVct(number_of_topics);
			}else if(d instanceof _ChildDoc4BaseWithPhi){
				((_ChildDoc4BaseWithPhi) d).createXSpace(number_of_topics, m_gamma.length, vocabulary_size, d_beta);
				((_ChildDoc4BaseWithPhi) d).setTopics4Gibbs(number_of_topics, 0);
				computeMu4Doc((_ChildDoc) d);
			}
			
			if(d instanceof _ParentDoc){
				for (_Word w:d.getWords()) {
					word_topic_sstat[w.getTopic()][w.getIndex()]++;
					m_sstat[w.getTopic()]++;
				}	
			}else if(d instanceof _ChildDoc4BaseWithPhi){
				for(_Word w: d.getWords()){
					int xid = w.getX();
					int tid = w.getTopic();
					int wid = w.getIndex();
					//update global
					if(xid==0){
						word_topic_sstat[tid][wid] ++;
						m_sstat[tid] ++;
					}else{
						m_childWordTopicSstat[0][wid] ++;
						m_childSstat[0] ++;
					}
				}
			}
		}
		
		imposePrior();	
		m_statisticsNormalized = false;

	}
	
	public String toString(){
		return String.format("ACCTM_CZGlobalPhi topic model [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta, m_gamma[0], m_gamma[1]);
	}
	
	protected void sampleInChildDoc(_ChildDoc d){
		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi) d;
		int wid, tid, xid;
		double normalizedProb;
		
		for(_Word w:cDoc.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			xid = w.getX();
			
			if(xid==0){
				cDoc.m_xTopicSstat[xid][tid] --;
				cDoc.m_xSstat[xid] --;
				cDoc.m_wordXStat.put(wid, cDoc.m_wordXStat.get(wid)-1);
				if(m_collectCorpusStats){
					word_topic_sstat[tid][wid] --;
					m_sstat[tid] --;
				}
			}else if(xid==1){
				cDoc.m_xTopicSstat[xid][wid] --;
				cDoc.m_xSstat[xid] --;
				if(m_collectCorpusStats){
					m_childWordTopicSstat[0][wid] --;
					m_childSstat[0] --;
				}
			}
			
			normalizedProb = 0;
			double pLambdaZero = childXInDocProb(0, cDoc);;
			double pLambdaOne = childXInDocProb(1, cDoc);
			
			for(tid=0; tid<number_of_topics; tid++){
				double pWordTopic = childWordByTopicProb(tid, wid);
				double pTopic = childTopicInDocProb(tid, cDoc);
				
				m_topicProbCache[tid] = pWordTopic*pTopic*pLambdaZero;
				normalizedProb += m_topicProbCache[tid];
			}
			
			double pWordTopic = localChildWordByTopicProb(0, wid);
			m_topicProbCache[tid] = pWordTopic*pLambdaOne;
			normalizedProb += m_topicProbCache[tid];
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<m_topicProbCache.length; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<=0)
					break;
			}
			
			if(tid==m_topicProbCache.length)
				tid --;
			
			if(tid<number_of_topics){
				xid = 0;
				w.setX(xid);
				w.setTopic(tid);
				
				cDoc.m_xTopicSstat[xid][tid] ++;
				cDoc.m_xSstat[xid] ++;
				
				if(cDoc.m_wordXStat.containsKey(wid)){
					cDoc.m_wordXStat.put(wid, cDoc.m_wordXStat.get(wid)+1);
				}else{
					cDoc.m_wordXStat.put(wid, 1);
				}
				
				if(m_collectCorpusStats){
					word_topic_sstat[tid][wid] ++;
					m_sstat[tid] ++;
 				}
			}else if(tid == number_of_topics){
				xid = 1;
				w.setX(xid);
				w.setTopic(tid);
				
				cDoc.m_xTopicSstat[xid][wid] ++;
				cDoc.m_xSstat[xid] ++;

				if(m_collectCorpusStats){
					m_childWordTopicSstat[0][wid] ++;
					m_childSstat[0] ++;
 				}
			}
		}
	}
	
	protected double localChildWordByTopicProb(int tid, int wid){
		return m_childWordTopicSstat[tid][wid] / m_childSstat[tid];
	}
	
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
			
			for(int i=0; i<m_childNumofTopic; i++){
				for(int v=0; v<vocabulary_size; v++){
					m_childWordTopicProb[i][v] += m_childWordTopicSstat[i][v];
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
	
	protected void printChildWordTopicProb(int k, String filePrefix){
		String childWordTopicProbFile = filePrefix+"childTopWords.txt";
		
		try{
			PrintWriter pw = new PrintWriter(new File(childWordTopicProbFile));
			for(int i=0; i<m_childWordTopicProb.length; i++){
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
				for(int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j), m_childWordTopicProb[i][j]));
				
				pw.format("Topic %d:\t", i);
				for(_RankItem it:fVector)
					pw.format("%s(%.5f)\t", it.m_name, m_logSpace?Math.exp(it.m_value):it.m_value);
				pw.write("\n");
			}
			
			pw.flush();
			pw.close();
		}catch(Exception e){
			System.err.print("File Not Found");
		}
	}
	
	@Override
	protected void finalEst(){
		super.finalEst();
		for(int i=0; i<m_childNumofTopic; i++)
			Utils.L1Normalization(m_childWordTopicProb[i]);
	}
	
	public void debugOutput(String filePrefix){
		super.debugOutput(filePrefix);
		int topKWord = 20;
		printChildWordTopicProb(topKWord, filePrefix);
	}
}
