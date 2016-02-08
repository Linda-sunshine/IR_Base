package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Random;

import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;
import structures._ChildDoc;
import structures._ChildDoc4ProbitModel;
import structures._Corpus;
import structures._Doc;
import structures._Word;
import utils.Utils;

public class ParentChildWithProbitModel_Gibbs extends ParentChild_Gibbs {
	
	Normal m_normal;
	Random m_Random;
	
	public ParentChildWithProbitModel_Gibbs(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double mu){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma, mu);
		m_normal = new Normal(0, 1, new DoubleMersenneTwister());
		m_Random = new Random();
	}
	
	public String toString(){
		return String.format("Parent Child topic model with probit model [k:%d, alpha:%.4f, beta:%.4f, Gibbs Sampling]",
				number_of_topics, d_alpha, d_beta);
	}
	
	@Override
	void sampleInChildDoc(_ChildDoc doc){
		_ChildDoc4ProbitModel d = (_ChildDoc4ProbitModel)doc;
		
		int wid, tid, xid;
		double normalizedProb;
		
		//sampling the indicator variable 
		_Word[] words = d.getWords();
		for(int i=0; i<words.length; i++){
			_Word w = words[i];
			xid = w.getX();
			tid = w.getTopic();	
			
			d.m_xSstat[xid] --;
			d.m_xTopicSstat[xid][tid] --;

			w.setXValue(xPredictiveProb(d, i));
			
			xid = w.getX();
			
			d.m_xSstat[xid] ++;
			d.m_xTopicSstat[xid][tid] ++;			
		}
		
		for(_Word w:words){
			wid = w.getIndex();
			tid = w.getTopic();
			xid = w.getX();
			
			d.m_xTopicSstat[xid][tid] --;
			d.m_xSstat[xid] --;			
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}
			
			normalizedProb = 0.0;			
			for(tid=0; tid<number_of_topics; tid++){
				m_topicProbCache[tid] = childWordByTopicProb(tid, wid) * childTopicInDocProb(tid, xid, d);		
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();			
			for(tid=0; tid<number_of_topics; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<=0)
					break;
			}

			if (tid == number_of_topics)
				tid--;
			
			w.setTopic(tid);
		
			d.m_xTopicSstat[xid][tid] ++;
			d.m_xSstat[xid] ++;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid]++;
				m_sstat[tid]++;
			}			
		}
	}
	
	//reject sampling for 
	double xPredictiveProb(_ChildDoc4ProbitModel d, int i){
		double mu = 0.0, sigma = 0.0, x;
		_Word[] words = d.getWords();
		
		int wid = words[i].getIndex(), tid = words[i].getTopic();
		double[] muVct = d.m_fixedMuPartMap.get(wid);
		for(int n=0; n<words.length; n++){
			if(n==i)
				continue;			
			
			mu += words[n].getXValue() * muVct[words[n].getLocalIndex()];			
		}
		
		sigma = d.m_fixedSigmaPartMap.get(wid);

		
		double x0 = childTopicInDocProb(tid, 0, d), x1 = childTopicInDocProb(tid, 1, d);//no need to repeatedly compute this
		do {
			x = m_normal.nextDouble(mu, sigma);

			if (x<=0 && Math.random()<x0)
				break;
			else if (x>0 && Math.random()<x1)
				break;
		} while (true);
		return x;
		
	}
	
	@Override
	protected void printChildXValue(_Doc d, File childXFolder){
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
			
			pw.println();
			for(int i=0; i<((_ChildDoc4ProbitModel)d).m_lambda.length; i++){
				pw.print(((_ChildDoc4ProbitModel)d).m_lambda[i]+"\t");
			}
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
