package topicmodels.correspondenceModels;

import java.util.Collection;

import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc4DCM;

/****
 * comments of an article share a topic proportion
 * each comment has a single topic. all words in a comment are assigned the same topic
 * ****/

public class DCMDMMCorrLDA extends DCMCorrLDA{
	public DCMDMMCorrLDA(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha_a,
			double alpha_c, double burnIn, double ksi, double tau, int lag,
			int newtonIter, double newtonConverge){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha_a, alpha_c, burnIn, ksi, tau, lag, newtonIter, newtonConverge);
	}
	
	public String toString(){
		return String.format("DCMDMMCorrLDA[k:%d, alphaA:%.2f, beta:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta);
	}
	
	protected double parentChildInfluenceProb(int tid, _ParentDoc4DCM d){
		double term = 1.0;
		
		if(tid==0)
			return term;
		
		int tidNum = 0;
		int zeroNum = 0;
		
		double mu = 0;
		for(_ChildDoc cDoc:d.m_childDocs){
			mu = cDoc.getMu();
			tidNum += (int)cDoc.m_sstat[tid];
			zeroNum += (int)cDoc.m_sstat[0];
		}
		
		double muDp = mu/d.getDocInferLength();
		term *= 
		
	}
	
}
