package topicmodels.correspondenceModels;

import java.util.Arrays;

import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4DCM;
import topicmodels.multithreads.TopicModel_worker;
import topicmodels.multithreads.multiEM_worker;
import topicmodels.multithreads.updateParam_worker.RunType;
import utils.Utils;

public class DCMCorrLDA_Multi_EM extends DCMCorrLDA{
	public class DCMCorrLDA_Multi_worker implements multiEM_worker{
		protected double[] alphaStat;
		double[] m_param;
		protected double m_paramIndex;
		protected double m_likelihood;
		
		public DCMCorrLDA_Multi_worker(int number_of_topics, int vocabulary_size){
			
			alphaStat = new double[number_of_topics];
		}

		@Override
		public void run() {
			System.out.println("runnig thread");
			
			for(int i=0; i<m_paramIndex.size(); i++){
				calculate_M_step(m_param.get(i), m_paramIndex.get(i));
			}
			
			for(int i=0; i<m_paramIndex.size(); i++){
				int paramLength = m_param.get(i).length;
				int paramIndex = m_paramIndex.get(i);
				System.arraycopy(m_param.get(i), 0, m_beta[paramIndex], 0, paramLength);
			}
		}

		@Override
		public void setType(RunType type) {
			m_type = type;
			
		}

		@Override
		public void addParameter(double[] t_param, int t_index) {
			int paramLen = t_param.length;
			double[] param = new double[paramLen];
			System.arraycopy(t_param, 0, param, 0, paramLen);
			m_param.add(param);
			m_paramIndex.add(t_index);
		}

		@Override
		public void clearParameter() {
			for(double[] param: m_param)
				Arrays.fill(param, 0);
		}

		@Override
		public void calculate_M_step() {
			System.out.println("topic optimization\t"+tid);
			double diff = 0;
			int iteration = 0;
			double smoothingBeta = 0.1;
			double totalBeta = 0;
			
			do{
				diff = 0;
				
				double deltaBeta = 0;
				double wordNum = 0;
				
				double[] wordNum4V = new double[vocabulary_size];
				double totalBetaDenominator = 0;
				double[] totalBetaNumerator = new double[vocabulary_size];
				
				Arrays.fill(totalBetaNumerator, 0);
				Arrays.fill(wordNum4V, 0);
				
				totalBeta = Utils.sumOfArray(param);
				double digBeta = Utils.digamma(totalBeta);
				
				for(_Doc d:m_trainSet){
					if(d instanceof _ParentDoc){
						_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
						totalBetaDenominator += Utils.digamma(totalBeta+pDoc.m_topic_stat[tid])-digBeta;
						for(int v=0; v<vocabulary_size; v++){
							wordNum += pDoc.m_wordTopic_stat[tid][v];
							wordNum4V[v] += pDoc.m_wordTopic_stat[tid][v];
							
							totalBetaNumerator[v] += Utils.digamma(param[v]+pDoc.m_wordTopic_stat[tid][v]);
							totalBetaNumerator[v] -= Utils.digamma(param[v]);
						}
					}
					
				}
				
				for(int v=0; v<vocabulary_size; v++){
					if(wordNum == 0)
						break;
					if(wordNum4V[v] == 0){
						deltaBeta = 0;
					}else{
						deltaBeta = totalBetaNumerator[v]/totalBetaDenominator;
					}
						
					double newBeta = param[v]*deltaBeta+d_beta;
					double t_diff = Math.abs(param[v]-newBeta);
					if(t_diff>diff)
						diff = t_diff;
					
					param[v] = newBeta;
				}
				
				iteration ++;
				if(iteration > m_newtonIter)
					break;
				
				System.out.println("beta iteration\t"+iteration);
			}while(diff > m_newtonConverge);
			
			System.out.println("iteration\t"+iteration);
		}

		@Override
		public void returnParameter(double[] param, int index) {
			if(param.length == m_param.get(index).length){
				System.arraycopy(param, 0, m_param.get(index), 0, param.length);
			}
		}

		@Override
		public double getLogLikelihood() {
			return m_likelihood;

		}

		@Override
		public void addDoc(_Doc d) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void clearCorpus() {
			// TODO Auto-generated method stub
			
		}

		@Override
		public double calculate_E_step(_Doc d) {
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public double inference(_Doc d) {
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public double accumluateStats(double[][] word_topic_sstat) {
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public void resetStats() {
			// TODO Auto-generated method stub
			
		}

		@Override
		public double getPerplexity() {
			// TODO Auto-generated method stub
			return 0;
		}
		
		
	}
}
