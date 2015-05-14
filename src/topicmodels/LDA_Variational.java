/**
 * 
 */
package topicmodels;

import java.util.Arrays;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;

/**
 * @author hongning
 * Variational sampling for Latent Dirichlet Allocation model
 * Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent dirichlet allocation." 
 */
public class LDA_Variational extends pLSA {

	// parameters to control variational inference
	int m_varMaxIter;
	double m_varConverge;
	
	public LDA_Variational(int number_of_iteration, double converge,
			double beta, _Corpus c, double lambda, double[] back_ground,
			int number_of_topics, double alpha, int varMaxIter, double varConverge) {
		super(number_of_iteration, converge, beta, c, lambda, back_ground, number_of_topics, alpha);
		m_varConverge = varConverge;
		m_varMaxIter = varMaxIter;
	}
	
	@Override
	public String toString() {
		return String.format("LDA[k:%d, alpha:%.2f, beta:%.2f, Variational]", number_of_topics, d_alpha, d_beta);
	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection) {
		// initialize with all smoothing terms
		init();
		
		// initialize topic-word allocation, p(w|z)
		for(_Doc d:collection) {
			d.setTopics4Variational(number_of_topics, d_alpha);//allocate memory and randomize it
			collectStats(d);
		}
		
		calculate_M_step(0);
	}
	
	void collectStats(_Doc d) {
		_SparseFeature[] fv = d.getSparse();
		int wid;
		double v; 
		for(int n=0; n<fv.length; n++) {
			wid = fv[n].getIndex();
			v = fv[n].getValue();
			for(int i=0; i<number_of_topics; i++) {
				word_topic_sstat[i][wid] += v*d.m_phi[n][i];
				m_sstat[i] += v*d.m_phi[n][i];
			}
		}
	}
	
	@Override
	protected void init() {//will be called at the beginning of each EM iteration
		// initialize with all smoothing terms
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		Arrays.fill(m_sstat, d_beta*vocabulary_size);
	}

	@Override
	protected void initTestDoc(_Doc d) {
		//this needs to be carefully implemented
	}
	
	@Override
	public double calculate_E_step(_Doc d) {	
		double last = calculate_log_likelihood(d), current, converge, logSum, v;
		int iter = 0, wid;
		_SparseFeature[] fv = d.getSparse();
		
		do {
			//variational inference for p(z|w,\phi)
			for(int n=0; n<fv.length; n++) {
				wid = fv[n].getIndex();
				for(int i=0; i<number_of_topics; i++)
					d.m_phi[n][i] = topic_term_probabilty[i][wid] + Utils.digamma(d.m_sstat[i]);
				
				logSum = Utils.logSumOfExponentials(d.m_phi[n]);
				for(int i=0; i<number_of_topics; i++)
					d.m_phi[n][i] = Math.exp(d.m_phi[n][i] - logSum);
			}
			
			//variational inference for p(\theta|\gamma)
			Arrays.fill(d.m_sstat, d_alpha);
			for(int n=0; n<fv.length; n++) {
				v = fv[n].getValue();
				for(int i=0; i<number_of_topics; i++)
					d.m_sstat[i] += d.m_phi[n][i] * v;// 
			}
			
			current = calculate_log_likelihood(d);			
			converge = Math.abs((current - last)/last);
			last = current;
		} while(++iter<m_varMaxIter && converge>m_varConverge);
		
		//collect the sufficient statistics after convergence
		collectStats(d);
		
		return current;
	}
	
	@Override
	public void calculate_M_step(int iter) {	
		//maximum likelihood estimation of p(w|z,\beta)
		for(int i=0; i<number_of_topics; i++) {
			for(int v=0; v<vocabulary_size; v++) {//will be in the log scale!!
				topic_term_probabilty[i][v] = Math.log(word_topic_sstat[i][v]/m_sstat[i]);
			}
		}
		
		//we need to estimate p(\theta|\alpha) as well later on
	}
	
	@Override
	protected void finalEst() {	
		//estimate p(z|d) from all the collected samples
		for(_Doc d:m_trainSet) 
			estThetaInDoc(d);
	}
	
	@Override
	public double calculate_log_likelihood(_Doc d) {
		int wid;
		double logLikelihood = -Utils.lgamma(Utils.sumOfArray(d.m_sstat)), v;
		for(int i=0; i<number_of_topics; i++)
			logLikelihood += Utils.lgamma(d.m_sstat[i]);
		
		//collect the sufficient statistics
		_SparseFeature[] fv = d.getSparse();
		for(int n=0; n<fv.length; n++) {
			wid = fv[n].getIndex();
			v = fv[n].getValue();
			for(int i=0; i<number_of_topics; i++) 
				logLikelihood += v * d.m_phi[n][i] * (topic_term_probabilty[i][wid] - Math.log(d.m_phi[n][i]));
		}
		return logLikelihood;
	}
	
	@Override
	protected double calculate_log_likelihood() {
		return 0;//right now we are not estimating \alpha yet
	}
}
