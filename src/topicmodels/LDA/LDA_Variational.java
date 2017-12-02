/**
 * 
 */
package topicmodels.LDA;

import java.util.Arrays;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import topicmodels.pLSA.pLSA;
import utils.Utils;

/**
 * @author hongning
 * Variational sampling for Latent Dirichlet Allocation model
 * Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent dirichlet allocation." 
 */
public class LDA_Variational extends pLSA {

	// parameters to control variational inference
	protected int m_varMaxIter;
	protected double m_varConverge;
	
	protected double[] m_alpha; // we can estimate a vector of alphas as in p(\theta|\alpha)
	protected double[] m_alphaStat; // statistics for alpha estimation
	double[] m_alphaG; // gradient for alpha
	double[] m_alphaH; // Hessian for alpha
	
	public LDA_Variational(int number_of_iteration, double converge,
			double beta, _Corpus c, double lambda, 
			int number_of_topics, double alpha, int varMaxIter, double varConverge) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha);
		
		m_varConverge = varConverge;
		m_varMaxIter = varMaxIter;		
		
		m_logSpace = true;
	}
	
	@Override
	protected void createSpace() {
		super.createSpace();
		
		m_alpha = new double[number_of_topics];
		m_alphaStat = new double[number_of_topics];
		m_alphaG = new double[number_of_topics];
		m_alphaH = new double[number_of_topics];
		
		Arrays.fill(m_alpha, d_alpha);
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
	
	@Override
	protected void init() {//will be called at the beginning of each EM iteration
		// initialize alpha statistics
		Arrays.fill(m_alphaStat, 0);
		
		// initialize with all smoothing terms
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta-1.0);
		imposePrior();
	}
	
	protected void collectStats(_Doc d) {
		_SparseFeature[] fv = d.getSparse();
		int wid;
		double v; 
		for(int n=0; n<fv.length; n++) {
			wid = fv[n].getIndex();
			v = fv[n].getValue();
			for(int i=0; i<number_of_topics; i++)
				word_topic_sstat[i][wid] += v*d.m_phi[n][i];
		}
		
		//if we need to use maximum likelihood to estimate alpha
		double diGammaSum = Utils.digamma(Utils.sumOfArray(d.m_sstat));
		for(int i=0; i<number_of_topics; i++)
			m_alphaStat[i] += Utils.digamma(d.m_sstat[i]) - diGammaSum;
	}

	@Override
	protected void initTestDoc(_Doc d) {
		d.setTopics4Variational(number_of_topics, d_alpha);
	}
	
	@Override
	public double calculate_E_step(_Doc d) {	
		double last = 1;		
		if (m_varConverge>0)
			last = calculate_log_likelihood(d);
		
		double current = last, converge, logSum, v;
		int iter = 0, wid;
		_SparseFeature[] fv = d.getSparse();
		
		do {
			//variational inference for p(z|w,\phi)
			for(int n=0; n<fv.length; n++) {
				wid = fv[n].getIndex();
				v = fv[n].getValue();
				for(int i=0; i<number_of_topics; i++)
					d.m_phi[n][i] = topic_term_probabilty[i][wid] + Utils.digamma(d.m_sstat[i]);
				
				logSum = Utils.logSumOfExponentials(d.m_phi[n]);
				for(int i=0; i<number_of_topics; i++)
					d.m_phi[n][i] = Math.exp(d.m_phi[n][i] - logSum);
			}
			
			//variational inference for p(\theta|\gamma)
			System.arraycopy(m_alpha, 0, d.m_sstat, 0, m_alpha.length);
			for(int n=0; n<fv.length; n++) {
				v = fv[n].getValue();
				for(int i=0; i<number_of_topics; i++)
					d.m_sstat[i] += d.m_phi[n][i] * v;
			}
			
			if (m_varConverge>0) {
				current = calculate_log_likelihood(d);			
				converge = Math.abs((current - last)/last);
				last = current;
				
				if (converge<m_varConverge)
					break;
			}
		} while(++iter<m_varMaxIter);		
		
		if (m_collectCorpusStats) {
			collectStats(d);//collect the sufficient statistics after convergence
		 	return current;
		} else if (m_varConverge>0)
			return current;//to avoid computing this again
		else
			return calculate_log_likelihood(d);//in testing, we need to compute log-likelihood
	}
	
	@Override
	public void calculate_M_step(int iter) {	
		//maximum likelihood estimation of p(w|z,\beta)
		for(int i=0; i<number_of_topics; i++) {
			double sum = Utils.sumOfArray(word_topic_sstat[i]);
			for(int v=0; v<vocabulary_size; v++) //will be in the log scale!!
				topic_term_probabilty[i][v] = Math.log(word_topic_sstat[i][v]/sum);
		}
		
		if (iter%5!=4)//no need to estimate \alpha very often
			return;
		
		//we need to estimate p(\theta|\alpha) as well later on
		int docSize = m_trainSet.size(), i = 0;
		double alphaSum, diAlphaSum, z, c, c1, c2, diff, deltaAlpha;
		do {
			alphaSum = Utils.sumOfArray(m_alpha);
			diAlphaSum = Utils.digamma(alphaSum);
			z = docSize * Utils.trigamma(alphaSum);
			
			c1 = 0; c2 = 0;
			for(int k=0; k<number_of_topics; k++) {
				m_alphaG[k] = docSize * (diAlphaSum - Utils.digamma(m_alpha[k])) + m_alphaStat[k];
				m_alphaH[k] = -docSize * Utils.trigamma(m_alpha[k]);
				
				c1 +=  m_alphaG[k] / m_alphaH[k];
				c2 += 1.0 / m_alphaH[k];
			}			
			c = c1 / (1.0/z + c2);
			
			diff = 0;
			for(int k=0; k<number_of_topics; k++) {
				deltaAlpha = (m_alphaG[k]-c) / m_alphaH[k];
				m_alpha[k] -= deltaAlpha;
				diff += deltaAlpha * deltaAlpha;
			}
			diff /= number_of_topics;
		} while(++i<m_varMaxIter && diff>m_varConverge);
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
		double[] diGamma = new double[this.number_of_topics];
		double logLikelihood = Utils.lgamma(Utils.sumOfArray(m_alpha)) - Utils.lgamma(Utils.sumOfArray(d.m_sstat)), v, diGammaSum = Utils.digamma(Utils.sumOfArray(d.m_sstat));
		for(int i=0; i<number_of_topics; i++) {
			diGamma[i] = Utils.digamma(d.m_sstat[i]) - diGammaSum;
			logLikelihood += Utils.lgamma(d.m_sstat[i]) - Utils.lgamma(m_alpha[i])
					+ (m_alpha[i] - d.m_sstat[i]) * diGamma[i];
		}
		
		//collect the sufficient statistics
		_SparseFeature[] fv = d.getSparse();
		for(int n=0; n<fv.length; n++) {
			wid = fv[n].getIndex();
			v = fv[n].getValue();
			for(int i=0; i<number_of_topics; i++) 
				logLikelihood += d.m_phi[n][i] * (diGamma[i] + v*topic_term_probabilty[i][wid] - Math.log(d.m_phi[n][i]));
		}

		return logLikelihood;
	}
	
	// perform inference of topic distribution in the document
	@Override
	public double inference(_Doc d) {
		initTestDoc(d);		
		double likelihood = calculate_E_step(d);
		estThetaInDoc(d);
		return likelihood;
	}
}
