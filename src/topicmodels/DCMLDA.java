package topicmodels;

import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collection;

import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._Word;
import sun.security.util.PropertyExpander.ExpandException;
import utils.Utils;

public class DCMLDA extends LDA_Gibbs {

	/*
	 * m_docWordTopicProb---D*V*K D: number of documents V: number of words K:
	 * number of topics
	 * 
	 * m_docWordTopicStats---D*V*K m_docTopicStats---D*K // this can be included
	 * in the d.m_sstat
	 **/

	double[][][] m_docWordTopicProb;
	double[][][] m_docWordTopicStats;
//	double[][] m_docTopicStats;

	/**
	 * 
	 * m_alpha K m_beta K*V;
	 * 
	 */
	double[] m_alpha;
	double[][] m_beta;

	double m_totalAlpha;
	double[] m_totalBeta;
	
	int m_newtonIter;
	double m_newtonConverge;

	public DCMLDA(int number_of_iteration, double converge, double beta, _Corpus c, double lambda, int number_of_topics,
			double alpha, double burnIn, int lag, int newtonIter, double newtonConverge) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag);

		int corpusSize = c.getSize();
		m_docWordTopicProb = new double[corpusSize][number_of_topics][vocabulary_size];
		m_docWordTopicStats = new double[corpusSize][number_of_topics][vocabulary_size];
//		m_docTopicStats = new double[corpusSize][number_of_topics];

		m_alpha = new double[number_of_topics];
		m_beta = new double[number_of_topics][vocabulary_size];

		m_totalAlpha = 0;
		m_totalBeta = new double[number_of_topics];

		m_newtonIter = newtonIter;
		m_newtonConverge = newtonConverge;
	}

	public void EM() {
		System.out.format("Starting %s...\n", toString());

		long starttime = System.currentTimeMillis();

		m_collectCorpusStats = true;
		initialize_probability(m_trainSet);

		String filePrefix = "./data/results/DCM_LDA";
		File weightFolder = new File(filePrefix + "");
		if (!weightFolder.exists()) {
			// System.out.println("creating directory for weight"+weightFolder);
			weightFolder.mkdir();
		}

		double delta = 0, last = 0, current = 0;
		int i = 0, displayCount = 0;
		do {

			for (int j = 0; j < number_of_iteration; j++) {
				init();
				for (_Doc d : m_trainSet)
					calculate_E_step(d);
			}

			calculate_M_step(i, weightFolder);

			if (m_converge > 0 || (m_displayLap > 0 && i % m_displayLap == 0 && displayCount > 6)) {// required
																									// to
																									// display
																									// log-likelihood
				current = calculate_log_likelihood();// together with
														// corpus-level
														// log-likelihood

				if (i > 0)
					delta = (last - current) / last;
				else
					delta = 1.0;
				last = current;
			}

			if (m_displayLap > 0 && i % m_displayLap == 0) {
				if (m_converge > 0) {
					System.out.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);
					infoWriter.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);

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
		} while (++i < this.number_of_iteration);

		finalEst();

		long endtime = System.currentTimeMillis() - starttime;
		System.out.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta,
				endtime / 1000);
		infoWriter.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta,
				endtime / 1000);
	}

	protected void initialize_probability(Collection<_Doc> collection) {

		// initialize topic-word allocation, p(w|z)
		for (_Doc d : collection) {
			int docID = d.getID();
			for (int k = 0; k < number_of_topics; k++) {
				Arrays.fill(m_docWordTopicStats[docID][k], 0);
			}

			// Arrays.fill(m_docTopicStats[docID], d_beta*vocabulary_size);
			m_totalAlpha = 0;
			for (int k = 0; k < number_of_topics; k++) {
				m_alpha[k] = d_alpha;
				m_totalAlpha += m_alpha[k];
				Arrays.fill(m_beta[k], d_beta);
				m_totalBeta[k] = vocabulary_size * d_beta;
			}

			d.setTopics4Gibbs(number_of_topics, 0);// allocate memory and
													// randomize it
			for (_Word w : d.getWords()) {
				int wid = w.getIndex();
				int tid = w.getTopic();
				m_docWordTopicStats[docID][tid][wid]++;
//				m_docTopicStats[docID][tid]++;

			}

		}

		imposePrior();
	}

	@Override
	protected int sampleTopic4Word(_Word w, _Doc d) {
		double p;
		int tid = 0;
		int wid = 0;

		p = 0;
		for (tid = 0; tid < number_of_topics; tid++) {
			double term1 = topicInDocProb(tid, d);
			term1 = wordTopicProb(tid, wid, d);
			m_topicProbCache[tid] = topicInDocProb(tid, d) * wordTopicProb(tid, wid, d); // p(z|d)
																							// *
																							// p(w|z)
			p += m_topicProbCache[tid];
		}

		p *= m_rand.nextDouble();

		tid = -1;
		while (p > 0 && tid < number_of_topics - 1) {
			tid++;
			p -= m_topicProbCache[tid];
		}

		return tid;

	};

	protected double topicInDocProb(int tid, _Doc d) {
		return d.m_sstat[tid] + m_alpha[tid];
	}

	/*
	 * p(w|z)
	 */
	protected double wordTopicProb(int tid, int wid, _Doc d) {
		int docID = d.getID();
		return (m_docWordTopicStats[docID][tid][wid] + m_beta[tid][wid])
				/ (d.m_sstat[tid] + m_beta[tid][wid] * vocabulary_size);
	}

	protected void updateStats(boolean preFlag, _Word w, _Doc d) {
		int docID = d.getID();
		int wid = w.getIndex();
		int tid = w.getTopic();

		if (!preFlag) {
			d.m_sstat[tid]++;
			m_docWordTopicStats[docID][tid][wid]++;
//			m_docTopicStats[docID][tid]++;
			if (m_collectCorpusStats) {
				word_topic_sstat[tid][wid]++;
				m_sstat[tid]++;
			}
		} else {
			d.m_sstat[tid]--;
			m_docWordTopicStats[docID][tid][wid]--;
//			m_docTopicStats[docID][tid]--;
			if (m_collectCorpusStats) {
				word_topic_sstat[tid][wid]--;
				m_sstat[tid]--;
			}
		}

	}

	public void calculate_M_step(int iter, File weightFolder) {
		// literally we do not have M-step in Gibbs sampling

		// accumulate p(z|d)
		for (_Doc d : m_trainSet)
			collectStats(d);

		File weightIterFolder = new File(weightFolder, "_" + iter);
		if (!weightIterFolder.exists()) {
			weightIterFolder.mkdir();
		}

		updateParameter(iter, weightIterFolder);

	}

	protected void collectStats(_Doc d) {
		int docID = d.getID();

		for (int k = 0; k < this.number_of_topics; k++) {
			d.m_topics[k] += d.m_sstat[k] + m_alpha[k];
			for (int v = 0; v < vocabulary_size; v++)
				m_docWordTopicProb[docID][k][v] += m_docWordTopicStats[docID][k][v] + m_beta[k][v];
		}

	}

	protected void updateParameter(int iter, File weightIterFolder) {
		updateAlpha();
		
		for(int k=0; k<number_of_topics; k++)
			updateBeta(k);

		String fileName = iter + ".txt";
		saveParameter2File(weightIterFolder, fileName);

	}

	protected void updateAlpha(){
		int i=0;
		double alphaSum, diAlphaSum, c;
		double[] alphaG = new double[number_of_topics];
		double[] alphaQ = new double[number_of_topics];
		
		double deltaAlpha, diff;
		int docSize = m_trainSet.size();
		do{
			deltaAlpha = 0;
			diff = 0;
			Arrays.fill(alphaG, 0);
			Arrays.fill(alphaQ, 0);
			alphaSum = Utils.sumOfArray(m_alpha);
			diAlphaSum = Utils.digamma(alphaSum);
			
			c = docSize*Utils.trigamma(alphaSum);
			
			for(_Doc d:m_trainSet){
				c -= Utils.trigamma(d.getTotalDocLength()+alphaSum);
				
				for(int k=0; k<number_of_topics; k++){
					alphaG[k] += Utils.digamma(d.m_sstat[k]+m_alpha[k])-Utils.digamma(m_alpha[k]);
					alphaG[k] += diAlphaSum-Utils.digamma(alphaSum+d.getTotalDocLength());
					alphaQ[k] += Utils.trigamma(d.m_sstat[k]+m_alpha[k])-Utils.trigamma(m_alpha[k]);
				}
			}
			
			double b1 = 0, b2=0, b=0;
			for(int k=0; k<number_of_topics; k++){
				b1 += alphaG[k]/alphaQ[k];
				b2 += 1.0/alphaQ[k];
			}
			b = b1/((1/c)+b2);
			
			for(int k=0; k<number_of_topics; k++){
				deltaAlpha = (alphaG[k]-b)/alphaQ[k];
				m_alpha[k] -= deltaAlpha;
				diff += deltaAlpha*deltaAlpha;
			}
			
			diff /= number_of_topics;	
			
		}while(++i<m_newtonIter && diff>m_newtonConverge);
	}

	protected void updateBeta(int tid){
		int fSize = m_beta[tid].length;
		int[] iflag = {0}, iprint={-1, 3};
		double[] betaDiag = new double[fSize];
		double[] betaG = new double[fSize];
		double[] t_beta = new double[fSize];
		
		for(int v=0; v<vocabulary_size; v++){
			t_beta[v] = Math.log(m_beta[tid][v]);
		}
		double[] exp_beta = new double[vocabulary_size];
		
		try{
			do{
				Arrays.fill(betaG, 0);
				Arrays.fill(betaDiag, 0);

				double fValue = 0;
				double betaSum = 0;
				
				for(int v=0; v<vocabulary_size; v++){
					exp_beta[v] = Math.exp(t_beta[v]);
					betaSum += Math.exp(t_beta[v]);
				}
				double diBetaSum = Utils.digamma(betaSum);
				
				int docSize = m_trainSet.size();
				
				for(_Doc d:m_trainSet){
					int docID = d.getID();
					for(int v=0; v<vocabulary_size; v++){
						fValue -= Utils.lgamma(exp_beta[v]+m_docWordTopicStats[docID][tid][v])-Utils.lgamma(exp_beta[v]);
						
						betaG[v] -= Utils.digamma(m_docWordTopicStats[docID][tid][v]+exp_beta[v])*exp_beta[v];
						betaG[v] += Utils.digamma(exp_beta[v])*exp_beta[v];
						betaG[v] -= diBetaSum*exp_beta[v]-Utils.digamma(betaSum+d.m_sstat[tid])*exp_beta[v];
						
					}
					fValue += Utils.lgamma(betaSum+d.m_sstat[tid]);
				}
				
				fValue -= docSize*Utils.lgamma(betaSum);
				
				LBFGS.lbfgs(fSize, 4, t_beta, fValue, betaG, false, betaDiag, iprint, 1e-2, 1e-32, iflag);
			}while(iflag[0] != 0);
			for(int v=0; v<vocabulary_size; v++)
				m_beta[tid][v] = Math.exp(t_beta[v]);
		}catch(ExceptionWithIflag e){
			e.printStackTrace();
		}
		
	
	}


	protected double psiValue(double inputArg) {
		double result = 0;
		result = Utils.digamma(inputArg);
		return result;
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

	public double calculate_log_likelihood(_Doc d) {
		double docLogLikelihood = 0.0;

		int docID = d.getID();

		for (int k = 0; k < number_of_topics; k++) {
			double term = Utils.lgamma(d.m_sstat[k] + m_alpha[k]);
			docLogLikelihood += term;

			term = Utils.lgamma(m_alpha[k]);
			docLogLikelihood -= term;

		}

		docLogLikelihood += Utils.lgamma(m_totalAlpha);
		docLogLikelihood -= Utils.lgamma(d.getTotalDocLength() + m_totalAlpha);

		for (int k = 0; k < number_of_topics; k++) {
			for (int v = 0; v < vocabulary_size; v++) {
				double term = Utils.lgamma(m_docWordTopicStats[docID][k][v] + m_beta[k][v]);
				docLogLikelihood += term;

				term = Utils.lgamma(m_beta[k][v]);
				docLogLikelihood -= term;

			}
			docLogLikelihood += Utils.lgamma(m_totalBeta[k]);
			docLogLikelihood -= Utils.lgamma(d.m_sstat[k] + m_totalBeta[k]);
		}

		return docLogLikelihood;
	}

	protected void finalEst() {
		for (_Doc d : m_trainSet) {
			int docID = d.getID();
			for (int i = 0; i < number_of_topics; i++)
				Utils.L1Normalization(m_docWordTopicProb[docID][i]);
			estThetaInDoc(d);
		}

	}

	public void printTopWords(int k, String topWordPath) {
		System.out.println("TopWord FilePath:" + topWordPath);

		Arrays.fill(m_sstat, 0);
		for (_Doc d : m_trainSet) {
			for (int i = 0; i < number_of_topics; i++)
				m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i]) : d.m_topics[i];
		}
		Utils.L1Normalization(m_sstat);

		try {
			PrintWriter topWordWriter = new PrintWriter(new File(topWordPath));

			for (int i = 0; i < m_beta.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j), m_beta[i][j]));

				topWordWriter.format("Topic %d(%.5f):\t", i, m_sstat[i]);
				for (_RankItem it : fVector)
					topWordWriter.format("%s(%.5f)\t", it.m_name, m_logSpace ? Math.exp(it.m_value) : it.m_value);
				topWordWriter.write("\n");
			}
			topWordWriter.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}

}
