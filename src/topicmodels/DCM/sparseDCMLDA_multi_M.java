package topicmodels.DCM;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._Doc4DCMLDA;
import topicmodels.multithreads.updateParam_worker;
import utils.Utils;

/**
 * Created by jetcai1900 on 12/5/16.
 */
public class sparseDCMLDA_multi_M extends sparseDCMLDA{
    protected updateParam_worker[] m_updateParam_workers;

    public class sparseDCMLDA_multi_mWorker extends updateParam_worker {
		protected ArrayList<double[]> m_param;
		protected ArrayList<Integer> m_paramIndex;

		public sparseDCMLDA_multi_mWorker() {
			super();
			m_param = new ArrayList<double[]>();
			m_paramIndex = new ArrayList<Integer>();
		}

		public void addParameter(double[] a_param, int a_index) {
			int paramLen = a_param.length;
			double[] param = new double[paramLen];
			System.arraycopy(a_param, 0, param, 0, paramLen);
			m_param.add(param);
			m_paramIndex.add(a_index);
		}

		public void clearParameter() {
			for (double[] param : m_param) {
				Arrays.fill(param, 0);
			}
		}

		public void returnParameter(double[] a_param, int a_index) {
			if (a_param.length == m_param.get(a_index).length) {
				System.arraycopy(a_param, 0, m_param.get(a_index), 0,
						a_param.length);
			}
		}

		public void run() {

			for (int i = 0; i < m_paramIndex.size(); i++) {
				calculate_M_step(m_param.get(i), m_paramIndex.get(i));
			}

			for (int i = 0; i < m_paramIndex.size(); i++) {
				int paramLength = m_param.get(i).length;
				int paramIndex = m_paramIndex.get(i);
				System.arraycopy(m_param.get(i), 0, m_beta[paramIndex], 0,
						paramLength);
			}
		}

		public void calculate_M_step(double[] param, int tid) {

			double diff = 0;
			int iteration = 0;
			double smoothingBeta = 0.1;
			double totalBeta = 0;

			do {
				diff = 0;

				double deltaBeta = 0;
				double wordNum4Tid = 0;

				double[] wordNum4Tid4V = new double[vocabulary_size];
				double totalBetaDenominator = 0;
				double[] totalBetaNumerator = new double[vocabulary_size];

				Arrays.fill(totalBetaNumerator, 0);
				Arrays.fill(wordNum4Tid4V, 0);

				totalBeta = Utils.sumOfArray(param);

				for (_Doc d : m_trainSet) {
					_Doc4DCMLDA doc = (_Doc4DCMLDA) d;
					for (int i = 0; i < doc.m_sstat[tid]; i++)
						totalBetaDenominator += 1.0 / (i + totalBeta);

					for (int v = 0; v < vocabulary_size; v++) {
						wordNum4Tid += doc.m_wordTopic_stat[tid][v];
						wordNum4Tid4V[v] += doc.m_wordTopic_stat[tid][v];
						for (int i = 0; i < doc.m_wordTopic_stat[tid][v]; i++) {
							totalBetaNumerator[v] += 1.0 / (i + param[v]);
						}
					}
				}

				for (int v = 0; v < vocabulary_size; v++) {
					if (wordNum4Tid == 0)
						break;
					if (wordNum4Tid4V[v] == 0) {
						deltaBeta = 0;
					} else {
						deltaBeta = totalBetaNumerator[v]
								/ totalBetaDenominator;
					}

					double newBeta = param[v] * deltaBeta + d_beta;
					double t_diff = Math.abs(param[v] - newBeta);
					if (t_diff > diff)
						diff = t_diff;

					param[v] = newBeta;
				}

				iteration++;
				if (iteration > m_newtonIter)
					break;

				System.out.println("beta iteration\t" + iteration);
			} while (diff > m_newtonConverge);

			System.out.println("iteration\t" + iteration);
		}
	}

	public sparseDCMLDA_multi_M(int number_of_iteration, double converge,
			double beta, _Corpus c, double lambda, int number_of_topics,
			double alpha, double burnIn, int lag, int newtonIter,
			double newtonConverge, double tParam, double sParam) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag, newtonIter, newtonConverge, tParam, sParam);
        m_multithread = true;

    }

	public String toString() {
		return String
				.format("sparseDCMLDA_multi_M[k:%d, alphaA:%.2f, beta:%.2f, Gibbs Sampling]",
						number_of_topics, d_alpha, d_beta);
	}

    protected void initialize_probability(Collection<_Doc> collection){
        super.initialize_probability(collection);

        int cores = Runtime.getRuntime().availableProcessors();

        m_threadpool = new Thread[cores];
        m_updateParam_workers = new sparseDCMLDA_multi_mWorker[cores];

        for(int i=0; i<cores; i++)
            m_updateParam_workers[i] = new sparseDCMLDA_multi_mWorker();

        int workerID = 0;
        for(int k=0; k<number_of_topics; k++){
            m_updateParam_workers[workerID%cores].addParameter(m_beta[k], k);
            workerID ++;
        }
    }

	protected void updateBeta(){
		for(int i=0; i<m_updateParam_workers.length; i++){
			m_updateParam_workers[i].setType(updateParam_worker.RunType.RT_EM);
			m_threadpool[i] = new Thread(m_updateParam_workers[i]);
			m_threadpool[i].start();
		}

		for(Thread thread:m_threadpool){
			try{
				thread.join();
			}catch(InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	protected void updateParameter(int iter, File weightFolder) {
		File weightIterFolder = new File(weightFolder, "_" + iter);
		if (!weightIterFolder.exists()) {
			weightIterFolder.mkdir();
		}

		initialAlphaBeta();
		updateAlpha();

		updateBeta();

		for (int k = 0; k < number_of_topics; k++)
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);

		String fileName = iter + ".txt";
		saveParameter2File(weightIterFolder, fileName);

	}
	

}
