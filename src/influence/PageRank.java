/**
 * 
 */
package influence;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import utils.Utils;
import Classifier.BaseClassifier;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;

/**
 * @author hongning
 * Class for social influence analysis
 * Base class: PageRank based, transition probability estimated via similarities.
 */
public class PageRank extends BaseClassifier {
	
	SparseDoubleMatrix2D m_transition;
	int m_topK; // k nearest neighbors
	
	double[] m_cache; // to store the pre-compute similarities
	int m_N;
	
	int m_maxIter;
	double m_converge;
	double m_alpha; // dumping factor
	
	public PageRank(_Corpus c, int class_number, int featureSize, double alpha, int topK, int maxIter, double converge) {
		super(c, class_number, featureSize);
		m_alpha = alpha;
		m_topK = topK;
		m_maxIter = maxIter;
		m_converge = converge;
	}

	@Override
	public void train(Collection<_Doc> trainSet) {
		constructGraph((ArrayList<_Doc>)trainSet);
		
		//to save space, we can reuse m_cache
		if (m_cache==null)
			m_cache = new double[2*m_N];
		Arrays.fill(m_cache, 1.0/m_N);//start from uniform	
		
		int iter = 0;
		double delta = 1.0, influence, prob, norm;
		do {
			norm = 0;
			for(int i=0; i<m_N; i++) {
				influence = 0;
				for(int j=0; j<m_N; j++) {
					if ((prob=m_transition.getQuick(j, i))>0) {
						influence += prob * m_cache[j];
					}
				}
				
				m_cache[i+m_N] = m_alpha/m_N + (1-m_alpha) * influence; // influence update
				norm += m_cache[i+m_N];
			}
			
			delta = 0;
			for(int i=0; i<m_N; i++) {
				m_cache[i+m_N] /= norm; // normalize
				delta += (m_cache[i] - m_cache[i+m_N]) * (m_cache[i] - m_cache[i+m_N]); // difference
				m_cache[i] = m_cache[i+m_N];
			}
			
			delta = Math.sqrt(delta/m_N);
			System.out.format("PageRank converge to %.3f after %d steps...", delta, iter);
		} while (iter<m_maxIter && delta>m_converge);
	}
	
	private void constructGraph(ArrayList<_Doc> collection) {
		m_N = collection.size();
		
		//we need to make this very sparse!
		m_transition = new SparseDoubleMatrix2D(m_N, m_N);
		
		//construct the connection
		MyPriorityQueue<_RankItem> queue = new MyPriorityQueue<_RankItem>(m_topK);
		for(int i=0; i<collection.size(); i++) {
			_Doc di = collection.get(i);
			//find k-nearest neighbor
			for(int j=0; j<collection.size(); j++) {
				if (i!=j)
					queue.add(new _RankItem(j, Utils.calculateSimilarity(di, collection.get(j))));
			}
			
			// transition probability is proportion to similarity
			double sum = 0;
			for(_RankItem item:queue) {
				item.m_value = Math.exp(item.m_value);
				sum += item.m_value;
			}
			
			// set up the transition
			for(_RankItem item:queue)
				m_transition.setQuick(i, item.m_index, item.m_value/sum); // i -> j
			queue.clear();
		}
	}
	
//	//time efficient implementation
//	private void constructGraph(ArrayList<_Doc> collection) {
//		m_N = collection.size();
//		
//		//pre-compute the similarities
//		initCache();
//		for(int i=0; i<collection.size(); i++) {
//			_Doc di = collection.get(i);
//			for(int j=i+1; j<collection.size(); j++)
//				setCache(i, j, Utils.calculateSimilarity(di, collection.get(j)));
//		}
//		
//		//we need to make this very sparse!
//		m_transition = new SparseDoubleMatrix2D(m_N, m_N);
//		
//		//construct the connection
//		MyPriorityQueue<_RankItem> queue = new MyPriorityQueue<_RankItem>(m_topK);
//		for(int i=0; i<collection.size(); i++) {
//			//find k-nearest neighbor
//			for(int j=0; j<collection.size(); j++) {
//				if (i!=j)
//					queue.add(new _RankItem(j, getCache(i, j)));
//			}
//			
//			// transition probability is proportion to similarity
//			double sum = 0;
//			for(_RankItem item:queue) {
//				item.m_value = Math.exp(item.m_value);
//				sum += item.m_value;
//			}
//			
//			// set up the transition
//			for(_RankItem item:queue)
//				m_transition.setQuick(i, item.m_index, item.m_value/sum); // i -> j
//			queue.clear();
//		}
//	}
	
	private void initCache() {
		m_cache = new double[m_N*(m_N-1)/2];
	}
	
	private int getPos(int i, int j) {
		return i*(2*m_N-i-1)/2 + j-i-1;
	}
	
	private double getCache(int i, int j) {
		if (i>j) {
			int t = i;
			i = j;
			j = t;
		}
		return m_cache[getPos(i,j)];
	}
	
	private void setCache(int i, int j, double v) {
		m_cache[getPos(i,j)] = v;
	}

	@Override
	public int predict(_Doc doc) {
		System.err.println("Not implemented yet!");
		System.exit(-1);
		return -1;
	}

	@Override
	protected void init() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void saveModel(String modelLocation) {
		// TODO Auto-generated method stub
		
	}

}
