/**
 * 
 */
package influence;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import Classifier.BaseClassifier;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import utils.Utils;

/**
 * @author hongning
 * Class for social influence analysis
 * Base class: PageRank based, transition probability estimated via similarities.
 */
public class PageRank extends BaseClassifier {
	
	DoubleMatrix2D m_transition;
	int m_topK; // k nearest neighbors
	
	double[] m_cache; // to store the pre-compute similarities
	int m_N;
	
	int m_maxIter;
	double m_converge;
	double m_alpha; // dumping factor
	
	public PageRank(_Corpus c, double alpha, int topK, int maxIter, double converge) {
		super(c);
		m_alpha = alpha;
		m_topK = topK;
		m_maxIter = maxIter;
		m_converge = converge;
	}

	@Override
	public double train(Collection<_Doc> trainSet) {
		ArrayList<_Doc> graph = new ArrayList<_Doc>();
		
		String lastItemID = null;
		for(_Doc d:trainSet) {
			if (lastItemID == null)
				lastItemID = d.getItemID();
			else if (lastItemID != d.getItemID()) {
				if (graph.size()>10)//otherwise the graph is too small
					calcPageRank(graph);
				graph.clear();
				lastItemID = d.getItemID();
			}
			
			graph.add(d);
		}
		
		//for the last product
		if (graph.size()>5)//otherwise the graph is too small
			calcPageRank(graph);
		return 0;
	}
	
	private void constructSparseGraph(ArrayList<_Doc> collection) {
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
	
	private void constructDenseGraph(ArrayList<_Doc> collection) {
		m_N = collection.size();
		
		//we need to make this very sparse!
		m_transition = new DenseDoubleMatrix2D(m_N, m_N);
		
		double sim;
		
		//construct the connection
		for(int i=0; i<collection.size(); i++) {
			_Doc di = collection.get(i);
			// transition probability is proportion to similarity
			double sum = 0;
			for(int j=0; j<collection.size(); j++) {
				if (i!=j) {
					sim = Math.exp(Utils.calculateSimilarity(di, collection.get(j)));
					m_transition.setQuick(i, j, sim);
					sum += sim;
				}
			}
			
			for(int j=0; j<collection.size(); j++) {
				if (i!=j) {
					sim = m_transition.getQuick(i, j) / sum;
					m_transition.setQuick(i, j, sim);
				}
			}
		}
	}
	
	void calcPageRank(ArrayList<_Doc> collection) {
		if (collection.size()<=m_topK)
			constructDenseGraph(collection);
		else
			constructSparseGraph(collection);
		
		//to save space, we can reuse m_cache
		if (m_cache==null || m_cache.length < 2*m_N)
			m_cache = new double[2*m_N];
		Arrays.fill(m_cache, 1.0/Math.sqrt(m_N));//start from uniform	
		
		int iter = 0;
		double delta = 1.0, influence, prob, norm;
		do {
			norm = 0;
			for(int i=0; i<m_N; i++) {
				influence = 0;
				for(int j=0; j<m_N; j++) {
					if (i!=j && (prob=m_transition.getQuick(j, i))>0) {
						influence += prob * m_cache[j];
					}
				}
				
				m_cache[i+m_N] = m_alpha/m_N + (1-m_alpha) * influence; // influence update
				norm += m_cache[i+m_N] * m_cache[i+m_N];
			}
			
			delta = 0;
			norm = Math.sqrt(norm);
			for(int i=0; i<m_N; i++) {
				m_cache[i+m_N] /= norm; // normalize
				delta += (m_cache[i] - m_cache[i+m_N]) * (m_cache[i] - m_cache[i+m_N]); // difference
				m_cache[i] = m_cache[i+m_N];
			}
			
			delta = Math.sqrt(delta/m_N);			
		} while (++iter<m_maxIter && delta>m_converge);
		
		for(int i=0; i<m_N; i++) {
			collection.get(i).setWeight(1.0 + 10*m_cache[i]);//what would be a reasonable weight setting?
		}
		
//		try {
//			PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter("pagerank_results.txt", true)));
//			writer.format("PageRank in %d*%d graph converge to %.7f after %d steps...\n", m_N, m_N, delta, iter);	
//			int idMax = Utils.maxOfArrayIndex(m_cache,m_N), idMin = Utils.minOfArrayIndex(m_cache,m_N);
//			writer.println(m_cache[idMax]+ "\t" + collection.get(idMax));//print the most typical review
//			writer.println(m_cache[idMin]+ "\t" + collection.get(idMin) + "\n\n");//print the most typical review
//			writer.close();
//			
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
		
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
	public double score(_Doc doc, int label) {
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

	@Override
	protected void debug(_Doc d) {
		// to be implemented
	}

}
