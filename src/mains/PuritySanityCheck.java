package mains;

import java.util.ArrayList;
import java.util.Random;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import utils.Utils;

public class PuritySanityCheck {
	int m_method; //similarity calculation method.
	int m_size; // the total number of documents.
	double[] m_similarity;
	_Corpus m_corpus;
	double[][] m_PatK;//m_PatK[0] is for negative class and m_PatK[1] is for positive class.
	double[] m_counts;
	
	public PuritySanityCheck(_Corpus c){
		m_method = 0;
		m_corpus = c;
		m_size = c.getCollection().size();
		m_counts = new double[2];
	}
	
	public PuritySanityCheck(int method, _Corpus c){
		m_method = method;
		m_corpus = c;
		m_size = c.getCollection().size();
		m_counts = new double[2];
	}
	
	//Calculate the similarity in advance based on the selection method.
	public void calculateSimilarity(){
		_Doc di, dj;
		ArrayList<_Doc> documents = m_corpus.getCollection();
		m_similarity = new double[m_size*(m_size-1)/2];
		if(m_method == 1){
			for(int i=1; i<m_size; i++){
				for(int j=0; j<i; j++){
					di = documents.get(i);
					dj = documents.get(j);
					m_similarity[i*(i-1)/2+j] = Utils.calculateSimilarity(di, dj);
				}
			}
		} else if(m_method == 2){
			for(int i=1; i<m_size; i++){
				for(int j=0; j<i; j++){
					di = documents.get(i);
					dj = documents.get(j);
					m_similarity[i*(i-1)/2+j] = Math.exp(-Utils.klDivergence(di.m_topics, dj.m_topics));
				}
			}
		} else if(m_method == 3){
			for(int i=1; i<m_size; i++){
				for(int j=0; j<i; j++){
					di = documents.get(i);
					dj = documents.get(j);
					m_similarity[i*(i-1)/2+j] = Math.exp(Utils.calculateSimilarity(di, dj)
							                   -Utils.klDivergence(di.m_topics, dj.m_topics));
				}
			}
		}
	}
	
	public int getIndex(int i, int j){
		//Swap i and j.
		if(i < j){
			int t = j;
			j = i;
			i = t;
		}
		return i*(i-1)/2+j;
	}
	//Calculate p@k for all the documents.
	public void calculatePatK4All(){
		m_PatK = new double[2][3];
		_Doc tmpD;//the current document.
		ArrayList<_Doc> documents = m_corpus.getCollection();
		int[] neighbors = new int[20];

		for(int i=0; i<m_size; i++){
			tmpD = documents.get(i);
			
			//Select random documents as neighbors.
			if(m_method == 0){
				Random r = new Random();
				for(int j=0; j<20; j++){
					_Doc neighbor = documents.get((int)(r.nextDouble()*m_size));
					neighbors[j] = neighbor.getYLabel();
				}
			} else{//else select neighbors based on similarity.
				MyPriorityQueue<_RankItem> neighborQueue= new MyPriorityQueue<_RankItem>(20);
				//select top k most similar documents as neighbors.
				for(int j=0; j<m_size; j++){
					if(j==i)
						continue;
					else
						neighborQueue.add(new _RankItem(j, m_similarity[getIndex(i, j)]));
				}
				//Traverse the top K neighbors to collect their labels.
				for(int k=0; k<neighborQueue.size(); k++){
					_RankItem tmp = neighborQueue.get(k);
					neighbors[k] = documents.get(tmp.m_index).getYLabel();
				}
			}
			m_PatK[tmpD.getYLabel()][0] += calcPatK(neighbors,5, tmpD.getYLabel());
			m_PatK[tmpD.getYLabel()][1] += calcPatK(neighbors,10, tmpD.getYLabel());
			m_PatK[tmpD.getYLabel()][2] += calcPatK(neighbors,20, tmpD.getYLabel());
			m_counts[tmpD.getYLabel()]++;
		}
		for(int i=0; i<2; i++){
			m_PatK[i][0] = m_PatK[i][0]/m_counts[i];//p@5
			m_PatK[i][1] = m_PatK[i][1]/m_counts[i];//p@10
			m_PatK[i][2] = m_PatK[i][2]/m_counts[i];//p@20
		}
	}
	
	public void printPatK(){
		System.out.format("\tp@5\tp@10\tp@20\n");
		System.out.format("neg:\t%.4f\t%.4f\t%.4f\n", m_PatK[0][0], m_PatK[0][1], m_PatK[0][2]);
		System.out.format("pos:\t%.4f\t%.4f\t%.4f\n", m_PatK[1][0], m_PatK[1][1], m_PatK[1][2]);
	}
	
	//calculate p@k for one document.
	public double calcPatK(int[] labels, int k, int label){
		if(labels.length < k)
			return 0;
		int count = 0;
		for(int i=0; i<k; i++){
			if(labels[i] == label)
				count++;
		}
		return (double)count/k;
	}
}
